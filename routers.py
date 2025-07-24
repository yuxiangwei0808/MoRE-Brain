import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertRouter(nn.Module):
    """Router for expert-choice of embeddings"""
    def __init__(self, num_exp, n_emb, capacity_factor):
        super().__init__()
        self.router = nn.Linear(n_emb, num_exp, bias=False)
        nn.init.uniform_(self.router.weight, -1e-4, 1e-4)  # Tiny initial weights
        
        self.capacity_factor = capacity_factor
        self.num_exp = num_exp

    def forward(self, x, training=True):
        # B D L
        x = x.transpose(-1, -2)
        B, D, H = x.size()
        router_logit = self.router(x)
        router_logit = F.softmax(router_logit, dim=-1)

        if training:
            noise = torch.randn_like(router_logit) * 0.001
            router_logit = router_logit + noise.to(router_logit.device)
        
        top_k = int(D / self.num_exp * self.capacity_factor)
        weights, topk_indices = torch.topk(router_logit, top_k, dim=1, sorted=False)
        _, k, E = topk_indices.shape
        
        lb_loss = self.load_balancing_loss(router_logit, topk_indices, D, top_k)
        
        expanded_tokens = x.unsqueeze(2).expand(B, D, E, H)  # [B, D, E, H]
        expanded_indices = topk_indices.unsqueeze(-1).expand(B, k, E, H)  # [B, k, E, H]
        selected_tokens = torch.gather(expanded_tokens, dim=1, index=expanded_indices)  # [B, k, E, H]
        # (E B H k), (E B k), (E, B, k)
        return selected_tokens.permute(2, 0, 3, 1), weights.permute(2, 0, 1), topk_indices.permute(2, 0, 1), lb_loss

    def load_balancing_loss(self, router_probs, topk_indices, D, top_k):
        expert_mask = F.one_hot(topk_indices, num_classes=D).float()  # [B, k, E, D]
        expert_usage = expert_mask.sum(dim=1).sum(dim=0)  # [E, D] -> sum to [E]
        
        # Normalize and compare to uniform distribution
        expert_usage = expert_usage / (expert_mask.shape[0] * top_k + 1e-9)  # Normalize 0-1
        uniform = torch.ones_like(expert_usage) / self.num_exp
        loss = F.mse_loss(expert_usage, uniform)
        return loss


class CLIPMoERouter(nn.Module):
    def __init__(self, seq_len, n_emb, num_exp, weight_type='rand', **kwargs):
        super().__init__()
        self.weight_type = weight_type
        if weight_type == 'ones':
            self.router = nn.Parameter(torch.ones(num_exp, seq_len))
        elif weight_type == 'rand':
            self.router = nn.Parameter(torch.randn(num_exp, seq_len))
        elif weight_type == 'query':
            self.router = nn.Linear(n_emb, 1)

    def forward(self, x, temp=1, **kwargs):
        # E B k D
        E, B, k, D = x.size()

        if self.weight_type == 'query':
            score = self.router(x.permute(1, 2, 0, 3).reshape(B * k, E, D))
            score = F.softmax(score.reshape(B, k, E) / temp, dim=-1)
            return torch.einsum('bke,ebkd->bkd', score, x)
        else:
            return (x * self.router.view(E, 1, k, 1)).sum(0)


class DiffuseSpaceRouterGate(nn.Module):
    """Embedding to diffusion (qkv condition) router via gating mechanism"""
    def __init__(self, num_exp, n_emb, mode='gumbel', temp=1, scale_output=False, **kwargs):
        super().__init__()
        self.gate = nn.Linear(n_emb, num_exp, bias=False)
        nn.init.constant_(self.gate.weight, 1 / num_exp)
        self.mode = mode
        self.temp = temp
        self.scale_output = scale_output

    def forward(self, expert_embeddings, **kwargs):
        # E B L D
        E, B, L, D = expert_embeddings.size()

        gate_input = expert_embeddings.mean(0)
        gate_logits = self.gate(gate_input)

        if self.mode == 'sigmoid':
            gate_weights = torch.sigmoid(gate_logits)
        elif self.mode == 'softmax':
            gate_weights = F.softmax(gate_logits / self.temp, dim=-1) # (B, L, E)
            if self.scale_output:
                gate_weights = gate_weights * self.num_exp
        elif self.mode == 'gumbel':
            gate_weights = F.gumbel_softmax(gate_logits, tau=self.temp, dim=-1)
            if self.scale_output:
                gate_weights = gate_weights * self.num_exp
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
        gate_weights = gate_weights.permute(2, 0, 1).unsqueeze(-1)  # (E, B, L, 1)

        final_embedding = (expert_embeddings * gate_weights).sum(dim=0)  # (B, L, D)
        return final_embedding

DiffuseSpaceRouterWeight = CLIPMoERouter

class DiffuseSpaceRouterSelfAttn(nn.Module):
    """Embedding to diffusion (qkv condition) router via self-attention"""
    def __init__(self, num_exp, n_emb, temp=1, scale_output=False, **kwargs):
        super().__init__()
        self.query_linear = nn.Linear(n_emb, n_emb)
        self.key_linear = nn.Linear(n_emb, n_emb)
        self.scale_output = scale_output
        self.temp = temp

    def forward(self, experts_embedding, **kwargs):
        E, B, L, D = experts_embedding.shape
        
        queries = self.query_linear(experts_embedding.mean(0))  # (B, L, D)
        keys = self.key_linear(experts_embedding)  # (E, B, L, D)

        # Attention scores: (E, B, L)
        attn_scores = (keys * queries.unsqueeze(0)).sum(dim=-1) / (D ** 0.5)

        gate_weights = F.softmax(attn_scores / self.temp, dim=0).unsqueeze(-1)  # (E, B, L, 1)
        if self.scale_output:
            gate_weights = gate_weights * E

        final_embedding = (experts_embedding * gate_weights).sum(dim=0)  # (B, L, D)
        return final_embedding, gate_weights
    
    
class DiffuseSpaceRouterCrossAttn(nn.Module):
    """Cross-attention router that manually implements the attention mechanism."""
    def __init__(self, n_emb, temp=1, dropout=0.0, scale_output=False, learn_query=False, **kwargs):
        super().__init__()
        self.embedding_dim = n_emb
        self.scale_output = scale_output
        self.temp = temp
        self.q_proj = nn.Linear(4096, n_emb, bias=False)
        
        # Add key and value projections
        self.k_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.v_proj = nn.Linear(n_emb, n_emb, bias=False)
        
        if learn_query:
            self.query = nn.Parameter(torch.randn(1, 1, 4096))
        
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(n_emb)

    def forward(self, experts_embedding, query=None, need_weights=False):
        E, B, L_key, D = experts_embedding.shape

        if query is None: query = self.query.repeat(B, 1, 1)  # (B, L_query, D)

        query = self.q_proj(query.flatten(2)).mean(1, keepdim=True)  # (B, L_query, D)
        Bq, L_query, Dq = query.shape
        
        if B != Bq or D != Dq:
            raise ValueError(f"Batch or dimension mismatch: query({Bq},{Dq}) vs experts({B},{D})")
        
        kv = experts_embedding.permute(1, 0, 2, 3).reshape(B, E * L_key, D)
        
        keys = self.k_proj(kv)      # (B, E*L_key, D)
        values = self.v_proj(kv)    # (B, E*L_key, D)
        
        # (B, L_query, D) x (B, D, E*L_key) -> (B, L_query, E*L_key)
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores / self.temp, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # (B, L_query, E*L_key) x (B, E*L_key, D) -> (B, L_query, D)
        attn_output = torch.bmm(attn_weights, values) + experts_embedding.mean(0)
        
        return attn_output, attn_weights


class DiffuseTimeRouterSimple:
    def __init__(self, num_granularity_levels: int = 3, **kwargs):
        self.num_granularity_levels = num_granularity_levels

    def __call__(self, timestep: torch.Tensor, total_steps: int = 1000, **kwargs):
        """
        Simple time router that selects granularity based on timestep, from coarse to fine.
        """
        normalized_timestep = timestep / total_steps

        # Divide the timestep range into equal segments based on granularity levels
        # Early timesteps (closer to 1.0) use coarser levels (lower indices)
        # Later timesteps (closer to 0.0) use finer levels (higher indices)
        segment_size = 1.0 / self.num_granularity_levels
        
        # Calculate which segment the timestep falls into
        selected_level = torch.clamp(
            ((1.0 - normalized_timestep) / segment_size).long(),
            0,
            self.num_granularity_levels - 1
        )
        return selected_level
    

class DiffuseTimeRouterMLP(nn.Module):
    def __init__(
        self,
        num_granularity_levels: int = 3,
        embedding_dim: int = 320,  # 1280
        hidden_dim: int = 768,
        num_layers: int = 3,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.num_granularity_levels = num_granularity_levels
        input_dim = embedding_dim
        
        # MLP to map timestep embedding to granularity selection probabilities
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.output_layer = nn.Linear(hidden_dim, num_granularity_levels)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.temperature = temperature

    def forward(self, t_emb: torch.Tensor, training=True, **kwargs) -> torch.Tensor:       
        combined_emb = t_emb
        
        # Process through MLP layers with residual connections
        x = combined_emb
        for i, layer in enumerate(self.layers):
            if i > 0:  # Apply residual connections after first layer
                x = x + layer(self.layer_norm(x))
            else:
                x = F.gelu(layer(x))
                
        # Calculate granularity probabilities
        logits = self.output_layer(x)
        granularity_probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Get the granularity level with highest probability 
        # (during inference can use argmax, during training can sample)
        if training:
            # During training, use Gumbel-Softmax to allow differentiable sampling
            selected_granularity = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            selected_idx = torch.argmax(selected_granularity, dim=-1)
        else:
            # During inference, just use argmax
            selected_idx = torch.argmax(granularity_probs, dim=-1)
            
        return granularity_probs, selected_idx


class DiffuseTimeRouterAttn(nn.Module):
    """
    Time Router that uses single-head attention with temperature scaling to dynamically 
    select granularity levels based on diffusion timestep and optional context.
    """
    
    def __init__(
        self,
        num_granularity_levels: int = 3,  # e.g., coarse, medium, fine
        hidden_dim: int = 768,
        embedding_dim: int = 320,
        dropout: float = 0.,
        context_dim: int = 784,
        use_context: bool = False,
        context_as_key: bool = False,
        temperature: float = 1.5,  # Temperature parameter for attention scaling
        **kwargs
    ):
        super().__init__()
        
        self.num_granularity_levels = num_granularity_levels
        self.hidden_dim = hidden_dim
        self.context_as_key = context_as_key
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

        if self.context_as_key:
            assert context_dim is not None, "Context dimension must be provided for using context as key"
        
        # Learnable granularity level embeddings
        self.granularity_embeddings = nn.Parameter(
            torch.randn(num_granularity_levels, hidden_dim)
        )
        self.timestep_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Optional context projection
        self.use_context = use_context
        if self.use_context:
            self.context_proj = nn.Linear(context_dim, hidden_dim)
        
    def forward(
        self, 
        t_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_step: float = None,
        multiply_prior: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = t_emb.shape[0]
        
        # 1. Embed the timestep
        query = self.timestep_proj(t_emb)  # [batch_size, hidden_dim]
        
        # 2. Incorporate context if provided
        if self.use_context and context is not None:
            context = context.flatten(2).mean(1)  # [batch_size, context_dim]
            context_emb = self.context_proj(context)  # [batch_size, hidden_dim]
            if not self.context_as_key: 
                query = query + context_emb
            
        # 3. Prepare key/values from granularity embeddings
        if self.context_as_key:
            key_value = context_emb.unsqueeze(1).repeat(1, self.num_granularity_levels, 1)
            key_value += self.granularity_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            key_value = self.granularity_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        # Shape: [batch_size, num_granularity_levels, hidden_dim]
        
        # 4. Manual single-head attention implementation
        # Calculate attention scores: dot product between query and keys
        # query shape: [batch_size, hidden_dim]
        # key_value shape: [batch_size, num_granularity_levels, hidden_dim]
        
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        attn_scores = torch.bmm(query, key_value.transpose(1, 2))  # [batch_size, 1, num_granularity_levels]
        attn_scores = attn_scores / (math.sqrt(self.hidden_dim) * self.temperature)        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, 1, num_granularity_levels]
        attn_weights = self.dropout(attn_weights)
        
        # Calculate weighted sum of values
        # attn_output = torch.bmm(attn_weights, key_value)  # [batch_size, 1, hidden_dim]
        
        attn_weights = attn_weights.squeeze(1)  # [batch_size, num_granularity_levels]
        if multiply_prior:
            gaussian_prior = self._gaussian_target_distribution(self.num_granularity_levels, time_step)
            attn_weights = attn_weights * gaussian_prior
        
        selection_result = torch.argmax(attn_weights, dim=-1)  # [batch_size]
        
        return attn_weights, selection_result

    @torch.no_grad()
    def _gaussian_target_distribution(self, N, i, sigma=1.0):
        """
        Create a probability distribution over N targets using a Gaussian centered at the i-th target.
        
        Args:
            N (int): Number of targets
            i (float or torch.Tensor): Index of the target at which the Gaussian is centered
                                    Can be fractional (e.g., 2.5 centers between targets 2 and 3)
            sigma (float): Standard deviation of the Gaussian
            
        Returns:
            torch.Tensor: Probability distribution over N targets, summing to 1
        """
        # Create target indices from 0 to N-1
        indices = torch.arange(N, dtype=torch.float32).to(i.device)

        if i.dim() == 0:
            i = i.unsqueeze(0)  # Ensure i is a tensor
        
        # Handle the case where i is a batch of indices
        if isinstance(i, torch.Tensor) and i.dim() > 0:
            # Reshape for broadcasting
            indices = indices.view(*[1 for _ in range(i.dim())], N)
            i = i.unsqueeze(-1)

        i_clamped = torch.clamp(i, 250, 750)
        target_center = (N - 1) * (1 - (i_clamped - 250) / 500)
        
        # Compute Gaussian values for each index
        gaussian_values = torch.exp(-0.5 * ((indices - target_center) / sigma) ** 2)
        
        # Normalize to get a proper probability distribution
        sum_dim = -1 if isinstance(i, torch.Tensor) and i.dim() > 0 else 0
        probabilities = gaussian_values / gaussian_values.sum(dim=sum_dim, keepdim=True)
        
        return probabilities


class DiffuseRouter(nn.Module):
    def __init__(
        self,
        time_router_class,
        space_router_class,
        num_granularity_levels: int = 3,
        num_experts_per_granularity: List[int] = [4, 8, 16],  # Each granularity can have different number of experts
        embedding_dim: int = 1280,
        time_router_hidden_dim: int = 768,
        soft_time_routing: bool = True,
        time_router_kwargs: dict = {},
        space_router_kwargs: dict = {},
        time_embedder: Optional[nn.Module] = None,
        enable_time=True,
        enable_space=False,
    ):
        super().__init__()
        self.num_granularity_levels = num_granularity_levels
        self.num_experts_per_granularity = num_experts_per_granularity
        self.embedding_dim = embedding_dim
        self.soft_time_routing = soft_time_routing
        self.enable_time = enable_time
        self.enable_space = enable_space
        
        if enable_time:
            self.time_router = time_router_class(
                num_granularity_levels=num_granularity_levels,
                embedding_dim=320,  # Dimension for timestep embedding
                hidden_dim=time_router_hidden_dim,
                temperature=1,
                **time_router_kwargs
            )

        if enable_space:
            self.space_routers = nn.ModuleList()
            for num_experts in num_experts_per_granularity:
                # Each space router will handle attention-based routing between experts # at a specific granularity level
                self.space_routers.append(space_router_class(
                    num_exp=num_experts,
                    n_emb=embedding_dim,
                    seq_len=1,
                    **space_router_kwargs
                ))

    def forward(
        self, 
        time_emb: torch.Tensor,
        expert_embeddings: List[torch.Tensor],
        time_step: int,
        total_steps: int,
        context_embedding: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """
        expert_embeddings: List, where each element contains embeddings from experts at a specific granularity level with the shape E*B*L*D
        """
        if self.enable_time:
            if isinstance(self.time_router, DiffuseTimeRouterSimple):
                assert not self.soft_time_routing, "Simple time router does not support soft routing"
                selected_granularity = self.time_router(time_step, total_steps)
            else:
                granularity_probs, selected_granularity = self.time_router(time_emb, training=training, context=context_embedding, time_step=time_step)
        else:
            assert self.soft_time_routing
            granularity_probs = torch.ones(time_emb.shape[0], self.num_granularity_levels, device=time_emb.device) / self.num_granularity_levels

        batch_size = expert_embeddings[0].shape[-3]
        L, D = expert_embeddings[0].shape[-2:]

        if self.soft_time_routing:
            # Process all granularity levels for the entire batch
            final_embeddings = torch.zeros(batch_size, L, D, device=expert_embeddings[0].device)
            
            # Apply each space router to its corresponding granularity level
            for gran_idx in range(self.num_granularity_levels):
                assert gran_idx < len(expert_embeddings), "Granularity index exceeds number of expert embeddings"
                
                if self.enable_space:
                    routed_embeddings, _ = self.space_routers[gran_idx](expert_embeddings[gran_idx], query=context_embedding)
                else:
                    if expert_embeddings[gran_idx].dim() == 4:
                        routed_embeddings = expert_embeddings[gran_idx].sum(0)
                    else:
                        routed_embeddings = expert_embeddings[gran_idx]
                # Weight by granularity probabilities and add to final result
                final_embeddings += routed_embeddings * granularity_probs[:, gran_idx].view(batch_size, 1, 1)
        else:
            assert max(selected_granularity) < len(expert_embeddings), "Selected granularity index exceeds number of expert embeddings"
            final_embeddings = torch.zeros(batch_size, L, D, device=expert_embeddings[0].device)
            for b in range(batch_size):
                sg = selected_granularity[b].item()
                if self.enable_space:
                    routed_embeddings, _ = self.space_routers[sg](expert_embeddings[sg][:, b:b+1], query=context_embedding[b:b+1] if context_embedding is not None else None)
                else:
                    if expert_embeddings[sg].dim() == 4:
                        routed_embeddings = expert_embeddings[sg][:, b:b+1].sum(0)
                    else:
                        routed_embeddings = expert_embeddings[sg][b:b+1]
                final_embeddings[b:b+1] = routed_embeddings
        
        return final_embeddings
