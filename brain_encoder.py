import torch
import torch.nn as nn
import torch.nn.functional as F

from base_encoders import (BrainNetwork, BrainNetworkEarly, BrainNetworkMulti, 
                          BrainNetworkUp, RidgeRegression)
from MindEyeV2.src.models import BrainDiffusionPrior, PriorNetwork


class BrainEncoder(nn.Module):
    """
    Main brain encoder that processes fMRI voxel data through ridge regression
    and a backbone network to produce CLIP-compatible embeddings.
    
    Args:
        num_voxels_list: List of voxel counts for each subject
        hidden_dim: Hidden dimension size for the ridge regression
        blurry_recon: Whether to enable blurry reconstruction
        n_blocks: Number of transformer blocks in the backbone
        clip_emb_dim: CLIP embedding dimension
        clip_seq_dim: CLIP sequence dimension
        clip_scale: Scale factor for CLIP loss
        interm_out: Whether to output intermediate representations
        enc_version: Encoder architecture version ('v1', 'text', 'up', 'early', 'multi')
    """
    
    def __init__(self, num_voxels_list, hidden_dim, blurry_recon, n_blocks, 
                 clip_emb_dim, clip_seq_dim, clip_scale, interm_out, enc_version='v1', **kwargs):
        super().__init__()
        self.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)
        self.kwargs = kwargs
        
        # Select backbone architecture based on version
        backbone_mapping = {
            'v1': BrainNetwork,
            'text': BrainNetworkText,
            'up': BrainNetworkUp,
            'early': BrainNetworkEarly,
            'multi': BrainNetworkMulti
        }
        
        BN = backbone_mapping.get(enc_version, BrainNetwork)
        if enc_version == 'text':
            self.kwargs.update({'clip_seq_dim': clip_seq_dim})
            
        self.BN = BN
        self.backbone = BN(
            h=hidden_dim, 
            in_dim=hidden_dim, 
            seq_len=1, 
            n_blocks=n_blocks,
            clip_size=clip_emb_dim, 
            out_dim=clip_emb_dim * clip_seq_dim, 
            blurry_recon=blurry_recon, 
            clip_scale=clip_scale, 
            interm_out=interm_out, 
            **self.kwargs
        )
        self.interm_out = interm_out

    def forward(self, voxel_list, subj_list=None):
        """
        Forward pass through the brain encoder.
        
        Args:
            voxel_list: List of voxel tensors for each subject
            subj_list: Optional list of subject indices
            
        Returns:
            Processed brain embeddings
        """
        if subj_list is None:
            voxel_ridge = self.ridge(voxel_list)
        else:
            voxel_ridge_list = [self.ridge(voxel_list[si], si) for si, s in enumerate(subj_list)]
            voxel_ridge = torch.cat(voxel_ridge_list, dim=0)
        return self.backbone(voxel_ridge)


class BrainNetworkText(nn.Module):
    """
    Brain network variant designed for text processing with dual CLIP encoders.
    
    Supports both CLIP-L (768d) and CLIP-G (1280d) text encoders and can operate
    in different output modes (joint, L-only, G-only).
    """
    
    def __init__(self, h=4096, in_dim=15724, out_dim=768, seq_len=2, n_blocks=4, 
                 drop=.15, clip_size=768, blurry_recon=False, clip_scale=1, 
                 interm_out=False, fusion_strategy='', clip_seq_dim=77, 
                 train_pool=False, output_mode='joint'):
        super().__init__()
        self.clip_scale = clip_scale
        self.blurry_recon = blurry_recon
        self.train_pool = train_pool
        self.output_mode = output_mode
        
        if output_mode in ['joint', 'L']:
            clip_emb_dim = 768
            self.bn_L = BrainNetwork(
                h=h, in_dim=h, seq_len=1, n_blocks=n_blocks,
                clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim, 
                blurry_recon=blurry_recon, clip_scale=clip_scale, interm_out=interm_out
            )
            self.bn_L.backbone_linear = None
            self.bn_L.clip_proj = None
            self.backbone_linear_L = nn.Linear(h, 768 * clip_seq_dim)
            self.clip_proj_L = self._projector(768, 768, h=768 * 2)
        
        if output_mode == 'joint' or output_mode == 'BigG':
            clip_emb_dim = 1280
            self.bn_G = BrainNetwork(h=h, in_dim=h, seq_len=1, n_blocks=n_blocks,
                            clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                            blurry_recon=blurry_recon, clip_scale=clip_scale, interm_out=interm_out)
            self.bn_G.backbone_linear = None
            self.bn_G.clip_proj = None
            self.backbone_linear_G = nn.Linear(h, 1280 * clip_seq_dim)
            self.clip_proj_G = self._projector(1280, 1280, h=1280 * 2)
    
        if train_pool: self.pool_attn = nn.Linear(1280, 1280)

        self.n_blocks = n_blocks
        self.fusion_strategy = fusion_strategy
        if fusion_strategy != '': assert output_mode == 'joint'

        if fusion_strategy == 'early':
            self.fusion_linears = nn.ModuleList([nn.Linear(2 * h, 2 * h) for _ in range(n_blocks)])
        elif fusion_strategy == 'mid':
            self.fusion_linears = nn.ModuleList([nn.Linear(2 * h, 2 * h) for _ in range(n_blocks // 2)])
        elif fusion_strategy == 'late':
            self.fusion_linears = nn.Linear(2 * h, 2 * h)
        else:
            self.fusion_linears = None

    def _projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )

    def _get_block_out(self, x, residual1, residual2, block1, block2):
        x = block1(x) + residual1
        residual1 = x
        x = x.permute(0,2,1)
        x = block2(x) + residual2
        residual2 = x
        x = x.permute(0,2,1)
        return x, residual1, residual2

    def forward(self, x):   
        # Mixer blocks
        x_L, x_G = x, x
        residual1_L, residual1_G = x_L, x_G
        residual2_L, residual2_G = x_L.permute(0,2,1), x_G.permute(0,2,1)
        for i in range(self.n_blocks):
            if self.output_mode == 'joint' or self.output_mode == 'L':
                x_L, residual1_L, residual2_L = self._get_block_out(x_L, residual1_L, residual2_L, self.bn_L.mixer_blocks1[i], self.bn_L.mixer_blocks2[i])
            if self.output_mode == 'joint' or self.output_mode == 'BigG':
                x_G, residual1_G, residual2_G = self._get_block_out(x_G, residual1_G, residual2_G, self.bn_G.mixer_blocks1[i], self.bn_G.mixer_blocks2[i])

            if self.fusion_strategy == 'early':
                fused_x = self.fusion_linears[i](torch.cat([x_L, x_G], dim=-1))
                x_L, x_G = fused_x[:, :, :x_L.size(-1)], fused_x[:, :, x_L.size(-1):]
            elif self.fusion_strategy == 'mid' and i >= self.n_blocks // 2:
                fused_x = self.fusion_linears[i - (self.n_blocks // 2)](torch.cat([x_L, x_G], dim=-1))
                x_L, x_G = fused_x[:, :, :x_L.size(-1)], fused_x[:, :, x_L.size(-1):]
            elif self.fusion_strategy == 'late' and i == self.n_blocks - 1:
                fused_x = self.fusion_linears(torch.cat([x_L, x_G], dim=-1))
                x_L, x_G = fused_x[:, :, :x_L.size(-1)], fused_x[:, :, x_L.size(-1):]
        
        if self.output_mode == 'joint' or self.output_mode == 'L':
            backbone_L = self.backbone_linear_L(x_L).reshape(len(x), -1, 768)
            clip_L = self.clip_proj_L(backbone_L)
        if self.output_mode == 'joint' or self.output_mode == 'BigG':
            backbone_G = self.backbone_linear_G(x_G).reshape(len(x), -1, 1280)
            clip_G = self.clip_proj_G(backbone_G)
        
        if self.output_mode == 'joint':
            backbone = torch.cat([backbone_L, backbone_G], dim=-1)
            clip = torch.cat([clip_L, clip_G], dim=-1)
        elif self.output_mode == 'L':
            backbone = backbone_L
            clip = clip_L
        else:
            backbone = backbone_G
            clip = clip_G
        
        if self.train_pool:
            attn = F.softmax(self.pool_attn(clip_G), dim=1).mean(-1, keepdim=True)  # token
            pooled = (clip_G * attn).sum(dim=1)
        else:
            pooled = 0

        if self.blurry_recon:
            raise Exception("Blurry reconstruction not implemented for text encoder")

        return backbone, clip, pooled
    
    
class BrainDiffusionPriorEncoder(nn.Module):
    def __init__(self, out_dim, depth, dim_head, heads, clip_seq_dim, timesteps, clip_seq_dim_target=None):
        super().__init__()
        if clip_seq_dim_target is None: clip_seq_dim_target = clip_seq_dim
        prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = clip_seq_dim,
            num_tokens_target = clip_seq_dim_target,
            learned_query_mode="pos_emb"
        )
        self.prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )
    def forward(self, text_embed, image_embed):
        return self.prior(text_embed=text_embed, image_embed=image_embed)

class BottleNeck(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BottleNeck, self).__init__()
        self.ln = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.ln(x)