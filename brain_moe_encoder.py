import torch
import torch.nn as nn

from brain_encoder import BrainEncoder, RidgeRegression, BottleNeck
from base_encoders import BrainNetwork
from routers import *


class BrainMoE(BrainEncoder):
    """Each expert produce CLIP-space embeddings"""
    def __init__(self, num_voxels_list, hidden_dim, blurry_recon, n_blocks, clip_emb_dim, clip_seq_dim, clip_scale, interm_out=False, enc_version='v1',
                 num_exp=4, capacity_factor=1, meta=False):
        super().__init__(num_voxels_list, hidden_dim, blurry_recon, n_blocks, clip_emb_dim, clip_seq_dim, clip_scale, interm_out, enc_version)
        self.expert_router = ExpertRouter(num_exp=num_exp, n_emb=1, capacity_factor=capacity_factor)

        self.experts = nn.ModuleList([self.BN(h=int(hidden_dim // num_exp * capacity_factor), in_dim=int(hidden_dim // num_exp * capacity_factor), seq_len=1, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale, interm_out=interm_out, **self.kwargs) for _ in range(num_exp)])
        if meta:
            self.meta_exp = self.BN(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks, clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale, interm_out=interm_out, **self.kwargs)
        else:
            self.meta_exp = None
        del self.backbone

        # self.clip_router = CLIPMoERouter(clip_seq_dim, clip_emb_dim, num_exp, 'rand')

    def forward_feat(self, voxel_list, subj_list, training=True):
        voxel_ridge_list = [self.ridge(voxel_list[si],si) for si, s in enumerate(subj_list)]
        voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

        selected_tokens, weights, topk_indices, lb_loss = self.expert_router(voxel_ridge, training)

        backbone_out, clip_out, blurry_out = [], [], []
        for expert, x, w in zip(self.experts, selected_tokens, weights):
            out = expert(x * w.unsqueeze(1))
            backbone_out.append(out[0])
            clip_out.append(out[1])
            blurry_out.append(out[2])
        backbone_out = torch.stack(backbone_out, dim=0)
        clip_out = torch.stack(clip_out, dim=0)
        return backbone_out, clip_out, blurry_out, lb_loss

    def forward(self, voxel_list, subj_list, training=True):
        backbone_out, clip_out, blurry_out, lb_loss = self.forward_feat(voxel_list, subj_list, training)
        # backbone_out = self.clip_router(backbone_out)
        # clip_out = self.clip_router(clip_out)

        if self.meta_exp:
            backbone_out_meta, clip_out_meta, _ = self.meta_exp(voxel_ridge)
            backbone_out += backbone_out_meta
            clip_out += clip_out_meta

        if blurry_out[0]: blurry_out = torch.stack(blurry_out, dim=0).sum(0)
        return backbone_out, clip_out, blurry_out, lb_loss

######################## Hierarchical MOE ##############################

class BrainMoEHier(BrainEncoder):
    def __init__(self, num_voxels_list, hidden_dim, blurry_recon, n_blocks, clip_emb_dim, clip_seq_dim, clip_scale, interm_out=False, enc_version='v1',
                 num_exps=[4, 8, 16, 32], capacity_factors=[1, 1, 1, 1], meta=False):
        super().__init__(num_voxels_list, hidden_dim, blurry_recon, n_blocks, clip_emb_dim, clip_seq_dim, clip_scale, interm_out, enc_version)

        num_hier = len(num_exps)
        self.expert_routers = nn.ModuleList([ExpertRouter(num_exp=num_exp, n_emb=1, capacity_factor=capacity_factor) for num_exp, capacity_factor in zip(num_exps, capacity_factors)])
        
        self.experts = nn.ModuleList([(nn.ModuleList([self.BN(h=int(hidden_dim // num_exp * capacity_factor), in_dim=int(hidden_dim // num_exp * capacity_factor), seq_len=1, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale, interm_out=interm_out, **self.kwargs) for _ in range(num_exp)]))
                          for num_exp, capacity_factor in zip(num_exps, capacity_factors)])
        self.experts = nn.ModuleList(self.experts)

        if meta:
            self.meta_exp = self.BN(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks, clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale, interm_out=interm_out, **self.kwargs)
        else:
            self.meta_exp = None
        del self.backbone

        # self.b_expert_weight = nn.Parameter(torch.ones(num_hier), requires_grad=True)
        # self.c_expert_weight = nn.Parameter(torch.ones(num_hier), requires_grad=True)

        self.back_router = nn.ModuleList([CLIPMoERouter(clip_seq_dim, clip_emb_dim, ne, 'rand') for ne in num_exps])
        self.clip_router = nn.ModuleList([CLIPMoERouter(clip_seq_dim, clip_emb_dim, ne, 'rand') for ne in num_exps])

    def forward(self, voxel_list, subj_list, training=True):
        voxel_ridge_list = [self.ridge(voxel_list[si],si) for si, s in enumerate(subj_list)]
        voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

        backbone_out_list, clip_out_list, lb_loss_list = [], [], []
            
        for i, expert_router in enumerate(self.expert_routers):
            selected_tokens, weights, topk_indices, lb_loss = expert_router(voxel_ridge, training)

            backbone_out, clip_out, blurry_out = [], [], []
            for expert, x, w in zip(self.experts[i], selected_tokens, weights):
                out = expert(x * w.unsqueeze(1))
                backbone_out.append(out[0])
                clip_out.append(out[1])
                blurry_out.append(out[2])
            # backbone_out = torch.stack(backbone_out, dim=0).sum(0)
            # clip_out = torch.stack(clip_out, dim=0).sum(0)

            backbone_out = self.back_router[i](torch.stack(backbone_out, dim=0)) 
            clip_out = self.clip_router[i](torch.stack(clip_out, dim=0))

            backbone_out_list.append(backbone_out)
            clip_out_list.append(clip_out)
            lb_loss_list.append(lb_loss)
        
        # Weighted Hierarchical MoE
        # backbone_out_list = torch.einsum('e,eblh->eblh', self.b_expert_weight, torch.stack(backbone_out_list, dim=0)).mean(0)
        # clip_out_list = torch.einsum('e,eblh->eblh', self.c_expert_weight, torch.stack(clip_out_list, dim=0)).mean(0)
        backbone_out_list = torch.stack(backbone_out_list, dim=0)
        clip_out_list = torch.stack(clip_out_list, dim=0)

        if self.meta_exp:
            backbone_out_meta, clip_out_meta, _ = self.meta_exp(voxel_ridge)
            backbone_out_list = torch.cat((backbone_out_list, backbone_out_meta.unsqueeze(0)), dim=0)
            clip_out_list = torch.cat((clip_out_list, clip_out_meta.unsqueeze(0)), dim=0)
            del backbone_out_meta, clip_out_meta

        backbone_out_list = backbone_out_list.mean(0)
        clip_out_list = clip_out_list.mean(0)
        
        lb_loss = torch.stack(lb_loss_list).mean()

        return backbone_out_list, clip_out_list, None, lb_loss



class BrainMoEMulti(nn.Module):
    """First MoE that decode betas to the CLIP space, then further MoEs to refine each expert's embeddings
    Align multiple embeddings from different granularity of MoE
    """
    def __init__(self, num_voxels_list, hidden_dim, blurry_recon, n_blocks, clip_emb_dim, clip_seq_dim, clip_scale, interm_out, 
        num_exp_0, capacity_factor_0, num_exp_layer, exp_factor_list, cap_fac_list, enc_version='v1', routing=False, b_size=-1):
        super().__init__()
        self.clip_seq_dim, self.clip_emb_dim = clip_seq_dim, clip_emb_dim
        self.num_exp_layer = num_exp_layer
        self.num_exp_0 = num_exp_0
        self.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)

        if b_size > 0:
            assert b_size >= 16
            self.bottleneck = BottleNeck(hidden_dim, b_size)
            hidden_dim = b_size

        self.router_0 = ExpertRouter(num_exp=num_exp_0, n_emb=1, capacity_factor=capacity_factor_0)
        self.experts_0 = []
        self.routing = routing

        hidden_dim = int(hidden_dim // num_exp_0 * capacity_factor_0)
        for _ in range(num_exp_0):
            net = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale, interm_out=interm_out)
            del net.backbone_linear, net.clip_proj
            self.experts_0.append(net)
        self.experts_0 = nn.ModuleList(self.experts_0)
        self.backbone_proj_0 = nn.ModuleList([nn.Linear(hidden_dim, clip_emb_dim * clip_seq_dim) for _ in range(num_exp_0)])
        self.clip_proj_0 = nn.ModuleList([self.projector(clip_emb_dim, clip_emb_dim, h=clip_emb_dim * 2) for _ in range(num_exp_0)])
        
        self.router_list, self.expert_list = [], []
        self.b_proj_list, self.c_proj_list = [], []
        self.back_router, self.clip_router = [], []
        assert num_exp_layer == len(cap_fac_list) == len(exp_factor_list)
        num_exp = num_exp_0
        for i in range(num_exp_layer):
            capacity_factor = cap_fac_list[i]
            router = nn.ModuleList([ExpertRouter(num_exp=exp_factor_list[i], n_emb=1, capacity_factor=capacity_factor) for _ in range(num_exp)])
            num_exp *= exp_factor_list[i]  # each expert output from the previous layer receives exp_factor experts
            hidden_dim = int(hidden_dim // exp_factor_list[i] * capacity_factor)
            experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_exp)])
            
            self.router_list.append(router)
            self.expert_list.append(experts)
            self.b_proj_list.append(nn.ModuleList([nn.Linear(hidden_dim, clip_emb_dim * clip_seq_dim) for _ in range(num_exp)]))
            self.c_proj_list.append(nn.ModuleList([self.projector(clip_emb_dim, clip_emb_dim, clip_emb_dim * 2) for _ in range(num_exp)]))
            
            # self.back_router.append(DiffuseSpaceRouterSelfAttn(num_exp, clip_emb_dim))
            # self.clip_router.append(DiffuseSpaceRouterSelfAttn(num_exp, clip_emb_dim))
            # self.back_router.append(DiffuseSpaceRouterCrossAttn(clip_emb_dim, learn_query=True))
            # self.clip_router.append(DiffuseSpaceRouterCrossAttn(clip_emb_dim, learn_query=True))

        self.router_list = nn.ModuleList(self.router_list)
        self.expert_list = nn.ModuleList(self.expert_list)
        self.b_proj_list = nn.ModuleList(self.b_proj_list)
        self.c_proj_list = nn.ModuleList(self.c_proj_list)
        
        self.back_router = nn.ModuleList(self.back_router)
        self.clip_router = nn.ModuleList(self.clip_router)

    def projector(self, in_dim, out_dim, h):
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

    def forward(self, voxel_list, subj_list=[1], training=True, return_exp_out=False, capture_routing=False):
        voxel_list = [voxel_list] if isinstance(voxel_list, torch.Tensor) else voxel_list
        voxel_ridge_list = [self.ridge(voxel_list[si],si) for si, s in enumerate(subj_list)]
        voxel_ridge = torch.cat(voxel_ridge_list, dim=0)
        B = voxel_ridge.size(0)

        if hasattr(self, 'bottleneck'):
            voxel_ridge = self.bottleneck(voxel_ridge)

        selected_tokens, weights, topk_indices, lb_loss = self.router_0(voxel_ridge, training)
        exp_out = [expert.forward_feat(selected_tokens[i] * weights[i].unsqueeze(1))[0] for i, expert in enumerate(self.experts_0)]
        backbone_out_0 = [self.backbone_proj_0[i](exp_out[i].flatten(1)).reshape(-1, self.clip_seq_dim, self.clip_emb_dim) for i in range(len(exp_out))]
        clip_out_0 = [self.clip_proj_0[i](backbone_out_0[i]) for i in range(len(exp_out))]

        if capture_routing:
            routing_info = {}
            routing_info['layer_0'] = {
                'topk_indices': topk_indices.detach().cpu(),  # E, B, k
                'weights': weights.detach().cpu(),            # E, B, k
            }

        if not return_exp_out:
            backbone_out_total, clip_out_total = torch.zeros(self.num_exp_layer + 1, B, self.clip_seq_dim, self.clip_emb_dim, device=voxel_ridge.device), torch.zeros(self.num_exp_layer + 1, B, self.clip_seq_dim, self.clip_emb_dim, device=voxel_ridge.device)
            lb_loss_list = [lb_loss]
            backbone_out_0, clip_out_0 = torch.stack(backbone_out_0, 0), torch.stack(clip_out_0, 0)
            backbone_out_total[0] = backbone_out_0.sum(0)
            clip_out_total[0] = clip_out_0.sum(0)

            for ei, (router, experts, b_proj, c_proj) in enumerate(zip(self.router_list, self.expert_list, self.b_proj_list, self.c_proj_list)):
                router_out = [router[i](exp_out[i], training) for i in range(len(exp_out))]
                selected_tokens, weights, topk_ind, _ = zip(*router_out)
                selected_tokens, weights = torch.cat(selected_tokens, 0), torch.cat(weights, 0)
                exp_out = [expert(selected_tokens[i] * weights[i].unsqueeze(1)) for i, expert in enumerate(experts)]
                backbone_out = [b_proj[i](exp_out[i].flatten(1)).reshape(-1, self.clip_seq_dim, self.clip_emb_dim) for i in range(len(exp_out))]
                clip_out = [c_proj[i](backbone_out[i]) for i in range(len(exp_out))]

                if capture_routing:
                    routing_info[f'layer_{ei+1}'] = {
                        'topk_indices': torch.cat([t.detach().cpu() for t in topk_ind], 0),
                        'weights': weights,
                    }
                
                if self.routing:
                    backbone_out = self.back_router[ei](torch.stack(backbone_out, 0))
                    clip_out = self.clip_router[ei](torch.stack(clip_out, 0))
                else:
                    backbone_out = torch.stack(backbone_out, 0).sum(0)
                    clip_out = torch.stack(clip_out, 0).sum(0)
                
                backbone_out_total[ei + 1], clip_out_total[ei + 1] = backbone_out, clip_out
                
            lb_loss_list = torch.stack(lb_loss_list).mean()

            if capture_routing:
                return backbone_out_total.mean(0), clip_out_total.mean(0), None, lb_loss_list, routing_info
            return backbone_out_total.mean(0), clip_out_total.mean(0), None, lb_loss_list
        else:
            if self.routing:
                backbone_out_total, clip_out_total = [torch.stack(backbone_out_0, 0).sum(0, keepdim=True)], [torch.stack(clip_out_0, 0).sum(0, keepdim=True)]
            else:
                backbone_out_total, clip_out_total = [torch.stack(backbone_out_0, 0)], [torch.stack(clip_out_0, 0)]
                # backbone_out_total, clip_out_total = [torch.stack(backbone_out_0, 0).sum(0, keepdim=True)], [torch.stack(clip_out_0, 0).sum(0, keepdim=True)]

            for ei, (router, experts, b_proj, c_proj) in enumerate(zip(self.router_list, self.expert_list, self.b_proj_list, self.c_proj_list)):
                router_out = [router[i](exp_out[i], training) for i in range(len(exp_out))]
                selected_tokens, weights, _, _ = zip(*router_out)
                selected_tokens, weights = torch.cat(selected_tokens, 0), torch.cat(weights, 0)
                exp_out = [expert(selected_tokens[i] * weights[i].unsqueeze(1)) for i, expert in enumerate(experts)]
                backbone_out = [b_proj[i](exp_out[i].flatten(1)).reshape(-1, self.clip_seq_dim, self.clip_emb_dim) for i in range(len(exp_out))]
                clip_out = [c_proj[i](backbone_out[i]) for i in range(len(exp_out))]

                if self.routing:
                    backbone_out = self.back_router[ei](torch.stack(backbone_out, 0)).unsqueeze(0)
                    clip_out = self.clip_router[ei](torch.stack(clip_out, 0)).unsqueeze(0)
                else:
                    backbone_out = torch.stack(backbone_out, 0)
                    clip_out = torch.stack(clip_out, 0)

                backbone_out_total.append(backbone_out)
                clip_out_total.append(clip_out)
            
            clip_out_total = [x.sum(0) for x in clip_out_total]
            clip_out_total = torch.stack(clip_out_total, 0).mean(0)
            return backbone_out_total, clip_out_total, None, None