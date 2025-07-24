import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import open_clip
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial
import kornia

from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.model import CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype, resize_text_pos_embed, set_model_preprocess_cfg
from open_clip.coca_model import CoCa
from open_clip.loss import ClipLoss, DistillClipLoss, CoCaLoss, SigLipLoss
from open_clip.openai import load_openai_model
from open_clip.pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from open_clip.transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer, DEFAULT_CONTEXT_LENGTH
from open_clip.utils import to_2tuple
from open_clip.pos_embed import get_2d_sincos_pos_embed
from open_clip.timm_model import TimmModel
from open_clip.transformer import LayerNormFp32, text_global_pool, VisionTransformer
from open_clip.modified_resnet import ModifiedResNet

HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


################# copied from openclip factory.py#####################
def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
):
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config = _get_hf_config(model_id, cache_dir)
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg'])
        model_cfg = config['model_cfg']
        pretrained_hf = False  # override, no need to load original HF text weights
    else:
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        checkpoint_path = None
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = model_cfg or open_clip.factory.get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f'Loaded {model_name} model config.')
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
        if pretrained_image:
            if is_timm_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
        cast_dtype = get_cast_dtype(precision)
        is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
        if is_hf_model:
            # load pretrained weights for HF text model IFF no CLIP weights being loaded
            model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf and not pretrained
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

        model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)
        if custom_text:
            if "multimodal_cfg" in model_cfg:
                model = CoCa(**model_cfg, cast_dtype=cast_dtype)
            else:
                model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        if precision in ("fp16", "bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            # manual mixed precision that matches original OpenAI behaviour
            if is_timm_model:
                # FIXME this is a bit janky, create timm based model in low-precision and
                # then cast only LayerNormFp32 instances back to float32 so they don't break.
                # Why? The convert_weights_to_lp fn only works with native models.
                model.to(device=device, dtype=dtype)
                from open_clip.transformer import LayerNormFp32

                def _convert_ln(m):
                    if isinstance(m, LayerNormFp32):
                        m.weight.data = m.weight.data.to(torch.float32)
                        m.bias.data = m.bias.data.to(torch.float32)
                model.apply(_convert_ln)
            else:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
        elif precision in ("pure_fp16", "pure_bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            model.to(device=device, dtype=dtype)
        else:
            model.to(device=device)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = ''
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
                preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                import torch.distributed as dist
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                open_clip.load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    f' Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                logging.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True
        elif has_hf_hub_prefix:
            logging.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
            open_clip.load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg['size'] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model



def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode)

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, preprocess_train, preprocess_val


##################copied from openclip model.py#####################
class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: open_clip.CLIPVisionCfg,
            text_cfg: open_clip.CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = open_clip.model._build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: open_clip.CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = open_clip.CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = open_clip.model.QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else nn.LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformerInterm(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


##################copied from openclip transformer.py#####################
def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class VisionTransformerInterm(VisionTransformer):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)
        scale = kwargs['width'] ** -0.5
        if 'attention_pool' in kwargs and kwargs['attention_pool']:
            pool_dim = kwargs['output_dim']
        else:
            pool_dim = kwargs['width']
        # self.projs = nn.ParameterList([nn.Parameter(scale * torch.randn(pool_dim, self.output_dim)) for _ in range(4)])

    def postprocessing(self, x: torch.Tensor, idx):
        if self.attn_pool is not None:
            raise NotImplementedError('attn_pool not implemented for VisionTransformerInterm')
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        # if self.projs is not None:
        #     pooled = pooled @ self.projs[idx]
        return tokens, pooled

    def forward(self, x: torch.Tensor):
        intermediates = []
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        num_layers = self.transformer.layers
        # intermediate_indices = [
        #     max(0, min((i * num_layers) // 4 - 1, num_layers))
        #     for i in range(1, 4)
        # ]
        intermediate_indices = np.arange(num_layers)
        proj_idx = 0
        for layer_idx, layer in enumerate(self.transformer.resblocks):
            x = layer(x)
            if layer_idx in intermediate_indices:
                intermediates.append(self.postprocessing(x.permute(1, 0, 2), proj_idx)[0])  # LND -> NLD
                proj_idx += 1

        x = x.permute(1, 0, 2)  # LND -> NLD

        tokens, pooled = self.postprocessing(x, 3)

        return pooled, tokens, intermediates
    

from .modules import FrozenOpenCLIPImageEmbedder
from ...util import (append_dims, autocast, count_params, default,
                     disabled_train, expand_dims_like, instantiate_from_config)
from einops import rearrange, repeat

class FrozenOpenCLIPImageEmbedderInterm(FrozenOpenCLIPImageEmbedder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        model, _, _ = create_model_and_transforms(
            kwargs['arch'],
            device=torch.device('cpu'),
            pretrained=kwargs['version'],
            cache_dir=None if 'cache_dir' not in kwargs else kwargs['cache_dir'],
        )
        del model.transformer
        self.model = model
        self.model.visual.output_tokens = kwargs['output_tokens']

    @autocast
    def _process_token(self, tokens):
        return (
            expand_dims_like(
                torch.bernoulli(
                    (1.0 - self.ucg_rate)
                    * torch.ones(tokens.shape[0], device=tokens.device)
                ),
                tokens,
            )
            * tokens
        )

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens, tokens_interm = z[0], z[1], z[2]
        z = z.to(image.dtype)
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device)
                )[:, None]
                * z
            )
            if tokens is not None:
                tokens = self._process_token(tokens)
                tokens_interm = [self._process_token(t) for t in tokens_interm]
        if self.unsqueeze_dim:
            z = z[:, None, :]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            if self.only_tokens:
                return tokens, tokens_interm
            return tokens, z
        
        raise NotImplementedError
        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(
                        z.shape[0],
                        self.max_length - z.shape[1],
                        z.shape[2],
                        device=z.device,
                    ),
                ),
                1,
            )
            return z_pad, z_pad[:, 0, ...]
        return z

    def _norm_token(self, tokens):
        token_shape = tokens.shape
        tokens = tokens.flatten(1)
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        tokens = (tokens - .0002) / .0015
        tokens = tokens.view(token_shape)
        tokens = (tokens * 1.0957) + .1598

    def encode_with_vision_transformer(self, img):
        # if self.max_crops > 0:
        #    img = self.preprocess_by_cropping(img)
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")
        img = self.preprocess(img)
        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens, intermediates = self.model.visual(img)
            
            if self.l2_norm_tokens:
                tokens = self._norm_token(tokens)
                for t in intermediates:
                    self._norm_token(t)
                
        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # drop out between 0 and all along the sequence axis
            x = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate)
                    * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                )
                * x
            )
            if tokens is not None:
                raise NotImplementedError
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)
                logpy.warning(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message."
                )
        if self.output_tokens:
            return x, tokens, intermediates
        return x



from .modules import FrozenOpenCLIPImageEmbedder
class FrozenOpenCLIPImageEmbedderProjTokens(FrozenOpenCLIPImageEmbedder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        init_device="cpu",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        l2_norm_tokens=False,
        only_tokens=False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device(init_device),
            pretrained=version,
            cache_dir=cache_dir,
        )
        del model.transformer
        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens
        if only_tokens:
            assert output_tokens
        self.only_tokens = only_tokens
        self.l2_norm_tokens = l2_norm_tokens
        if l2_norm_tokens:
            assert output_tokens
            
    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z.to(image.dtype)
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device)
                )[:, None]
                * z
            )
            if tokens is not None:
                tokens = (
                    expand_dims_like(
                        torch.bernoulli(
                            (1.0 - self.ucg_rate)
                            * torch.ones(tokens.shape[0], device=tokens.device)
                        ),
                        tokens,
                    )
                    * tokens
                )
        if self.unsqueeze_dim:
            z = z[:, None, :]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            if self.only_tokens:
                return tokens
            return tokens, z
        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(
                        z.shape[0],
                        self.max_length - z.shape[1],
                        z.shape[2],
                        device=z.device,
                    ),
                ),
                1,
            )
            return z_pad, z_pad[:, 0, ...]
        return z

    def encode_with_vision_transformer(self, img):
        # if self.max_crops > 0:
        #    img = self.preprocess_by_cropping(img)
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")
        img = self.preprocess(img)
        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)
            tokens = tokens @ self.model.visual.proj
            if self.l2_norm_tokens:
                token_shape = tokens.shape
                tokens = tokens.flatten(1)
                tokens = torch.nn.functional.normalize(tokens, dim=-1)
                tokens = (tokens - .0002) / .0015
                tokens = tokens.view(token_shape)
                tokens = (tokens * 1.0957) + .1598
        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # drop out between 0 and all along the sequence axis
            x = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate)
                    * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                )
                * x
            )
            if tokens is not None:
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)
                logpy.warning(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message."
                )
        if self.output_tokens:
            return x, tokens
        return x

    def encode(self, text):
        return self(text)
  