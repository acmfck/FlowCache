# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from torch.backends.cuda import sdp_kernel
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention

from .attention import flash_attention

from .compression.compress_kv import R1KV
import time

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")

DISABLE_COMPILE = False  # get os env

__all__ = ["WanModel"]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs, group_idx):
    n, c = x.size(2), x.size(3) // 2
    bs = x.size(0)

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    f, h, w = grid_sizes.tolist()
    seq_len = f * h * w

    # precompute multipliers
    start_f = group_idx * f
    end_f = start_f + f

    x = torch.view_as_complex(x.to(torch.float32).reshape(bs, seq_len, n, -1, 2))
    freqs_i = torch.cat(
        [
            freqs[0][start_f:end_f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    # apply rotary embedding
    x = torch.view_as_real(x * freqs_i).flatten(3)

    return x


@torch.compile(dynamic=True, disable=DISABLE_COMPILE)
def fast_rms_norm(x, weight, eps):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x = x.type_as(x) * weight
    return x


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return fast_rms_norm(x, self.weight, self.eps)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, layer_id=0, num_layers=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self._flag_ar_attention = False

        self.layer_id = layer_id
        self.num_layers = num_layers

        self.register_buffer('kv_cache', None)  # [B, L, nH, d]
        self.register_buffer('k_cache_even', None)
        self.register_buffer('v_cache_even',  None)
        self.register_buffer('k_cache_odd', None)
        self.register_buffer('v_cache_odd',  None)
        self.register_buffer('k_cache', None)
        self.register_buffer('v_cache', None)

    def set_ar_attention(self):
        self._flag_ar_attention = True

    def _alloc_kv(self, total_tokens, batch_size, device, dtype):
        return torch.zeros(
            batch_size,
            total_tokens,
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_and_return_kv(self, q, k, v, cond_flag, group_idx, group_size, grid_hw, num_groups, batch_size,
                              update_mask_per_group_list=None, kv_cluster=None, use_kvrange: bool = False, use_compress: bool = False):
        total_tokens   = num_groups *group_size* grid_hw         
        token_per_grp  = group_size * grid_hw 
        start = group_idx * token_per_grp
        end   = start + k.size(1)

        buf_k = self.k_cache_even if cond_flag else self.k_cache_odd
        buf_v = self.v_cache_even if cond_flag else self.v_cache_odd

        if buf_k is None and buf_v is None:
            buf_k = self._alloc_kv(total_tokens, batch_size, k.device, k.dtype)
            buf_v = self._alloc_kv(total_tokens, batch_size, v.device, v.dtype)
            
        buf_k[:,start:end] = k.detach()
        buf_v[:,start:end] = v.detach()

        if cond_flag: 
            self.k_cache_even = buf_k
            self.v_cache_even = buf_v
        else:
            self.k_cache_odd  = buf_k
            self.v_cache_odd  = buf_v

        if not use_kvrange and not use_compress:
            k_full = buf_k[:, :end]
            v_full = buf_v[:, :end]
            return k_full, v_full

        if use_compress:
            clean_idx_all = kv_cluster.clean_chunk_idx_even if cond_flag else kv_cluster.clean_chunk_idx_odd
            budget_block = getattr(kv_cluster, 'budget_block', 0) or 0

            if update_mask_per_group_list is None:
                update_mask_per_group_list = [False] * num_groups
            active_indices = [idx for idx in range(group_idx + 1) if update_mask_per_group_list[idx]]

            if len(clean_idx_all) <= budget_block or budget_block <= 0:
                parts_k = []
                parts_v = []

                if clean_idx_all:
                    for idx in sorted(clean_idx_all):
                        s_c = idx * token_per_grp
                        e_c = s_c + token_per_grp
                        parts_k.append(buf_k[:, s_c:e_c])
                        parts_v.append(buf_v[:, s_c:e_c])

                active_indices = [idx for idx in active_indices if idx not in clean_idx_all]
                
                for idx in active_indices:
                    s_a = idx * token_per_grp
                    e_a = s_a + token_per_grp
                    parts_k.append(buf_k[:, s_a:e_a])
                    parts_v.append(buf_v[:, s_a:e_a])

                if len(parts_k) == 0:
                    parts_k.append(buf_k[:, start:end])
                    parts_v.append(buf_v[:, start:end])

                k_full = torch.cat(parts_k, dim=1)
                v_full = torch.cat(parts_v, dim=1)
                return k_full, v_full
            else:
                clean_k_parts = []
                clean_v_parts = []
                for idx in sorted(clean_idx_all):
                    s_c = idx * token_per_grp
                    e_c = s_c + token_per_grp
                    clean_k_parts.append(buf_k[:, s_c:e_c])
                    clean_v_parts.append(buf_v[:, s_c:e_c])

                if len(clean_k_parts) == 0:
                    k_full = buf_k[:, :end]
                    v_full = buf_v[:, :end]
                    return k_full, v_full

                clean_k_cat = torch.cat(clean_k_parts, dim=1)  # [B, clean_tokens, nH, d]
                clean_v_cat = torch.cat(clean_v_parts, dim=1)

                clean_tokens = clean_k_cat.size(1)

                key_states = clean_k_cat[0]
                value_states = clean_v_cat[0]
                query_states = q[0]  # [token_per_grp, nH, d]


                key_comp, val_comp, _ = kv_cluster.update_kv_token(
                    key_states=key_states,
                    query_states=query_states,
                    value_states=value_states,
                    clean_chunk_tokens=clean_tokens,
                )

                keep_idx = sorted(clean_idx_all)[-budget_block:]
                for i, idx in enumerate(keep_idx):
                    s = idx * token_per_grp
                    e = s + token_per_grp
                    s_comp = i * token_per_grp
                    e_comp = s_comp + token_per_grp
                    buf_k[0, s:e] = key_comp[s_comp:e_comp]
                    buf_v[0, s:e] = val_comp[s_comp:e_comp]

                if self.layer_id == self.num_layers - 1:
                    if cond_flag:
                        kv_cluster.clean_chunk_idx_even = keep_idx
                    else:
                        kv_cluster.clean_chunk_idx_odd = keep_idx

                parts_k = []
                parts_v = []
                for idx in keep_idx:
                    s = idx * token_per_grp
                    e = s + token_per_grp
                    parts_k.append(buf_k[:, s:e])
                    parts_v.append(buf_v[:, s:e])
                
                active_indices = [idx for idx in active_indices if idx not in clean_idx_all]

                for idx in active_indices:
                    s_a = idx * token_per_grp
                    e_a = s_a + token_per_grp
                    parts_k.append(buf_k[:, s_a:e_a])
                    parts_v.append(buf_v[:, s_a:e_a])

                k_full = torch.cat(parts_k, dim=1)
                v_full = torch.cat(parts_v, dim=1)
                return k_full, v_full
            
        if not use_compress and use_kvrange:
            parts_k = []
            parts_v = []

            if kv_cluster is not None:
                clean_idx_all = kv_cluster.clean_chunk_idx_even if cond_flag else kv_cluster.clean_chunk_idx_odd
                kvrange = getattr(kv_cluster, 'kvrange', 0)
                if clean_idx_all:
                    clean_sorted = sorted(clean_idx_all)
                    select_clean = clean_sorted[-kvrange:] if kvrange > 0 else []
                    for idx in select_clean:
                        s_c = idx * token_per_grp
                        e_c = s_c + token_per_grp
                        parts_k.append(buf_k[:, s_c:e_c])
                        parts_v.append(buf_v[:, s_c:e_c])

            if update_mask_per_group_list is None:
                update_mask_per_group_list = [False] * num_groups
            active_indices = [idx for idx in range(group_idx + 1) if update_mask_per_group_list[idx]]
            active_indices = [idx for idx in active_indices if idx not in clean_idx_all]
            
            for idx in active_indices:
                s_a = idx * token_per_grp
                e_a = s_a + token_per_grp
                parts_k.append(buf_k[:, s_a:e_a])
                parts_v.append(buf_v[:, s_a:e_a])

            if len(parts_k) == 0:
                parts_k.append(buf_k[:, start:end])
                parts_v.append(buf_v[:, start:end])

            k_full = torch.cat(parts_k, dim=1)
            v_full = torch.cat(parts_v, dim=1)
            return k_full, v_full
    
    def forward(self, x, grid_sizes, freqs, block_mask, group_idx, cond_flag, num_groups,
                update_mask_per_group_list=None, kv_cluster=None, use_kvrange: bool = False, use_compress: bool = False):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        x = x.to(self.q.weight.dtype)
        q, k, v = qkv_fn(x)

        if not self._flag_ar_attention:
            q = rope_apply(q, grid_sizes, freqs, group_idx)
            k = rope_apply(k, grid_sizes, freqs, group_idx)
            #------------
            group_size = grid_sizes[0]
            grid_hw    = grid_sizes[1] * grid_sizes[2]
            k_full, v_full = self._update_and_return_kv(
                q, k, v, cond_flag, group_idx, group_size, grid_hw, num_groups, batch_size=b,
                update_mask_per_group_list=update_mask_per_group_list,
                kv_cluster=kv_cluster,
                use_kvrange=use_kvrange,
                use_compress=use_compress,
            )
            #------------
            x = flash_attention(q=q, k=k_full, v=v_full, window_size=self.window_size)
        else:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

            with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                x = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=block_mask
                    )
                    .transpose(1, 2)
                    .contiguous()
                )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = flash_attention(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img)
        # compute attention
        x = flash_attention(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


def mul_add(x, y, z):
    return x.float() + y.float() * z.float()


def mul_add_add(x, y, z):
    return x.float() * (1 + y) + z


mul_add_compile = torch.compile(mul_add, dynamic=True, disable=DISABLE_COMPILE)
mul_add_add_compile = torch.compile(mul_add_add, dynamic=True, disable=DISABLE_COMPILE)


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        layer_id=0,
        num_layers=0,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        
        self.layer_id = layer_id
        self.num_layers = num_layers

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, layer_id, num_layers) 
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps) 
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)


    def set_ar_attention(self):
        self.self_attn.set_ar_attention()

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs,
        context,
        block_mask,
        group_idx, 
        cond_flag,
        num_groups,
        update_mask_per_group_list=None,
        kv_cluster=None,
        use_kvrange: bool = False,
        use_compress: bool = False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """

        if e.dim() == 3:
            modulation = self.modulation  # 1, 6, dim
            with amp.autocast("cuda", dtype=torch.float32):
                e = (modulation + e).chunk(6, dim=1)
        elif e.dim() == 4:
            modulation = self.modulation.unsqueeze(2)  # 1, 6, 1, dim
            with amp.autocast("cuda", dtype=torch.float32):
                e = (modulation + e).chunk(6, dim=1)
            e = [ei.squeeze(1) for ei in e]

        # self-attention
        out = mul_add_add_compile(self.norm1(x), e[1], e[0])
        y = self.self_attn(
            out, grid_sizes, freqs, block_mask, group_idx, cond_flag, num_groups,
            update_mask_per_group_list=update_mask_per_group_list,
            kv_cluster=kv_cluster,
            use_kvrange=use_kvrange,
            use_compress=use_compress,
        )

        with amp.autocast("cuda", dtype=torch.float32):
            x = mul_add_compile(x, y, e[2])

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, e):
            dtype = context.dtype
            x = x + self.cross_attn(self.norm3(x.to(dtype)), context)
            y = self.ffn(mul_add_add_compile(self.norm2(x), e[4], e[3]).to(dtype))
            with amp.autocast("cuda", dtype=torch.float32):
                x = mul_add_compile(x, y, e[5])
            return x

        x = cross_attn_ffn(x, context, e)
        return x.to(torch.bfloat16)


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        with amp.autocast("cuda", dtype=torch.float32):
            if e.dim() == 2:
                modulation = self.modulation  # 1, 2, dim
                e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)

            elif e.dim() == 3:
                modulation = self.modulation.unsqueeze(2)  # 1, 2, seq, dim
                e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
                e = [ei.squeeze(1) for ei in e]
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        inject_sample_info=False,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = 1
        self.flag_causal_attention = False
        self.block_mask = None
        self.enable_teacache = False
        

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        if inject_sample_info:
            self.fps_embedding = nn.Embedding(2, dim)
            self.fps_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, layer_id=i, num_layers=num_layers)
                for i in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        self.gradient_checkpointing = False

        self.cpu_offloading = False

        self.inject_sample_info = inject_sample_info
        # initialize weights
        self.init_weights()

        self.group_size = 5 
        self.num_groups = 5 
        self.overlap = False 
        self.overlap_frames = 0 
        self.latent_width = 0 
        self.latent_height = 0 
        self.cnt_even = None 
        self.cnt_odd = None 
        self.cnt = 0 
        self.inference_steps = 0
        self.kv_cluster = R1KV()
        self.use_kvrange = False
        self.use_compress = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def zero_init_i2v_cross_attn(self):
        print("zero init i2v cross attn")
        for i in range(self.num_layers):
            self.blocks[i].cross_attn.v_img.weight.data.zero_()
            self.blocks[i].cross_attn.v_img.bias.data.zero_()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21, frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        return block_mask
                
    def initialize_asynchronous_teacache(self, enable_teacache=True, num_steps=25, teacache_thresh=0.15, use_ret_steps=False, ckpt_dir='', inference_steps=0):
        self.enable_teacache = enable_teacache
        self.inference_steps = inference_steps
        print('using asynchronous teacache')
        self.cnt = 0
        self.num_steps = num_steps
        self.teacache_thresh = teacache_thresh
        self.use_ref_steps = use_ret_steps
        if use_ret_steps:
            if '1.3B' in ckpt_dir:
                self.coefficients = [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
            if '14B' in ckpt_dir:
                self.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
            self.ret_steps = 5
            self.cutoff_steps = inference_steps - 1
        else:
            if '1.3B' in ckpt_dir:
                self.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
            if '14B' in ckpt_dir:
                self.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
            self.ret_steps = 1
            self.cutoff_steps = inference_steps - 1

    def clear_teacache(self):
        for i in range(self.num_layers):
            self.blocks[i].self_attn.kv_cache = None
            self.blocks[i].self_attn.k_cache_even = None
            self.blocks[i].self_attn.v_cache_even = None
            self.blocks[i].self_attn.k_cache_odd  = None
            self.blocks[i].self_attn.v_cache_odd  = None


    def forward(self, x, t, context, update_mask_i ,clip_fea=None, y=None, fps=None):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        #-----------------
        group_size      = self.group_size   
        num_groups      = self.num_groups 
        overlap         = self.overlap           
        overlap_frames  = self.overlap_frames
        update_mask_per_group = update_mask_i.view(num_groups, group_size).any(dim=1)
        update_mask_per_group_list = [False]*num_groups 
        for indx in range(num_groups):
            if update_mask_per_group[indx]==True:
                update_mask_per_group_list[indx] = True
        should_forward_groupe = [False]*num_groups 
        for indx in range(num_groups-1, -1, -1):
            if update_mask_per_group_list[indx]==True:
                last_true = indx
                break
        for j in range(last_true+1):
            should_forward_groupe[j] = True

        #------------------------------------------------
        for g in range(num_groups):
            if should_forward_groupe[g]:
                cnt_vec = self.cnt_even if (self.cnt % 2 == 0) else self.cnt_odd
                if cnt_vec[g] >= self.inference_steps: 
                    should_forward_groupe[g] = False
        if self.overlap:
            if self.cnt <= 1:
                should_forward_groupe[0] = True
            else:
                should_forward_groupe[0] = False

        if self.overlap and self.cnt==1: 
            self.kv_cluster.clean_chunk_idx_even.append(0)
        if self.overlap and self.cnt==2:
            self.kv_cluster.clean_chunk_idx_odd.append(0)
        if y is not None: 
            x = torch.cat([x, y], dim=1)

        # embeddings
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)

        #-----------------
        self.latent_width = grid_sizes[2]
        self.latent_height = grid_sizes[1]
        token_per_frame = self.latent_width * self.latent_height
        token_per_group = group_size * token_per_frame
        #-----------------

        x = x.flatten(2).transpose(1, 2)

        if self.flag_causal_attention:
            frame_num = grid_sizes[0]
            height = grid_sizes[1]
            width = grid_sizes[2]
            block_num = frame_num // self.num_frame_per_block
            range_tensor = torch.arange(block_num).view(-1, 1)
            range_tensor = range_tensor.repeat(1, self.num_frame_per_block).flatten()
            casual_mask = range_tensor.unsqueeze(0) <= range_tensor.unsqueeze(1)  # f, f
            casual_mask = casual_mask.view(frame_num, 1, 1, frame_num, 1, 1).to(x.device)
            casual_mask = casual_mask.repeat(1, height, width, 1, height, width)
            casual_mask = casual_mask.reshape(frame_num * height * width, frame_num * height * width)
            self.block_mask = casual_mask.unsqueeze(0).unsqueeze(0)

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            if t.dim() == 2:
                b, f = t.shape
                _flag_df = True
            else:
                _flag_df = False

            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(self.patch_embedding.weight.dtype) 
            )  # b, dim

            e0 = self.time_projection(e).unflatten(1, (6, self.dim))  

            if self.inject_sample_info:
                fps = torch.tensor(fps, dtype=torch.long, device=device)

                fps_emb = self.fps_embedding(fps).float()
                if _flag_df:
                    e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim)).repeat(t.shape[1], 1, 1)
                else:
                    e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim))

            if _flag_df:
                e = e.view(b, f, 1, 1, self.dim)
                e0 = e0.view(b, f, 1, 1, 6, self.dim)
                e = e.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1).flatten(1, 3)
                e0 = e0.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1, 1).flatten(1, 3)
                e0 = e0.transpose(1, 2).contiguous()

            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context = self.text_embedding(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        x_chunks = torch.chunk(x,  num_groups, dim=1)   
        e0_chunks = torch.chunk(e0, num_groups, dim=2)  

        cond_flag = (self.cnt % 2 == 0)

        out_chunks = [torch.zeros_like(x_g) for x_g in x_chunks]
        
        for g, (x_g, e0_g) in enumerate(zip(x_chunks, e0_chunks)):
            if should_forward_groupe[g]==True: 
                grid_sizes[0] =  group_size
                kwargs = dict(
                    e=e0_g,
                    grid_sizes=grid_sizes,
                    freqs=self.freqs,
                    context=context,
                    block_mask=self.block_mask,
                    group_idx=g,
                    cond_flag=cond_flag,
                    num_groups=num_groups,
                    update_mask_per_group_list=update_mask_per_group_list,
                    kv_cluster=self.kv_cluster,
                    use_kvrange=self.use_kvrange,
                    use_compress=self.use_compress,
                )

                modulated_inp = e0_g     
                cnt_vec   = self.cnt_even if cond_flag else self.cnt_odd
                step_cnt  = cnt_vec[g]
                if cond_flag:
                    acc   = getattr(self, 'accumulated_rel_l1_distance_even', {})
                    prev  = getattr(self, 'previous_e0_even', {})
                    res   = getattr(self, 'previous_residual_even', {})
                else:
                    acc   = getattr(self, 'accumulated_rel_l1_distance_odd', {})
                    prev  = getattr(self, 'previous_e0_odd', {})
                    res   = getattr(self, 'previous_residual_odd', {})

                if self.enable_teacache and update_mask_per_group_list[g]==True: 
                    if step_cnt < self.ret_steps or step_cnt >= self.cutoff_steps: 
                        should_calc = True
                        acc[g] = 0.0
                    else:
                        prev_feat = prev[g]
                        rescale_func = np.poly1d(self.coefficients)
                        dist = rescale_func(((modulated_inp - prev_feat).abs().mean() / prev_feat.abs().mean()).cpu().item())
                        acc[g] = acc[g] + dist
                        should_calc = acc[g] >= self.teacache_thresh
                        if should_calc:
                            acc[g] = 0.0
                    prev[g] = modulated_inp.clone()
                    if cond_flag: 
                        self.accumulated_rel_l1_distance_even = acc
                        self.previous_e0_even = prev
                    else:
                        self.accumulated_rel_l1_distance_odd = acc
                        self.previous_e0_odd  = prev
                else:
                    should_calc = True
            
                if not should_calc:
                    if cond_flag:
                        self.skip_even[g].append(self.cnt//2+1)
                    else:
                        self.skip_odd[g].append((self.cnt+1)//2)
                    x_g = x_g + res[g]
                else:
                    ori_g = x_g.clone()
                    for block in self.blocks:
                        x_g = block(x_g,**kwargs)
                    if update_mask_per_group_list[g]==True:
                        res[g] = x_g - ori_g
                        if cond_flag:
                            self.previous_residual_even = res
                        else:
                            self.previous_residual_odd = res
                if update_mask_per_group_list[g]==True:
                    cnt_vec[g] = cnt_vec[g]+1

                    if cnt_vec[g] >= self.inference_steps: 
                        if cond_flag:
                            self.kv_cluster.clean_chunk_idx_even.append(g) 
                        else:
                            self.kv_cluster.clean_chunk_idx_odd.append(g)
                        
                    if cond_flag:
                        self.cnt_even = cnt_vec
                    else:
                        self.cnt_odd = cnt_vec

                out_chunks[g] = x_g
            else:
                continue

        self.cnt = self.cnt + 1
        

        x = torch.cat(out_chunks, dim=1)

        x = self.head(x, e)

        grid_sizes[2] = self.latent_width
        grid_sizes[1] = self.latent_height
        grid_sizes[0] = group_size * num_groups

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        return x.float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        bs = x.shape[0]
        x = x.view(bs, *grid_sizes, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])

        return x

    def set_ar_attention(self, causal_block_size):
        self.num_frame_per_block = causal_block_size
        self.flag_causal_attention = True
        for block in self.blocks:
            block.set_ar_attention()

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        if self.inject_sample_info:
            nn.init.normal_(self.fps_embedding.weight, std=0.02)

            for m in self.fps_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

            nn.init.zeros_(self.fps_projection[-1].weight)
            nn.init.zeros_(self.fps_projection[-1].bias)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
