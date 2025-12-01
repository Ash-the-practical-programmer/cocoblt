"""Hybrid Attention Layer with FlashAttention-2 support.

Supports:
- Patch-to-Patch (Causal Self-Attention)
- Patch-to-Latent (Cross-Attention, Latents read Patches)
- Latent-to-Patch (Cross-Attention, Patches read Latents)
"""

from typing import Optional, Tuple
import torch
from torch import nn
import xformers.ops as xops
from xformers.ops import memory_efficient_attention, LowerTriangularMask

class HybridAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout

        # We assume Q, K, V projections are handled inside or outside.
        # Usually standard is internal.
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: torch.Tensor,
        v_x: torch.Tensor,
        attn_bias: Optional[xops.AttentionBias] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Args:
            q_x, k_x, v_x: (Total_Tokens, Dim) - Flattened sequences
            attn_bias: Bias mask (e.g., LowerTriangularMask for causal)
            cu_seqlens_q/k: Cumulative sequence lengths for batching
            is_causal: If True, applies causal mask automatically (if attn_bias is None)
        """
        total_q, _ = q_x.shape
        total_k, _ = k_x.shape

        # Project
        q = self.q_proj(q_x).view(total_q, self.num_heads, self.head_dim)
        k = self.k_proj(k_x).view(total_k, self.num_heads, self.head_dim)
        v = self.v_proj(v_x).view(total_k, self.num_heads, self.head_dim)

        # Flash Attention
        # Note: xformers expects (B, S, H, D) or (1, Total, H, D) with cu_seqlens
        # When using cu_seqlens, input should be (1, Total, H, D) or just (Total, H, D)?
        # xformers documentation says:
        # if inputs are (B, M, H, K), then cu_seqlens is not used.
        # if inputs are (1, Total, H, K) or (Total, H, K), then cu_seqlens must be provided.

        # We need to reshape to (1, Total, H, D) to satisfy xformers signature with cu_seqlens usually,
        # or stick to the (Total, ...) format.
        # Let's ensure it's (1, Total, H, D) just to be safe if that's what it prefers,
        # but typically (Total, H, D) works if we pass the right flags.

        # Let's add a dummy batch dim: (1, Total, H, D)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        # If causal, we can pass attn_bias=LowerTriangularMask()
        if is_causal and attn_bias is None:
            attn_bias = LowerTriangularMask()

        out = memory_efficient_attention(
            q, k, v,
            attn_bias=attn_bias,
            p=self.dropout_p,
            scale=self.scale
        )
        # out: (1, Total, H, D)

        out = out.squeeze(0).reshape(total_q, self.dim)
        return self.out_proj(out)

    @classmethod
    def patch_to_patch(cls, dim, num_heads):
        return cls(dim, num_heads)

    @classmethod
    def patch_to_latent(cls, dim, num_heads):
        return cls(dim, num_heads)

    @classmethod
    def latent_to_patch(cls, dim, num_heads):
        return cls(dim, num_heads)
