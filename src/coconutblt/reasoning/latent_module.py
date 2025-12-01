"""Continuous Latent Reasoning Module.

This module maintains a set of learnable latent vectors ("thoughts") that interact with the token stream.
"""

from typing import Optional, Tuple
import torch
from torch import nn
import xformers.ops as xops
from xformers.ops.fmha import BlockDiagonalMask

from src.coconutblt.attention.hybrid_attention import HybridAttention

class ContinuousLatentModule(nn.Module):
    def __init__(self,
                 n_latents: int = 6,
                 latent_dim: int = 2048,
                 n_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.n_latents = n_latents
        self.latent_dim = latent_dim

        # Learnable latent embeddings (the "thoughts")
        # Initialize small
        self.latents = nn.Parameter(torch.randn(n_latents, latent_dim) * 0.02)

        # Norms
        self.ln_latents_write = nn.LayerNorm(latent_dim)
        self.ln_tokens_write = nn.LayerNorm(latent_dim)

        self.ln_tokens_read = nn.LayerNorm(latent_dim)
        self.ln_latents_read = nn.LayerNorm(latent_dim)

        # Attentions
        # 1. Write: Latents (Q) read Tokens (K, V) -> Update Latents
        self.write_attn = HybridAttention(latent_dim, n_heads, dropout=dropout)

        # 2. Read: Tokens (Q) read Latents (K, V) -> Update Tokens
        self.read_attn = HybridAttention(latent_dim, n_heads, dropout=dropout)

        # Gating
        self.gate_write = nn.Linear(latent_dim, latent_dim) # For latents
        self.gate_read = nn.Linear(latent_dim, latent_dim) # For tokens

    def forward(self,
                tokens: torch.Tensor,
                current_latents: torch.Tensor,
                cu_seqlens: torch.Tensor,
                max_seqlen: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (Total_Tokens, D)
            current_latents: (B, n_latents, D) - Passed from previous layer or init
            cu_seqlens: (B+1,)
            max_seqlen: max length of patch sequence in batch
        Returns:
            updated_tokens: (Total_Tokens, D)
            updated_latents: (B, n_latents, D)
        """
        B = current_latents.shape[0]
        D = self.latent_dim
        n_L = self.n_latents

        # Flatten latents for xformers: (B*n_latents, D)
        latents_flat = current_latents.reshape(B * n_L, D)

        # Construct cu_seqlens for latents
        # simple: [0, n_L, 2*n_L, ...]
        cu_seqlens_latents = torch.arange(0, (B + 1) * n_L, n_L, device=current_latents.device, dtype=torch.int32)

        # --- 1. WRITE PHASE: Latents attend to Tokens ---
        # Q = Latents, K=Tokens, V=Tokens
        # This is cross-attention. We need to respect batch boundaries.
        # xformers memory_efficient_attention supports block diagonal masks implicitly via cu_seqlens

        l_norm = self.ln_latents_write(latents_flat)
        t_norm = self.ln_tokens_write(tokens)

        # We need an AttentionBias that allows:
        # q (latents of batch b) to attend to k (tokens of batch b)
        # BlockDiagonalMask does exactly this given the cu_seqlens for q and k.
        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=torch.full((B,), n_L, dtype=torch.int32, device=tokens.device).tolist(),
            kv_seqlen=[(cu_seqlens[i+1] - cu_seqlens[i]).item() for i in range(B)]
        )

        # We need to reshape inputs to (1, Total, H, D) inside HybridAttention, but here we pass flattened.
        # HybridAttention handles the unsqueeze.

        # Note: HybridAttention signature: q_x, k_x, v_x
        # But for xformers with cu_seqlens, we need to pass cu_seqlens.
        # The HybridAttention wrapper needs to pass those through.

        delta_latents = self.write_attn(
            q_x=l_norm,
            k_x=t_norm,
            v_x=t_norm,
            attn_bias=attn_bias
        )

        # Gated update
        g_l = torch.sigmoid(self.gate_write(delta_latents))
        new_latents = latents_flat + g_l * delta_latents

        # --- 2. READ PHASE: Tokens attend to Latents ---
        # Q = Tokens, K=Latents, V=Latents

        t_norm_2 = self.ln_tokens_read(tokens) # Use original tokens (skip connection usually around whole block?)
        # Wait, usually we daisy chain: tokens -> update -> next layer.
        # But here tokens and latents are updated in parallel or sequential?
        # "Cross-attention layers (patches bidirectional with latents)"
        # Coconut usually does: Latents update, then Tokens update using OLD or NEW latents?
        # Let's use NEW latents (sequential).

        l_norm_2 = self.ln_latents_read(new_latents)

        attn_bias_read = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=[(cu_seqlens[i+1] - cu_seqlens[i]).item() for i in range(B)],
            kv_seqlen=torch.full((B,), n_L, dtype=torch.int32, device=tokens.device).tolist()
        )

        delta_tokens = self.read_attn(
            q_x=t_norm_2,
            k_x=l_norm_2,
            v_x=l_norm_2,
            attn_bias=attn_bias_read
        )

        g_t = torch.sigmoid(self.gate_read(delta_tokens))
        new_tokens = tokens + g_t * delta_tokens

        return new_tokens, new_latents.reshape(B, n_L, D)
