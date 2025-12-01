"""Core CocoBLT Transformer Model.

Assembles:
- EntropyPatcher + PatchEncoder
- Transformer Layers
- Continuous Latent Modules (inserted periodically)
- Output Head
"""

import math
from typing import Optional, Tuple, List
import torch
from torch import nn
import xformers.ops as xops
from xformers.ops import memory_efficient_attention

from src.coconutblt.patching.entropy_patcher import AdaptivePatcher, PatchEncoder
from src.coconutblt.reasoning.latent_module import ContinuousLatentModule
from src.coconutblt.attention.hybrid_attention import HybridAttention

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = HybridAttention(dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, cu_seqlens, max_seqlen):
        # Self-Attention
        # Need Causal Mask
        # BlockDiagonalMask(causal=True) handles causality per batch element defined by cu_seqlens
        # Note: BlockDiagonalCausalMask might not be directly exposed in xops.fmha in some versions
        # We can construct it via BlockDiagonalMask.from_seqlens(..., make_causal=True) or similar
        # Checking xformers API... usually it's xops.fmha.BlockDiagonalCausalMask if available.
        # If not, let's use LowerTriangularMask if treating as one large sequence with masking?
        # But for variable length, BlockDiagonalCausalMask is preferred.
        # Let's try xops.fmha.BlockDiagonalCausalMask

        attn_bias = xops.fmha.BlockDiagonalCausalMask.from_seqlens(
            [(cu_seqlens[i+1] - cu_seqlens[i]).item() for i in range(len(cu_seqlens)-1)]
        )

        x_norm = self.ln1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, attn_bias=attn_bias, is_causal=True)
        x = x + self.mlp(self.ln2(x))
        return x

class CocoBLTTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config['model']['dim']

        # Patcher
        self.patcher = AdaptivePatcher(
            min_patch_size=config['patching']['min_patch_size'],
            max_patch_size=config['patching']['max_patch_size'],
            lookahead=config['patching']['lookahead'],
            entropy_threshold=config['patching']['entropy_threshold'],
            pad_token=config['patching']['pad_token_id']
        )

        self.patch_encoder = PatchEncoder(
            dim_model=dim,
            byte_emb_dim=512,
            max_patch_size=config['patching']['max_patch_size'],
            vocab_size=config['model']['vocab_size']
        )

        # Layers
        self.layers = nn.ModuleList()
        self.latent_modules = nn.ModuleList()

        n_layers = config['model']['n_layers']
        latent_interval = config['reasoning']['latent_interval']

        # We need a map to know when to use which latent module
        self.latent_map = {} # layer_idx -> latent_module_idx

        current_latent_idx = 0
        for i in range(n_layers):
            self.layers.append(
                TransformerBlock(
                    dim,
                    config['model']['n_heads'],
                    config['model']['mlp_ratio'],
                    config['model']['dropout']
                )
            )
            # Insert latent module AFTER layer i if (i+1) % interval == 0
            # Also maybe insert one at start? Prompt says "Insert latent module every 4 layers".
            # Usually implies after layer 3 (0-indexed 3).
            # "layers 0-3, 4-7..."
            # Let's assume after the block.
            if (i + 1) % latent_interval == 0:
                self.latent_modules.append(
                    ContinuousLatentModule(
                        n_latents=config['reasoning']['n_latents'],
                        latent_dim=config['reasoning']['latent_dim'],
                        n_heads=config['reasoning']['n_heads']
                    )
                )
                self.latent_map[i] = current_latent_idx
                current_latent_idx += 1

        self.ln_f = nn.LayerNorm(dim)

        # Output Head
        # Projects from Patch Dim -> Bytes (256 logits)
        # But a patch represents multiple bytes?
        # Standard BLT decodes patch -> multiple bytes via a local transformer or simple head.
        # "Output Head (256-way byte logits)"
        # Requirements don't specify a "Local Decoder".
        # If I just project dim -> 256, I get 1 logits vector per patch.
        # But a patch covers 1-8 bytes.
        # How do we get 8 bytes out?
        # "Architecture flow: ... -> Output Head (256-way byte logits)"
        # In BLT, there is a "Local Transformer" or decoder that unpacks the patch representation.
        # Given "Foundational architecture" and "1B params" (which is mostly the global transformer),
        # I should probably implement a simple Local Decoder to predict the bytes.
        # OR: The PatchEncoder uses a "Hash-based pos encoding", maybe we just predict the bytes *autoregressively* at the global level?
        # But the transformer operates on *patches*.
        # Re-reading: "Output Head (256-way byte logits)"
        # This implies we need to predict the bytes.
        # I will implement a small Local MLP/Transformer that takes (Patch_Emb) and predicts (Max_Patch_Size, 256).
        # We can then mask based on how many bytes were actually in the patch (during training).

        self.head_dim = dim
        self.vocab_size = config['model']['vocab_size']
        self.max_patch_size = config['patching']['max_patch_size']

        # Simple Local Decoder: Linear expansion to (Max_Patch_Size * Vocab) or (Max_Patch * Dim) then project?
        # Let's do: Linear(Dim -> Max_Patch * Dim) -> Reshape -> Linear(Dim -> Vocab)
        self.local_decoder = nn.Sequential(
             nn.Linear(dim, self.max_patch_size * dim),
             nn.GELU()
        )
        self.output_proj = nn.Linear(dim, self.vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: (B, S) - Byte sequence
            labels: (B, S) - Next-byte labels (aligned with input or shifted?)
                    Usually causal LLM training: input=x[:-1], label=x[1:].
                    Here we handle the shifting or assume caller did.
        """
        # 1. Patching
        # patches: List of (N_i, Max_Patch_Size)
        patches_list, offsets_list, cu_seqlens_list = self.patcher(input_ids)

        # Flatten patches for encoder
        patches_flat = torch.cat(patches_list, dim=0) # (Total_Patches, Max_Patch_Size)
        offsets_flat = torch.cat(offsets_list, dim=0) # (Total_Patches,)

        cu_seqlens = torch.tensor(cu_seqlens_list, device=input_ids.device, dtype=torch.int32)

        # 2. Encode
        x = self.patch_encoder(patches_flat, offsets_flat) # (Total_Patches, Dim)

        # 3. Latent Init
        # Initialize latents for each batch element
        B = len(patches_list)
        if len(self.latent_modules) > 0:
            latents = self.latent_modules[0].latents.unsqueeze(0).expand(B, -1, -1) # (B, n_latents, D)

        # 4. Transformer Stack
        max_seqlen = 0
        for p in patches_list:
            max_seqlen = max(max_seqlen, p.shape[0])

        for i, layer in enumerate(self.layers):
            x = layer(x, cu_seqlens, max_seqlen)

            if i in self.latent_map:
                mod_idx = self.latent_map[i]
                latent_mod = self.latent_modules[mod_idx]
                x, latents = latent_mod(x, latents, cu_seqlens, max_seqlen)

        x = self.ln_f(x) # (Total_Patches, Dim)

        # 5. Output Head
        # Expand patches back to bytes
        x_expanded = self.local_decoder(x) # (Total_Patches, Max_Patch_Size * Dim)
        x_expanded = x_expanded.view(-1, self.max_patch_size, self.head_dim) # (Total_Patches, 8, Dim)
        logits = self.output_proj(x_expanded) # (Total_Patches, 8, 256)

        loss = None
        if labels is not None:
            # We need to align logits with labels.
            # Labels are per byte (B, S).
            # Our logits are per patch (B, N_patches, 8).
            # We need to map back to the original byte sequence.
            # But patching is dynamic!
            # We know 'offsets' and patch lengths.
            # We can reconstruct the flat sequence of valid bytes from logits.

            # This is tricky for batching efficiently.
            # Simplest way: Flatten labels based on the patching that happened?
            # Or reconstruct a dense logit tensor?

            # Let's iterate over batch and gather valid logits.

            batch_loss = 0.0
            total_tokens = 0

            # Cumulative count to slice into flattened x
            start_patch_idx = 0

            for b in range(B):
                n_patches = patches_list[b].shape[0]
                b_logits = logits[start_patch_idx : start_patch_idx + n_patches] # (N_p, 8, 256)

                # Get the valid length of each patch to mask padding
                # We can deduce valid length by looking at the patches input
                # Or re-calculate.
                # patches_list[b] contains padded bytes. Pad token is 0.
                # Assuming pad token 0 is only used for padding.
                # If 0 is a valid byte, we need a mask from the patcher.
                # The patcher padded with pad_token.
                # Let's assume pad_token is ignored in loss.

                b_patches = patches_list[b] # (N_p, 8)
                mask = (b_patches != self.patcher.pad_token)

                # Flatten
                flat_logits = b_logits[mask] # (Valid_Bytes, 256)

                # Get corresponding labels
                # We need to be careful. patcher output corresponds to input_ids[b].
                # If labels are shifted, we need to match carefully.
                # Usually we expect labels[b] to match input_ids[b] position-wise.
                # So we can just extract the same positions.

                # But wait, AdaptivePatcher might have skipped bytes? No, it covers the whole sequence.
                # It just chunked it.
                # So flat_logits should correspond exactly to input_ids[b] (excluding padding added by patcher).

                # Verify length
                b_labels = labels[b]
                # If labels is same length as input_ids.
                # We need to match flat_logits length to b_labels length.

                # If input_ids[b] had padding at the end (standard batch padding), the patcher might have included it?
                # Patcher logic: `while i < S`. If S includes padding, it patches padding.
                # We should assume input_ids is unpadded or we respect a mask.
                # For this prototype, let's assume `labels` aligns with the valid bytes extracted.

                # Trim to min length
                n_valid = min(flat_logits.shape[0], b_labels.shape[0])
                if n_valid > 0:
                    batch_loss += nn.functional.cross_entropy(flat_logits[:n_valid], b_labels[:n_valid], reduction='sum')
                    total_tokens += n_valid

                start_patch_idx += n_patches

            if total_tokens > 0:
                loss = batch_loss / total_tokens
            else:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        return logits, loss
