"""Entropy-driven byte patcher module.

Implements:
1. EntropyCalculator/LearnedEntropyEstimator
2. AdaptivePatcher (segmentation)
3. PatchEncoder (concatenation + MLP + Hash Positional Encoding)
"""

import math
from typing import List, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F

class LearnedEntropyEstimator(nn.Module):
    """Small model to predict next-byte entropy, used for boundary decisions.

    Can be pretrained and frozen.
    """
    def __init__(self, vocab_size: int = 256, emb_dim: int = 64, hidden: int = 128, kernel: int = 5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.conv = nn.Conv1d(emb_dim, hidden, kernel_size=kernel, padding=kernel // 2)
        self.act = nn.GELU()
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, S)
        emb = self.emb(tokens)  # (B, S, E)
        x = emb.transpose(1, 2)  # (B, E, S)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, S, hidden)
        x = self.act(x)
        out = self.head(x).squeeze(-1) # (B, S)
        return out


def shannon_entropy_of_window(window: torch.Tensor) -> float:
    if window.numel() == 0:
        return 0.0
    vals, counts = torch.unique(window, return_counts=True)
    probs = counts.float() / counts.sum()
    ent = -(probs * torch.log2(probs)).sum().item()
    return float(ent)


class AdaptivePatcher(nn.Module):
    """
    Segments a byte sequence into patches based on entropy.
    """
    def __init__(self,
                 min_patch_size: int = 1,
                 max_patch_size: int = 8,
                 lookahead: int = 4,
                 entropy_threshold: float = 0.5,
                 pad_token: int = 0,
                 estimator: Optional[nn.Module] = None):
        super().__init__()
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.lookahead = lookahead
        self.entropy_threshold = entropy_threshold
        self.pad_token = pad_token
        self.estimator = estimator

    def _patch_one_sequence(self, seq: torch.Tensor, preds: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[int]]:
        """
        Returns:
            patches_flat: Tensor of shape (num_patches * max_patch_size) containing the padded patches flattened.
                          Wait, better to return (num_patches, max_patch_size).
            global_offsets: List[int] starting byte index for each patch.
        """
        S = seq.shape[0]
        i = 0
        patches_list = []
        global_offsets = []

        while i < S:
            size = min(self.min_patch_size, S - i)

            # Grow
            while size < self.max_patch_size and (i + size) < S:
                start_look = i + size

                if preds is not None:
                    ent = float(preds[start_look].item())
                else:
                    end_look = min(S, start_look + self.lookahead)
                    look = seq[start_look:end_look]
                    ent = shannon_entropy_of_window(look)

                if ent < self.entropy_threshold:
                    size += 1
                else:
                    break

            patch_segment = seq[i : i + size]
            # Pad to max_patch_size
            if size < self.max_patch_size:
                pad_len = self.max_patch_size - size
                # We use the pad_token.
                # Note: In the PatchEncoder, we might want a special mask or handle padding explicitly.
                # For now, standard padding.
                padding = torch.full((pad_len,), self.pad_token, dtype=seq.dtype, device=seq.device)
                patch_padded = torch.cat([patch_segment, padding], dim=0)
            else:
                patch_padded = patch_segment

            patches_list.append(patch_padded)
            global_offsets.append(i)
            i += size

        if not patches_list:
             return torch.empty((0, self.max_patch_size), dtype=seq.dtype, device=seq.device), []

        return torch.stack(patches_list), global_offsets

    def forward(self, tokens: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        """
        Args:
            tokens: (B, S)
        Returns:
            all_patches: List[Tensor] of shape (num_patches_i, max_patch_size) for each batch.
            all_offsets: List[Tensor] of shape (num_patches_i,) containing global byte offsets.
            cu_seqlens: List[int] cumulative sequence lengths (in number of patches) for flash attention.
                        [0, n_p1, n_p1+n_p2, ...]
        """
        B, S = tokens.shape
        if self.estimator is not None:
            with torch.no_grad():
                preds = self.estimator(tokens)
        else:
            preds = None

        all_patches_tensors = []
        all_offsets_tensors = []
        cu_seqlens = [0]

        for b in range(B):
            seq = tokens[b]
            p_preds = preds[b] if preds is not None else None
            # Filter out padding in input sequence if any (assuming pad_token at end)
            # Actually, the model should probably handle exact lengths via cu_seqlens if passed,
            # but usually 'tokens' here is a dense tensor.
            # Let's assume the user handles masking or we treat 0 as valid unless trailing.
            # For simplicity, we process the whole row `seq`.

            patches, offsets = self._patch_one_sequence(seq, p_preds)

            all_patches_tensors.append(patches)
            all_offsets_tensors.append(torch.tensor(offsets, device=tokens.device, dtype=torch.long))
            cu_seqlens.append(cu_seqlens[-1] + len(patches))

        return all_patches_tensors, all_offsets_tensors, cu_seqlens


class PatchEncoder(nn.Module):
    """
    Encodes patches:
    1. Embed bytes (256 -> 512)
    2. Concatenate 8 bytes -> 4096
    3. MLP -> 2048
    4. Add Hash-based Positional Encoding
    """
    def __init__(self,
                 dim_model: int = 2048,
                 byte_emb_dim: int = 512,
                 max_patch_size: int = 8,
                 vocab_size: int = 256,
                 hash_table_size: int = 100000): # large table for hashing
        super().__init__()
        self.dim_model = dim_model
        self.max_patch_size = max_patch_size
        self.byte_emb_dim = byte_emb_dim

        self.byte_embedding = nn.Embedding(vocab_size, byte_emb_dim)

        input_dim = max_patch_size * byte_emb_dim
        self.projector = nn.Sequential(
            # MLP: Input (4096) -> Hidden (4096) -> Output (2048)
            nn.Linear(input_dim, dim_model * 2),
            nn.GELU(),
            nn.Linear(dim_model * 2, dim_model)
        )

        # Hash-based positional encoding
        self.hash_table_size = hash_table_size
        self.pos_table = nn.Parameter(torch.randn(hash_table_size, dim_model) * 0.02)

    def forward(self, patches: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (N_total_patches, max_patch_size) - Byte IDs
            offsets: (N_total_patches) - Global byte offset for each patch
        Returns:
            encoded: (N_total_patches, dim_model)
        """
        # 1. Embed bytes
        x = self.byte_embedding(patches) # (N, 8, 512)

        # 2. Flatten/Concat
        N = x.shape[0]
        x = x.view(N, -1) # (N, 8*512) = (N, 4096)

        # 3. Project
        h = self.projector(x) # (N, 2048)

        # 4. Add Positional Encoding
        # Simple hash: offset % table_size
        pos_ids = offsets % self.hash_table_size
        pos_embs = self.pos_table[pos_ids] # (N, 2048)

        return h + pos_embs
