"""Entropy-Aware Patch DataLoader.

Features:
- Loads (Bytes, Scores) or just Bytes and computes Scores on the fly (or loads from cache).
- Segments into variable patches.
- Batches without padding (FlashAttention format).
"""

import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.coconutblt.patching.entropy_patcher import AdaptivePatcher

class PatchedDataset(Dataset):
    """
    Wraps a ByteSequenceDataset and an optional HDF5 scores file.
    """
    def __init__(self, byte_dataset, scores_path=None, predictor=None):
        self.byte_dataset = byte_dataset
        self.scores_path = scores_path
        self.predictor = predictor # Only for on-the-fly (slow)
        self.scores_h5 = None

    def __len__(self):
        return len(self.byte_dataset)

    def _init_h5(self):
        if self.scores_path and self.scores_h5 is None:
            self.scores_h5 = h5py.File(self.scores_path, 'r')

    def __getitem__(self, idx):
        # x: (S,) bytes
        # offset: global byte offset
        x, offset = self.byte_dataset[idx]

        scores = None
        if self.scores_path:
            self._init_h5()
            # Assuming HDF5 structure matches dataset indices exactly or keyed by offset?
            # Keying by string index is safer: f"{offset}"
            # But sequential integers is faster if we aligned them.
            # ByteDataset shuffles indices.
            # If we precomputed scores for *every* stride, we can look it up.
            # Let's assume scores_h5 is dataset "scores" shape (Total_Seqs, Seq_Len)
            # We need the original index.
            # ByteDataset gives us `real_idx` (which is `offset // stride`).
            real_idx = offset // self.byte_dataset.stride
            scores = torch.from_numpy(self.scores_h5["scores"][real_idx])

        return x, scores

class PatchCollator:
    def __init__(self, patcher: AdaptivePatcher):
        self.patcher = patcher

    def __call__(self, batch):
        """
        Args:
            batch: List of (bytes_tensor, scores_tensor or None)
        Returns:
            flat_patches: (Total_Patches, Max_Patch_Size)
            flat_offsets: (Total_Patches,)
            cu_seqlens: (B+1,)
            cu_latents: (B+1,) - Offset for latent indexing (if we had variable latents, but we have fixed N per batch?)
                        Actually, latents are inserted periodically. We just need batch boundaries.
            labels: (Total_Patches * Max_Patch_Size) ??? No, labels for training.
                    We usually predict NEXT byte.
                    If we patch input x, we need labels y corresponding to x[1:].

                    Handling labels with patching is tricky.
                    Option 1: The model output head produces bytes.
                    We need to align output with target bytes.

                    In `CocoBLTTransformer`, we pass `labels` as (B, S).
                    But here we are flattening B.

                    So we should also flatten labels?
                    Wait, if we pass (B, S) labels to model, model expects dense tensor?
                    No, our model `forward` logic handled `patches_list` from `input_ids`.
                    And then sliced `labels` (which were (B, S)) inside the loop.

                    BUT, here we are doing "Padding Free" batching.
                    So we pass `flat_patches`.
                    The model `forward` currently expects `input_ids` (B, S) and patches internally.

                    **Optimization**: The user wants "DataLoader output: (flattened_patches...)".
                    This means we move the Patching logic OUT of the model and INTO the DataLoader.
                    This is much more efficient (can be done on CPU workers).

                    So we need to modify the Model to accept `flat_patches` directly.
                    I will update the Collator to produce what the Model *should* consume.

                    I will return `flat_patches` and `flat_labels`.
                    `flat_labels` will be the byte sequence corresponding to the patches, shifted by 1.
                    Wait, strict causal masking on bytes means:
                    Input: Bytes[0..N-1]
                    Target: Bytes[1..N]

                    If we patch Bytes[0..N-1], we get Patches.
                    The labels are Bytes[1..N].
                    We can just return the raw bytes as labels, but flattened?
                    Yes. `flat_labels` = concatenation of all valid bytes in the batch.
                    Masking of padding inside patches is handled by the model (we decided padding is 0).
                    Wait, `flat_patches` are padded to 8 bytes.
                    The labels should align with the UNROLLED patches?

                    If model predicts 8 bytes per patch.
                    We need 8 label bytes per patch (some might be pad/ignore).

                    So `flat_labels` should be (Total_Patches, Max_Patch_Size).

        """
        # Separate inputs
        inputs_list = []
        scores_list = []
        for x, s in batch:
            inputs_list.append(x)
            scores_list.append(s)

        # Run patcher
        # self.patcher.forward expects (B, S)
        # But here inputs have different lengths? No, ByteDataset fixed length.
        # But we want to support variable.
        # Let's stack if equal length.

        # If scores are present, use them.
        # patcher._patch_one_sequence can take single.

        all_patches = []
        all_offsets = []
        all_labels = [] # We need labels shaped like patches

        cu_seqlens = [0]

        for i, seq in enumerate(inputs_list):
            s = scores_list[i]

            # Create Input/Label pair
            # Causal: Input = seq[:-1], Label = seq[1:]
            # But wait, patching consumes bytes.
            # If we feed seq[:-1] to patcher.

            input_bytes = seq[:-1]
            label_bytes = seq[1:]

            # Patch the input
            # patches: (N, 8), offsets: list
            patches, offsets = self.patcher._patch_one_sequence(input_bytes, preds=s[:-1] if s is not None else None)

            all_patches.append(patches)
            all_offsets.append(torch.tensor(offsets, dtype=torch.long))
            cu_seqlens.append(cu_seqlens[-1] + len(patches))

            # Construct Labels corresponding to patches
            # For each patch, we need the corresponding labels.
            # Patch j starts at offsets[j], has length L (before padding).
            # The labels for these bytes are label_bytes[offsets[j] : offsets[j]+L].
            # Then pad with -100 (ignore index).

            n_p = patches.shape[0]
            max_p = self.patcher.max_patch_size
            lbls = torch.full((n_p, max_p), -100, dtype=torch.long)

            for j in range(n_p):
                start = offsets[j]
                # Length of VALID bytes in this patch
                # We can count non-pad in patches[j] (assuming pad is 0)
                # Or re-calculate from offsets (next offset - curr offset)
                # Last patch needs explicit length handling.
                # simpler: count non-pad
                # valid_len = (patches[j] != self.patcher.pad_token).sum()
                # But what if byte 0 is valid?
                # We need exact length.
                if j < n_p - 1:
                    l = offsets[j+1] - offsets[j]
                else:
                    l = len(input_bytes) - offsets[j]

                # Copy labels
                # Labels corresponding to input_bytes[start : start+l]
                # are label_bytes[start : start+l]
                lbl_chunk = label_bytes[start : start+l]
                lbls[j, :len(lbl_chunk)] = lbl_chunk

            all_labels.append(lbls)

        # Cat everything
        flat_patches = torch.cat(all_patches, dim=0) # (Total_P, 8)
        flat_offsets = torch.cat(all_offsets, dim=0) # (Total_P,)
        flat_labels = torch.cat(all_labels, dim=0)   # (Total_P, 8)
        cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32)

        return {
            "flat_patches": flat_patches,
            "flat_offsets": flat_offsets,
            "flat_labels": flat_labels,
            "cu_seqlens": cu_seqlens_t
        }

def create_dataloader(dataset, batch_size=32, num_workers=4, patcher=None):
    collator = PatchCollator(patcher)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
