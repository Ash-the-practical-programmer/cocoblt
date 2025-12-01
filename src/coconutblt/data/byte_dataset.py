"""Core Data Preprocessing.

Handles:
- ByteSequenceDataset (MMap wrapper)
"""

import os
import mmap
import torch
import numpy as np
from torch.utils.data import Dataset

class ByteSequenceDataset(Dataset):
    """
    Dataset that reads a binary file as a sequence of bytes.
    Supports random access via mmap.
    """
    def __init__(self, bin_path, seq_len=2048, stride=None, split="train", val_ratio=0.05, seed=42):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.stride = stride or seq_len

        if not os.path.exists(bin_path):
            # Create dummy file if not exists for testing (or raise error)
            # raising error is better for production but for dev flow...
            raise FileNotFoundError(f"{bin_path} not found.")

        # Determine file size
        self.file_size = os.path.getsize(bin_path)

        # Calculate total sequences
        # num_seqs = (size - seq_len) // stride + 1
        self.total_seqs = max(0, (self.file_size - seq_len) // self.stride + 1)

        # Split
        indices = np.arange(self.total_seqs)
        # Shuffle conceptually (we can't shuffle mmap easily, so we shuffle indices)
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

        split_idx = int(self.total_seqs * (1 - val_ratio))
        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        self.mmap_obj = None
        self.file_obj = None

    def _init_mmap(self):
        if self.mmap_obj is None:
            self.file_obj = open(self.bin_path, "rb")
            self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._init_mmap()

        real_idx = self.indices[idx]
        start_byte = real_idx * self.stride
        end_byte = start_byte + self.seq_len

        # Read bytes
        # mmap slicing returns bytes
        byte_data = self.mmap_obj[start_byte:end_byte]

        # Convert to tensor
        # frombuffer is fast
        # Note: torch.frombuffer requires writable buffer usually? No, copy=True if not.
        # Use numpy then torch
        arr = np.frombuffer(byte_data, dtype=np.uint8)
        # Copy to ensure we don't have mmap issues with concurrency if loaders use threading?
        # Creating tensor copies data by default unless using from_blob (c++).
        # torch.tensor(arr) copies.
        x = torch.tensor(arr, dtype=torch.long) # Embeddings need long

        return x, start_byte

    def close(self):
        if self.mmap_obj:
            self.mmap_obj.close()
        if self.file_obj:
            self.file_obj.close()
