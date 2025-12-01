"""Compute entropy scores for a binary dataset.

Loads bin file, runs EntropyPredictor, saves scores to HDF5.
"""

import os
import argparse
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.coconutblt.data.byte_dataset import ByteSequenceDataset
from src.coconutblt.entropy.entropy_predictor import EntropyPredictor

def compute_scores(bin_path, model_path, out_path, stride=2048):
    print(f"Computing scores for {bin_path}...")

    # We use stride=seq_len to cover file non-overlappingly for scoring cache
    ds = ByteSequenceDataset(bin_path, seq_len=stride, stride=stride, split="train", val_ratio=0.0)
    # Use full dataset
    ds.indices = np.arange(ds.total_seqs)

    loader = DataLoader(ds, batch_size=64, num_workers=4, shuffle=False)

    predictor = EntropyPredictor(model_path)

    # Estimate size
    n_seqs = len(ds)

    with h5py.File(out_path, "w") as f:
        # Create dataset
        dset = f.create_dataset("scores", (n_seqs, stride), dtype='f2') # float16

        idx = 0
        for x, offsets in tqdm(loader):
            scores = predictor.compute_scores(x) # (B, S)
            B = scores.shape[0]

            # Write to HDF5
            dset[idx : idx+B] = scores.numpy().astype('float16')
            idx += B

    print(f"Saved scores to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=str, required=True)
    parser.add_argument("--model", type=str, default=None) # None = random weights
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    compute_scores(args.bin, args.model, args.out)
