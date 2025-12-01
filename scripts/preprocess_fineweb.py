"""Preprocess FineWeb-Edu sample.

Downloads sample, encodes to bytes, saves as .bin.
"""

import os
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def preprocess(output_path, samples=100000):
    print(f"Loading FineWeb-Edu sample ({samples} docs)...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    # We write to a binary file
    with open(output_path, "wb") as f:
        count = 0
        pbar = tqdm(total=samples)
        for item in ds:
            text = item['text']
            # UTF-8 encode
            b = text.encode("utf-8")
            # Write raw bytes
            f.write(b)
            # Add a separator? Usually 0 byte or just concat.
            # GPT training usually just concats with EOS token.
            # We can use 0 as EOS if we want.
            f.write(b'\0')

            count += 1
            pbar.update(1)
            if count >= samples:
                break
    print(f"Finished. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/fineweb_sample.bin")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    preprocess(args.out, args.samples)
