"""Entropy Calibration Tool.

Analyzes patch length distributions for different entropy thresholds.
Finds threshold that yields target average patch length (e.g. 4.5).
"""

import torch
import numpy as np

class EntropyCalibrator:
    def __init__(self, predictor):
        self.predictor = predictor

    def simulate_patching(self, scores_list, threshold, min_len=1, max_len=8):
        """
        Simulate patching on a list of score tensors.
        scores_list: list of (S,) tensors. scores[i] is entropy of x[i+1].

        Returns:
            avg_len: float
            lengths: list of ints
        """
        all_lens = []
        for scores in scores_list:
            S = scores.shape[0]
            i = 0
            while i < S:
                size = min(min_len, S - i)
                while size < max_len and (i + size) < S:
                    # Check entropy at boundary (i + size - 1)
                    # We want to check if the NEXT byte is predictable.
                    # scores[k] is entropy of byte k+1 given 0..k.
                    # The boundary is after byte `i+size-1`.
                    # We look at `scores[i+size-1]`.

                    ent = scores[i + size - 1].item()
                    if ent < threshold:
                        size += 1
                    else:
                        break
                all_lens.append(size)
                i += size

        if not all_lens:
            return 0.0, []

        return sum(all_lens) / len(all_lens), all_lens

    def calibrate(self, dataset_sample, target_avg=4.5, tolerance=0.1):
        """
        dataset_sample: List[torch.Tensor] - raw bytes sequences
        """
        print("Computing scores for calibration sample...")
        scores_list = []
        # Process in batches
        batch_size = 16
        for i in range(0, len(dataset_sample), batch_size):
            batch = dataset_sample[i:i+batch_size]
            # Pad for batching
            max_len = max(len(b) for b in batch)
            padded = torch.zeros(len(batch), max_len, dtype=torch.long)
            for j, b in enumerate(batch):
                padded[j, :len(b)] = b

            with torch.no_grad():
                batch_scores = self.predictor.compute_scores(padded)

            # Unpad
            for j, b in enumerate(batch):
                scores_list.append(batch_scores[j, :len(b)])

        print("Searching for optimal threshold...")
        low = 0.0
        high = 8.0 # Max entropy (8 bits)
        best_thresh = 4.0
        best_diff = float('inf')

        # Binary searchish
        for _ in range(20):
            mid = (low + high) / 2
            avg, _ = self.simulate_patching(scores_list, mid)

            diff = abs(avg - target_avg)
            if diff < best_diff:
                best_diff = diff
                best_thresh = mid

            print(f"Thresh: {mid:.4f} -> Avg Len: {avg:.4f}")

            if abs(avg - target_avg) < tolerance:
                break

            if avg < target_avg:
                # Patches too small -> Need to grow more -> Need HIGHER threshold (easier to be < thresh)
                # Wait. condition: ent < thresh -> continue.
                # If we increase thresh, condition ent < thresh is met MORE often.
                # So we grow MORE. Patch size increases.
                # So if avg < target (too small), we need to INCREASE thresh.
                low = mid
            else:
                # Patches too big -> Need to break more -> Lower threshold
                high = mid

        return best_thresh

    def plot_histogram(self, lengths):
        # Text based histogram
        counts = torch.bincount(torch.tensor(lengths))
        total = counts.sum().item()
        print("\nPatch Length Distribution:")
        for i, count in enumerate(counts):
            if i == 0: continue
            if count == 0: continue
            pct = (count / total) * 100
            bar = '#' * int(pct / 2)
            print(f"Len {i}: {count:5d} ({pct:5.1f}%) | {bar}")

if __name__ == "__main__":
    # Dummy test
    from src.coconutblt.entropy.entropy_predictor import EntropyPredictor
    pred = EntropyPredictor() # random
    dummy_data = [torch.randint(0, 256, (512,)) for _ in range(10)]
    calib = EntropyCalibrator(pred)
    th = calib.calibrate(dummy_data)
    print(f"Optimal threshold: {th}")
    _, lens = calib.simulate_patching([torch.rand(512)*8 for _ in range(10)], th)
    calib.plot_histogram(lens)
