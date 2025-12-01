"""Entropy Predictor Model and Scorer.

Architecture:
- 2 Layer Transformer
- 512 Hidden Dim
- 8 Heads
- Causal Masking
- Output: 256-way probabilities
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class EntropyModel(nn.Module):
    def __init__(self, vocab_size=256, dim=512, n_layers=2, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(2048, dim) # Max seq len 2048

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim*4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # x: (B, S)
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)

        h = self.embed(x) + self.pos_embed(pos)

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), diagonal=1)

        h = self.transformer(h, mask=mask, is_causal=True)
        logits = self.head(self.ln_f(h))
        return logits

def calculate_entropy(logits):
    """
    Args:
        logits: (B, S, 256)
    Returns:
        entropy: (B, S) - Shannon entropy in bits
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    # Entropy = -sum(p * log_2(p)) = -sum(p * (ln(p)/ln(2)))
    # = -1/ln(2) * sum(p * ln(p))
    entropy = -torch.sum(probs * log_probs, dim=-1) / torch.log(torch.tensor(2.0))
    return entropy

class EntropyPredictor:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = EntropyModel().to(device)
        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"Loaded entropy model from {model_path}")
        else:
            print("Initialized fresh entropy model (random weights)")

    def train_on_bytes(self, byte_iterator, save_path="entropy_model.pt", steps=1000, batch_size=64):
        """Simple training loop for calibration."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.model.train()

        losses = []
        pbar = tqdm(range(steps), desc="Training Entropy Model")

        # byte_iterator yields (B, S) batches of bytes
        step = 0
        for batch in byte_iterator:
            if step >= steps:
                break

            x = batch.to(self.device) # (B, S)
            if x.shape[0] != batch_size:
                # Optional: handle last batch or force fixed size
                pass

            # Causal prediction: Input x[:, :-1], Target x[:, 1:]
            input_ids = x[:, :-1]
            targets = x[:, 1:]

            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            pbar.update(1)
            step += 1

        torch.save(self.model.state_dict(), save_path)
        print(f"Saved entropy model to {save_path}")

    @torch.no_grad()
    def compute_scores(self, byte_tensor):
        """
        Args:
            byte_tensor: (B, S)
        Returns:
            scores: (B, S) - Entropy of NEXT byte prediction.
                    Note: Position i contains entropy of P(x_{i+1} | x_{0:i}).
                    We align it so that score[i] is the "split cost" at i.
                    If we split at i, we start a new patch.
                    Wait, BLT uses entropy of *next* byte to decide if we can *continue* the patch.
                    "If entropy is low, grow."
                    So at index i (last byte of current patch candidate), we look at entropy of i+1.
                    The model predicts i+1 from 0..i.
                    So we want the entropy of the prediction at step i.
        """
        self.model.eval()
        x = byte_tensor.to(self.device)
        logits = self.model(x) # (B, S, 256) - logits[i] is prediction for x[i+1] (if trained causally??)
        # Wait. In standard causal training:
        # logits[i] depends on x[0...i]. It predicts x[i+1].
        # So logits[i] is the distribution for the byte *after* position i.

        # We need to return these entropies.
        # However, the last position S-1 predicts S (which doesn't exist in input).
        # We usually input (B, S) and get (B, S) logits.
        # Entropy[i] corresponds to "uncertainty of byte i+1".

        ent = calculate_entropy(logits)
        return ent.cpu()

if __name__ == "__main__":
    # Smoke test
    model = EntropyPredictor()
    dummy = torch.randint(0, 256, (2, 128))
    scores = model.compute_scores(dummy)
    print(f"Scores shape: {scores.shape}")
    print(f"Scores sample: {scores[0, :5]}")
