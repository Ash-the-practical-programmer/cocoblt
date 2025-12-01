"""Training Loop for CocoBLT.

Features:
- Custom Data Collator (padding-free handling handled inside model, but we need to pass raw sequences)
- Mixed Precision (BF16)
- Gradient Accumulation
- WandB logging
"""

import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import yaml
from tqdm import tqdm

from src.coconutblt.model.cocoblt_transformer import CocoBLTTransformer

class TextDataset(Dataset):
    """Simple byte dataset."""
    def __init__(self, data_path, seq_len=1024):
        # For prototype, generate random data or load simple text
        self.seq_len = seq_len
        self.data = torch.randint(1, 255, (1000, seq_len), dtype=torch.long) # Dummy data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # Causal modeling: input=x, label=x shifted?
        # Model takes input_ids and labels.
        # Usually input: x[:-1], label: x[1:]
        return x[:-1], x[1:]

def collate_fn(batch):
    # Batch is list of (input, label)
    # Since model handles variable lengths via patcher, we can stack them or pad standardly.
    # The patcher takes (B, S).
    # If sequences are same length (from Dataset), we can stack.
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels

class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CocoBLTTransformer(self.config).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=self.config['training']['weight_decay']
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config['training']['mixed_precision'] == "fp16"))
        self.bf16 = (self.config['training']['mixed_precision'] == "bf16")

    def train(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2
        )

        self.model.train()
        grad_accum = self.config['training']['grad_accum']
        max_steps = self.config['training']['max_steps']
        global_step = 0

        pbar = tqdm(total=max_steps)

        while global_step < max_steps:
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.bf16 else torch.float16, enabled=True):
                    logits, loss = self.model(inputs, labels)
                    loss = loss / grad_accum

                if self.bf16:
                    loss.backward()
                else:
                    self.scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum == 0:
                    if self.bf16:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                    global_step += 1
                    pbar.update(1)
                    pbar.set_description(f"Loss: {loss.item() * grad_accum:.4f}")

                    if global_step >= max_steps:
                        break

        print("Training complete.")
        torch.save(self.model.state_dict(), "cocoblt_final.pt")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/model/cocoblt_1b.yaml"

    trainer = Trainer(config_path)
    # Dummy dataset for running immediately
    ds = TextDataset(None)
    trainer.train(ds)
