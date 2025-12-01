
import os
import torch
import pytest
import numpy as np
from src.coconutblt.data.byte_dataset import ByteSequenceDataset
from src.coconutblt.entropy.entropy_predictor import EntropyPredictor
from src.coconutblt.data.patch_dataloader import PatchCollator, PatchedDataset
from src.coconutblt.patching.entropy_patcher import AdaptivePatcher

@pytest.fixture
def dummy_bin(tmp_path):
    p = tmp_path / "test.bin"
    # Write 10KB of random bytes
    with open(p, "wb") as f:
        f.write(os.urandom(10240))
    return str(p)

def test_byte_dataset(dummy_bin):
    # Disable shuffle to test order
    ds = ByteSequenceDataset(dummy_bin, seq_len=128, stride=64, seed=42)
    # Note: Dataset shuffles by default.
    # To test sequential access, we should perhaps check if shuffle can be disabled or inspect internal indices.
    # Or just verify that the data we got matches the offset.

    x, off = ds[0]
    assert x.shape == (128,)
    assert x.dtype == torch.long

    # Read manually to verify
    with open(dummy_bin, "rb") as f:
        f.seek(off)
        data = f.read(128)
        arr = np.frombuffer(data, dtype=np.uint8)
        assert np.array_equal(x.numpy(), arr)

def test_patch_collator():
    # Mock patcher
    patcher = AdaptivePatcher(min_patch_size=1, max_patch_size=4)
    collator = PatchCollator(patcher)

    # Batch: [(bytes, scores)]
    # 2 items
    b1 = torch.randint(0, 255, (32,))
    b2 = torch.randint(0, 255, (32,))
    # No scores
    batch = [(b1, None), (b2, None)]

    out = collator(batch)

    # Check keys
    assert "flat_patches" in out
    assert "flat_labels" in out
    assert "cu_seqlens" in out

    # Check shapes
    flat = out["flat_patches"]
    assert flat.dim() == 2
    assert flat.shape[1] == 4 # Max patch size

    lbls = out["flat_labels"]
    assert lbls.shape == flat.shape

    cu = out["cu_seqlens"]
    assert len(cu) == 3 # 0, len1, len1+len2
    assert cu[-1] == flat.shape[0]

def test_entropy_predictor_training():
    pred = EntropyPredictor(device='cpu')
    # Dummy iterator
    data = [torch.randint(0, 256, (4, 128)) for _ in range(2)]

    # Should not crash
    pred.train_on_bytes(data, steps=2, save_path="/tmp/test_ent.pt")
    assert os.path.exists("/tmp/test_ent.pt")
