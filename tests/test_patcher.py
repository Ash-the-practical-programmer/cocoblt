
import torch
import pytest
from src.coconutblt.patching.entropy_patcher import AdaptivePatcher, PatchEncoder

def test_entropy_patcher_shapes():
    B, S = 2, 128
    tokens = torch.randint(1, 255, (B, S))

    # Test Patcher
    patcher = AdaptivePatcher(min_patch_size=1, max_patch_size=8, lookahead=4)
    patches_list, offsets_list, cu_seqlens = patcher(tokens)

    assert len(patches_list) == B
    assert len(cu_seqlens) == B + 1
    assert cu_seqlens[0] == 0
    assert cu_seqlens[-1] == sum(len(p) for p in patches_list)

    for p in patches_list:
        assert p.shape[1] == 8 # Max patch size
        assert p.dtype == tokens.dtype

    # Test Encoder
    encoder = PatchEncoder(dim_model=128, byte_emb_dim=64, max_patch_size=8)

    patches_flat = torch.cat(patches_list, dim=0)
    offsets_flat = torch.cat(offsets_list, dim=0)

    encoded = encoder(patches_flat, offsets_flat)

    assert encoded.shape == (patches_flat.shape[0], 128)
    assert not torch.isnan(encoded).any()

def test_adaptive_behavior():
    # Construct sequence with low entropy (repeating) and high entropy (random)
    # Low entropy should produce larger patches.
    # High entropy should produce smaller patches.

    # Repeating 'A' (65)
    repeating = torch.full((1, 32), 65, dtype=torch.long)
    patcher = AdaptivePatcher(min_patch_size=1, max_patch_size=8, entropy_threshold=1.0)

    patches, _, _ = patcher(repeating)
    # Expect mostly max_patch_size patches because entropy is 0
    # 32 / 8 = 4 patches
    assert len(patches[0]) <= 6 # Allow some overhead but close to optimal

    # Random
    random_seq = torch.randint(0, 256, (1, 32))
    # Entropy will be high.
    patcher_strict = AdaptivePatcher(min_patch_size=1, max_patch_size=8, entropy_threshold=0.1) # very strict
    patches_rand, _, _ = patcher_strict(random_seq)

    # Should have more patches than repeating
    assert len(patches_rand[0]) > len(patches[0])
