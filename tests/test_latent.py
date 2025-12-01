
import torch
import pytest
from src.coconutblt.reasoning.latent_module import ContinuousLatentModule

@pytest.mark.skipif(not torch.cuda.is_available(), reason="xformers usually needs CUDA")
def test_latent_module_shapes():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    B = 2
    Total_Tokens = 20 # 10 per batch
    D = 128

    tokens = torch.randn(Total_Tokens, D, device=device)
    cu_seqlens = torch.tensor([0, 10, 20], dtype=torch.int32, device=device)
    max_seqlen = 10

    latent_mod = ContinuousLatentModule(n_latents=4, latent_dim=D, n_heads=4).to(device)

    # Init latents
    latents = torch.randn(B, 4, D, device=device)

    new_tokens, new_latents = latent_mod(tokens, latents, cu_seqlens, max_seqlen)

    assert new_tokens.shape == tokens.shape
    assert new_latents.shape == latents.shape
    assert not torch.isnan(new_tokens).any()
