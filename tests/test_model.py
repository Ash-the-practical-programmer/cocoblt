
import torch
import pytest
import yaml
from src.coconutblt.model.cocoblt_transformer import CocoBLTTransformer

@pytest.fixture
def model_config():
    with open("configs/model/cocoblt_1b.yaml", 'r') as f:
        config = yaml.safe_load(f)
    # Scale down for test
    config['model']['n_layers'] = 2
    config['model']['dim'] = 128
    config['model']['n_heads'] = 4
    config['reasoning']['latent_interval'] = 1 # insert every layer
    config['reasoning']['latent_dim'] = 128
    return config

@pytest.mark.skipif(not torch.cuda.is_available(), reason="xformers needs CUDA")
def test_full_forward_pass(model_config):
    device = torch.device("cuda")
    model = CocoBLTTransformer(model_config).to(device)

    B, S = 2, 64
    input_ids = torch.randint(1, 255, (B, S), device=device)
    labels = torch.randint(1, 255, (B, S), device=device)

    logits, loss = model(input_ids, labels)

    # logits: (Total_Patches, 8, 256)
    # We don't know total patches exactly, but check dims
    assert len(logits.shape) == 3
    assert logits.shape[1] == 8
    assert logits.shape[2] == 256

    assert loss is not None
    assert loss.item() > 0

    loss.backward()
