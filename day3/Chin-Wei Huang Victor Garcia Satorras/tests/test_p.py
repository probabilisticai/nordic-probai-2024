import pytest 
import torch

from probai.src.models.ddpm import DDPM
from probai.src.models.gt.ddpm import DDPM as GT


@pytest.fixture
def model() -> torch.nn.Module:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, z_t, t, **kwargs):
            return z_t
    return Model()

@pytest.fixture
def ddpm(model: torch.nn.Module) -> DDPM:
    return DDPM(model)


@pytest.fixture
def ddpm_gt(model: torch.nn.Module) -> GT:
    return GT(model)



@pytest.fixture
def model_gnn() -> torch.nn.Module:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, z_t, t, edge_index, batch, context):
            assert edge_index == 2, "edge_index not correctly passed to model"
            assert batch == 3, "batch not correctly passed to model"
            assert context == 4, "context not correctly passed to model"
            return z_t
    return Model()

@pytest.fixture
def ddpm_gnn(model_gnn: torch.nn.Module) -> DDPM:
    return DDPM(model_gnn)


@pytest.fixture
def ddpm_gt_gnn(model_gnn: torch.nn.Module) -> GT:
    return GT(model_gnn)


def test_mean(ddpm: DDPM, ddpm_gt: GT, ddpm_gnn: DDPM, ddpm_gt_gnn: GT):
    
    z = torch.ones(2, 1)
    t = torch.ones(2, dtype=torch.long)
    edge_index = 2
    batch = 3
    context = 4
    assert torch.allclose(
        ddpm._p_mean(z, t, edge_index, batch, context), 
        ddpm_gt._p_mean(z, t, edge_index, batch, context)
    ), "_p_mean is incorrectly implemented"
    
    
    assert torch.allclose(
        ddpm_gnn._p_mean(z, t, edge_index, batch, context), 
        ddpm_gt_gnn._p_mean(z, t, edge_index, batch, context)
    ), "arguments not correctly passed to model"
    
    
def test_std(ddpm: DDPM, ddpm_gt: GT):
    
    z = torch.ones(2, 1)
    t = torch.ones(2, dtype=torch.long)
    assert torch.allclose(ddpm._p_std(z, t), ddpm_gt._p_std(z, t)), (
        "_q_std is incorrectly implemented"
    ) 