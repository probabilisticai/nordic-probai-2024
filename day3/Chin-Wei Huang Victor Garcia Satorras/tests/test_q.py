import pytest 
import torch

from probai.src.models.ddpm import DDPM
from probai.src.models.gt.ddpm import DDPM as GT


@pytest.fixture
def ddpm() -> DDPM:
    return DDPM(None)


@pytest.fixture
def ddpm_gt() -> GT:
    return GT(None)


def test_mean(ddpm: DDPM, ddpm_gt: GT):
    
    z = torch.ones(2, 1)
    t = torch.ones(2, dtype=torch.long)
    assert torch.allclose(ddpm._q_mean(z, t), ddpm_gt._q_mean(z, t)), (
        "_q_mean is incorrectly implemented"
    )
    

def test_std(ddpm: DDPM, ddpm_gt: GT):
    
    z = torch.ones(2, 1)
    t = torch.ones(2, dtype=torch.long)
    assert torch.allclose(ddpm._q_std(z, t), ddpm_gt._q_std(z, t)), (
        "_q_std is incorrectly implemented"
    ) 