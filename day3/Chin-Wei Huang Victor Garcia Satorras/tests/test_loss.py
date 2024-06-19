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


def test_loss(ddpm: DDPM, ddpm_gt: GT):
    
    epsilon_pred = torch.randn(2, 5)
    epsilon = torch.randn(2, 5)
    
    sol = ddpm._losses(epsilon_pred, epsilon)
    gt = ddpm_gt._losses(epsilon_pred, epsilon)
    
    # verify shape
    assert sol.shape == gt.shape, (
        "_losses has incorrect shape. "
        "remember to collapse across feature dimensions but not the batch dimension"
    )
    assert torch.allclose(
        sol, gt
    ), (
        "_losses is incorrectly implemented. " 
        "Note it's supposed to be sum of squared errors across all feature dimensions"
    )
