import pytest 
import torch

from probai.src.models.ddpm import DDPM
from probai.src.models.gt.ddpm import DDPM as GT


@pytest.mark.parametrize("N", [10, 100, 1000])
@pytest.mark.parametrize("type", ["linear", "cosine"])                         
def test_coefs_absolute(N: int, type: str):
    for sol, gt, coef in zip(
        DDPM.get_coefs(N, type), GT.get_coefs(N, type), ["betas", "alphas", "alpha_bars"]
    ):
        assert torch.allclose(sol, gt, atol=1e-6), f"{coef} are different"


@pytest.mark.parametrize("type", ["linear", "cosine"])
def test_coefs_relative(type: str):
    betas, alphas, alpha_bars = DDPM.get_coefs(1000, type)
    assert torch.allclose(alphas, 1 - betas), "alphas are not equal to 1 - betas"
    assert torch.allclose(alpha_bars, torch.cumprod(alphas, dim=0)), (
        "alpha_bars are not equal to the cumulative product of alphas"
    )
    
    
    