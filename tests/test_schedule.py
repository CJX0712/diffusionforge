import numpy as np
import pytest

from diffusionforge.core.errors import NumericalError
from diffusionforge.schedules.schedule import (
    build_scheduler,
    check_schedule_invariants,
    LinearSchedule,
    CosineSchedule,
)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedule_invariants_pass(kind):
    sched = build_scheduler(kind, timesteps=200, data_dim=2)
    check_schedule_invariants(sched)  # 不抛异常即通过


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_acp_monotonic(kind):
    sched = build_scheduler(kind, timesteps=200)
    acp = sched.alphas_cumprod()
    assert np.all(np.diff(acp) <= 1e-9)


def test_add_noise_endpoints():
    for cls in (LinearSchedule, CosineSchedule):
        sched = cls(timesteps=100, data_dim=2)
        x0 = np.ones((4, 2))
        # t=0 闭式：x_t == sqrt_acp[0]·x0
        xt0 = sched.add_noise(x0, 0, np.zeros((4, 2)))
        assert np.allclose(xt0, sched.sqrt_alphas_cumprod()[0] * x0, atol=1e-9)
        # t=T-1 闭式：x_t == sqrt_1macp[T-1]·noise
        noise = np.full((4, 2), 3.0)
        xtl = sched.add_noise(np.zeros((4, 2)), 99, noise)
        assert np.allclose(xtl, sched.sqrt_one_minus_alphas_cumprod()[99] * noise, atol=1e-9)


def test_posterior_variance_nonneg():
    for cls in (LinearSchedule, CosineSchedule):
        sched = cls(timesteps=100)
        assert sched.posterior_variance().min() >= -1e-12


def test_extract_broadcast():
    sched = LinearSchedule(timesteps=50)
    arr = sched.betas()
    ext = sched.extract(arr, 7, batch=5)
    assert ext.shape == (5, 1)
    assert np.allclose(ext, arr[7])
