import numpy as np

from diffusionforge.core.types import TrainingConfig
from diffusionforge.data.toys import sample_toy
from diffusionforge.models.numpy_mlp import NumpyMLP
from diffusionforge.sampling.samplers import DDPMSampler, DDIMSampler, build_sampler
from diffusionforge.schedules.schedule import build_scheduler
from diffusionforge.training.trainer import train_model


def _trained_model(seed=0, epochs=6):
    m = NumpyMLP(data_dim=2, hidden=16, depth=2, seed=seed)
    toy = sample_toy("moons", n=300, seed=1)
    sched = build_scheduler("cosine", timesteps=50, data_dim=2)
    train_model(m, sched, toy["samples"], TrainingConfig(n_train=300, epochs=epochs, batch_size=64, seed=seed), verbose=False)
    return m, sched


def test_ddpm_sampler_shape_finite():
    m, sched = _trained_model()
    s = DDPMSampler()
    out = s.sample(m, sched, n=50, seed=0)
    assert out.shape == (50, 2)
    assert np.all(np.isfinite(out))


def test_ddim_sampler_shape_finite():
    m, sched = _trained_model()
    s = DDIMSampler(steps=20)
    out = s.sample(m, sched, n=50, seed=0)
    assert out.shape == (50, 2)
    assert np.all(np.isfinite(out))


def test_build_sampler_unknown():
    import pytest
    with pytest.raises(ValueError):
        build_sampler("nope")
