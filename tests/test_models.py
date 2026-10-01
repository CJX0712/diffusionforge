import numpy as np
import pytest

from diffusionforge.core.errors import NumericalError
from diffusionforge.data.toys import sample_toy
from diffusionforge.models.numpy_mlp import NumpyMLP
from diffusionforge.schedules.schedule import build_scheduler
from diffusionforge.training.trainer import train_model
from diffusionforge.core.types import TrainingConfig


def test_predict_noise_shape():
    m = NumpyMLP(data_dim=2, hidden=16, depth=3, seed=0)
    m.T = 50
    x = np.random.default_rng(0).normal(size=(7, 2))
    t = np.array([1, 5, 9, 2, 0, 49, 10])
    out = m.predict_noise(x, t)
    assert out.shape == (7, 2)


def test_parameters_count():
    m = NumpyMLP(data_dim=2, hidden=32, depth=4, seed=0)
    assert m.parameters_count() > 0


def test_backward_matches_finite_diff():
    m = NumpyMLP(data_dim=2, hidden=8, depth=2, seed=0)
    m.T = 10
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 2))
    t = np.array([1, 2, 3])
    target = rng.normal(size=(3, 2))
    _, gW, _ = m.loss_and_grad(x, t, target)
    eps = 1e-5
    orig = m.W[0][0, 0]
    m.W[0][0, 0] = orig + eps
    lp = m.predict_loss(x, t, target)
    m.W[0][0, 0] = orig - eps
    lm = m.predict_loss(x, t, target)
    m.W[0][0, 0] = orig
    numeric = (lp - lm) / (2 * eps)
    assert abs(numeric - gW[0][0, 0]) < 1e-3


def test_training_reduces_loss():
    m = NumpyMLP(data_dim=2, hidden=64, depth=3, seed=0)
    toy = sample_toy("gaussians25", n=400, seed=1)
    sched = build_scheduler("cosine", timesteps=20, data_dim=2)
    cfg = TrainingConfig(n_train=400, epochs=30, batch_size=64, seed=0)
    train_model(m, sched, toy["samples"], cfg, verbose=False)
    assert m.history[-1] < m.history[0]
    assert len(m.history) == cfg.epochs


def test_t_requires_binding():
    m = NumpyMLP(data_dim=2, seed=0)
    m.T = None
    with pytest.raises(NumericalError):
        m.predict_noise(np.zeros((2, 2)), np.array([0, 1]))
