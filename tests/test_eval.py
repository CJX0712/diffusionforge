import numpy as np

from diffusionforge.eval.metrics import (
    mmd,
    wasserstein2,
    mode_coverage,
    energy_distance,
    evaluate_samples,
)
from diffusionforge.data.toys import sample_toy


def test_mmd_self_near_zero():
    x = np.random.default_rng(0).normal(size=(400, 2))
    assert mmd(x, x) < 1e-6


def test_mmd_disjoint_positive():
    x = np.random.default_rng(0).normal(0.0, 1.0, size=(400, 2))
    y = np.random.default_rng(1).normal(8.0, 1.0, size=(400, 2))
    assert mmd(x, y) > 0.1


def test_mode_coverage_known():
    modes = np.array([[0.0, 0.0], [5.0, 5.0]])
    # 仅 [0.1,0.0] 落在第一个模式半径内 → 覆盖 1/2 = 0.5
    samples = np.array([[0.1, 0.0], [100.0, 100.0]])
    cov = mode_coverage(samples, modes, radius=1.0)
    assert abs(cov - 0.5) < 1e-9


def test_wasserstein2_finite():
    x = np.random.default_rng(0).normal(size=(300, 2))
    y = np.random.default_rng(1).normal(size=(300, 2))
    w2 = wasserstein2(x, y)
    assert w2 >= 0 and np.isfinite(w2)


def test_energy_distance_nonneg():
    x = np.random.default_rng(0).normal(size=(300, 2))
    y = np.random.default_rng(1).normal(size=(300, 2))
    ed = energy_distance(x, y)
    assert ed >= -1e-9


def test_evaluate_samples_returns_result():
    toy = sample_toy("gaussians25", n=1000, seed=2)
    # 用真值自身评测：指标应接近最优（同分布 → MMD 很小，覆盖率高）
    ev = evaluate_samples(toy["samples"][:400], toy)
    assert ev.mmd < 0.1
    assert ev.mode_coverage > 0.9
    assert ev.n_samples == 400
