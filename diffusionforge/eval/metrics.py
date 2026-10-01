"""生成质量评测指标（越小越好者已注明）。

- mmd：RBF 核最大均值差异（中位数启发式带宽），越小越好，MMD(x,x)≈0。
- wasserstein2：2D 经验 Wasserstein-2 距离（等样本最优匹配），越小越好。
- mode_coverage：真实模式中被生成样本覆盖的比例 ∈[0,1]，越大越好。
- energy_distance：能量距离（分布间散度），越小越好。

所有指标对样本集合大小不敏感（子采样到共同规模），可在 (数据集×配置) 间公平比较。
"""

import numpy as np
import scipy.optimize as opt

from ..core.types import EvalResult


def _rbf(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    Xn = np.sum(X ** 2, axis=1)
    Yn = np.sum(Y ** 2, axis=1)
    sq = Xn[:, None] + Yn[None, :] - 2.0 * (X @ Y.T)
    sq = np.maximum(sq, 0.0)
    return np.exp(-gamma * sq)


def mmd(x: np.ndarray, y: np.ndarray, gamma: float = None) -> float:
    """最大均值差异（MMD² 的非负估计）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if gamma is None:
        z = np.vstack([x, y])
        # 中位数启发式下采样，避免大 n 的 O(n²) 内存与溢出风险
        if z.shape[0] > 1500:
            rng = np.random.default_rng(0)
            z = z[rng.permutation(z.shape[0])[:1500]]
        d = np.sqrt(np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2))
        iu = np.triu_indices(z.shape[0], 1)
        med = np.median(d[iu]) if iu[0].size > 0 else 1.0
        gamma = 1.0 / (2.0 * (med ** 2) + 1e-12) if med > 0 else 1.0
    Kxx = _rbf(x, x, gamma)
    Kyy = _rbf(y, y, gamma)
    Kxy = _rbf(x, y, gamma)
    m, n = len(x), len(y)
    mmd2 = Kxx.sum() / (m * m) - 2.0 * Kxy.sum() / (m * n) + Kyy.sum() / (n * n)
    return float(max(mmd2, 0.0))


def wasserstein2(x: np.ndarray, y: np.ndarray) -> float:
    """2D 经验 Wasserstein-2：等规模下用线性指派求最优匹配成本。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m, n = len(x), len(y)
    if m != n:
        k = min(m, n)
        rng = np.random.default_rng(0)
        x = x[rng.permutation(m)[:k]]
        y = y[rng.permutation(n)[:k]]
        m = n = k
    diff = x[:, None, :] - y[None, :, :]
    C = np.sum(diff ** 2, axis=2)
    row, col = opt.linear_sum_assignment(C)
    cost = C[row, col].sum() / m
    return float(np.sqrt(cost))


def mode_coverage(x: np.ndarray, modes: np.ndarray, radius: float) -> float:
    """真实模式中被至少一个生成样本（半径内）覆盖的比例。"""
    x = np.asarray(x, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.float64)
    if modes.shape[0] == 0:
        return 0.0
    covered = 0
    for c in modes:
        d = np.linalg.norm(x - c, axis=1)
        if d.min() <= radius:
            covered += 1
    return covered / modes.shape[0]


def _mean_pair_distance(a: np.ndarray, b: np.ndarray, max_n: int = 1200) -> float:
    """样本间平均欧氏距离（确定性，无随机子采样），支持 a、b 长度不同。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] > max_n:
        a = a[:max_n]
    if b.shape[0] > max_n:
        b = b[:max_n]
    d = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
    return float(d.sum() / (a.shape[0] * b.shape[0]))


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """能量距离平方 d² = 2·E||X−Y|| − E||X−X'|| − E||Y−Y'||（≥0，越小越接近）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    exx = _mean_pair_distance(x, x)
    eyy = _mean_pair_distance(y, y)
    exy = _mean_pair_distance(x, y)
    return 2.0 * exy - exx - eyy


def evaluate_samples(samples: np.ndarray, toy: dict, n_eval: int = 2000) -> EvalResult:
    """对生成样本相对玩具真值分布做全面评测。"""
    samples = np.asarray(samples, dtype=np.float64)
    true = toy["samples"]
    rng = np.random.default_rng(12345)
    if len(samples) > n_eval:
        samples = samples[rng.permutation(len(samples))[:n_eval]]
    if len(true) > n_eval:
        true = true[rng.permutation(len(true))[:n_eval]]

    mmd_v = mmd(samples, true)
    w2 = wasserstein2(samples, true)
    ed = energy_distance(samples, true)
    if toy.get("has_modes", False) and toy["modes"].shape[0] > 0:
        mc = mode_coverage(samples, toy["modes"], toy.get("radius", 1.0))
    else:
        mc = 0.0
    return EvalResult(
        mmd=mmd_v,
        wasserstein2=w2,
        mode_coverage=mc,
        energy_distance=ed,
        n_samples=len(samples),
        has_modes=toy.get("has_modes", False),
    )
