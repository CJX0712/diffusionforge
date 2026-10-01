"""2D 玩具分布生成器（确定性，可复现）。

每个分布返回 ``{samples, modes, has_modes, radius}``：
- samples: (N, 2) 真实目标样本（用于训练与评测的「真值分布」）。
- modes:   (K, 2) 真实模式中心（用于模式覆盖评测；has_modes=False 时为空）。
- radius:  模式覆盖判定半径（数据相关）。

覆盖的分布：25 高斯、瑞士卷、双月、双环、棋盘格、正弦曲线。难度梯度明确，
既有离散多峰（25 高斯 / 棋盘 / 双月）也有流形（瑞士卷 / 正弦）。
"""

import numpy as np

from ..core.errors import DataError

_TOY_META = {
    "gaussians25": {"radius": 1.0, "has_modes": True},
    "swissroll": {"radius": 0.0, "has_modes": False},
    "moons": {"radius": 0.35, "has_modes": True},
    "circles": {"radius": 0.0, "has_modes": False},
    "checkerboard": {"radius": 0.25, "has_modes": True},
    "sine": {"radius": 0.0, "has_modes": False},
}


def list_toys() -> list:
    return list(_TOY_META.keys())


def _gaussians25(rng: np.random.Generator, n: int):
    grid = np.linspace(-7.0, 7.0, 5)
    centers = np.array([[x, y] for x in grid for y in grid])  # (25,2)
    idx = rng.integers(0, 25, size=n)
    noise = rng.normal(0.0, 0.5, size=(n, 2))
    samples = centers[idx] + noise
    return samples, centers


def _swissroll(rng: np.random.Generator, n: int):
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n)
    x = t * np.cos(t) / (4.5 * np.pi)
    y = t * np.sin(t) / (4.5 * np.pi)
    samples = np.stack([x, y], axis=1)
    samples += rng.normal(0.0, 0.04, size=samples.shape)
    return samples, np.empty((0, 2))


def _moons(rng: np.random.Generator, n: int):
    half = n // 2
    t = rng.uniform(0.0, np.pi, size=half)
    top = np.stack([np.cos(t), np.sin(t)], axis=1)
    t2 = rng.uniform(0.0, np.pi, size=n - half)
    bottom = np.stack([1.0 - np.cos(t2), -np.sin(t2) + 0.6], axis=1)
    samples = np.concatenate([top, bottom], axis=0)
    samples += rng.normal(0.0, 0.05, size=samples.shape)
    modes = np.array([
        [np.cos(np.pi / 2), np.sin(np.pi / 2)],
        [1.0 - np.cos(np.pi / 2), -np.sin(np.pi / 2) + 0.6],
    ])
    return samples, modes


def _circles(rng: np.random.Generator, n: int):
    half = n // 2
    theta1 = rng.uniform(0.0, 2 * np.pi, size=half)
    inner = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    theta2 = rng.uniform(0.0, 2 * np.pi, size=n - half)
    outer = 2.0 * np.stack([np.cos(theta2), np.sin(theta2)], axis=1)
    samples = np.concatenate([inner, outer], axis=0)
    samples += rng.normal(0.0, 0.04, size=samples.shape)
    return samples, np.empty((0, 2))


def _checkerboard(rng: np.random.Generator, n: int):
    samples = []
    while len(samples) < n:
        cand = rng.uniform(-1.0, 1.0, size=2)
        if (int(np.floor(2 * cand[0])) + int(np.floor(2 * cand[1]))) % 2 == 0:
            samples.append(cand)
    samples = np.array(samples[:n])
    samples += rng.normal(0.0, 0.02, size=samples.shape)
    modes = []
    for i in (-2, -1, 0, 1):
        for j in (-2, -1, 0, 1):
            if (i + j) % 2 == 0:
                modes.append([(i + 0.5) / 2.0, (j + 0.5) / 2.0])
    return samples, np.array(modes)


def _sine(rng: np.random.Generator, n: int):
    x = rng.uniform(-4.0, 4.0, size=n)
    y = np.sin(x)
    samples = np.stack([x, y], axis=1)
    samples += rng.normal(0.0, 0.1, size=samples.shape)
    return samples, np.empty((0, 2))


_GENERATORS = {
    "gaussians25": _gaussians25,
    "swissroll": _swissroll,
    "moons": _moons,
    "circles": _circles,
    "checkerboard": _checkerboard,
    "sine": _sine,
}


def sample_toy(name: str, n: int = 5000, seed: int = 0) -> dict:
    """采样一个玩具分布。

    返回 ``{name, samples, modes, has_modes, radius, data_dim}``。
    """
    if name not in _GENERATORS:
        raise DataError(f"unknown toy distribution: {name}")
    rng = np.random.default_rng(seed)
    samples, modes = _GENERATORS[name](rng, int(n))
    meta = _TOY_META[name]
    return {
        "name": name,
        "samples": samples.astype(np.float64),
        "modes": modes.astype(np.float64),
        "has_modes": meta["has_modes"],
        "radius": meta["radius"],
        "data_dim": 2,
    }
