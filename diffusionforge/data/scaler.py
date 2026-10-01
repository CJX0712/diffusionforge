"""轻量 StandardScaler（零依赖）。

扩散模型对数据尺度敏感：不同玩具分布量级差异大（gaussians25 ~±8，moons ~±1.5），
不归一化会导致小尺度分布训练发散。在标准化空间内训练/采样，生成后再逆变换回原空间，
与真值分布公平比对。
"""

import numpy as np


class StandardScaler:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * self.std + self.mean

    def __repr__(self):
        return f"StandardScaler(mean={self.mean}, std={self.std})"
