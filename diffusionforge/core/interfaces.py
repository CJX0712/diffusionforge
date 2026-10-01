"""模块间契约（Protocol）。

调用单向无环：cli → pipeline → {data, schedules, models, training, sampling, eval}
→ core。任何跨模块对象都通过这些 Protocol 交互，保证可插拔与离线可测。
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NoiseScheduler(Protocol):
    """噪声调度：提供前向加噪闭式量与前向/反向所需统计量。"""

    T: int
    data_dim: int

    def betas(self) -> np.ndarray: ...
    def alphas_cumprod(self) -> np.ndarray: ...
    def sqrt_alphas_cumprod(self) -> np.ndarray: ...
    def sqrt_one_minus_alphas_cumprod(self) -> np.ndarray: ...
    def posterior_variance(self) -> np.ndarray: ...

    def add_noise(self, x0: np.ndarray, t: int, noise: np.ndarray) -> np.ndarray:
        """q(x_t | x_0)：返回单步加噪后的 x_t（batch 共享同一整数 t）。"""

    def extract(self, arr: np.ndarray, t: int, batch: int) -> np.ndarray:
        """把长度 T 的数组按整数 t 抽取 batch 份（用于采样循环）。"""


@runtime_checkable
class EpsModel(Protocol):
    """ε-预测模型：给定 (x_t, t) 预测噪声 ε。"""

    data_dim: int

    def predict_noise(self, x: np.ndarray, t: np.ndarray) -> np.ndarray: ...
    def parameters_count(self) -> int: ...


@runtime_checkable
class Sampler(Protocol):
    """采样器：由训练好的 ε-模型与调度还原 x_0 分布。"""

    name: str

    def sample(
        self, model: EpsModel, scheduler: NoiseScheduler, n: int, seed: int
    ) -> np.ndarray: ...
