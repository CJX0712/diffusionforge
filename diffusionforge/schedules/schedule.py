"""噪声调度（Noise Schedule）。

两种 SOTA 调度：
- linear：β 从 beta_start 线性升到 beta_end（DDPM 原始设定）。
- cosine：Nichol & Dhariwal 2021 的余弦调度，低噪声区更平滑、样本质量更优。

不变量（由 test_schedule 交叉验证）：
- alphas_cumprod 单调非增，且 acp[-1] 接近 0（噪声充分）。
- t=0 时 sqrt_acp≈1、sqrt_1macp≈0 → add_noise 近似恒等。
- t=T-1 时 sqrt_acp≈0、sqrt_1macp≈1 → add_noise 近似纯噪声。
- posterior_variance 全程 ≥ 0。
"""

import math
import numpy as np

from ..core.errors import NumericalError

_BETA_START = 1e-4
_BETA_END_LINEAR = 0.02
_COSINE_S = 0.008
_COSINE_MAX = 0.999


class LinearSchedule:
    """线性 β 调度。"""

    def __init__(self, timesteps: int = 1000, data_dim: int = 2,
                 beta_start: float = _BETA_START, beta_end: float = _BETA_END_LINEAR):
        self.T = int(timesteps)
        self.data_dim = data_dim
        self._betas = np.linspace(beta_start, beta_end, self.T, dtype=np.float64)
        self._build()

    def _build(self) -> None:
        self._alphas = 1.0 - self._betas
        self._acp = np.cumprod(self._alphas)
        self._sqrt_acp = np.sqrt(self._acp)
        self._sqrt_1macp = np.sqrt(1.0 - self._acp)
        # beta_tilde_t = (1-acp_{t-1})/(1-acp_t) * beta_t
        acp_prev = np.concatenate([[1.0], self._acp[:-1]])
        self._post_var = (1.0 - acp_prev) / (1.0 - self._acp) * self._betas
        self._post_var = np.clip(self._post_var, 0.0, None)

    def betas(self) -> np.ndarray:
        return self._betas

    def alphas(self) -> np.ndarray:
        return self._alphas

    def alphas_cumprod(self) -> np.ndarray:
        return self._acp

    def sqrt_alphas_cumprod(self) -> np.ndarray:
        return self._sqrt_acp

    def sqrt_one_minus_alphas_cumprod(self) -> np.ndarray:
        return self._sqrt_1macp

    def posterior_variance(self) -> np.ndarray:
        return self._post_var

    def add_noise(self, x0: np.ndarray, t: int, noise: np.ndarray) -> np.ndarray:
        return self._sqrt_acp[t] * x0 + self._sqrt_1macp[t] * noise

    def extract(self, arr: np.ndarray, t: int, batch: int) -> np.ndarray:
        return np.full((batch, 1), arr[t], dtype=arr.dtype)


class CosineSchedule:
    """余弦 β 调度（Nichol & Dhariwal 2021）。"""

    def __init__(self, timesteps: int = 1000, data_dim: int = 2, s: float = _COSINE_S):
        self.T = int(timesteps)
        self.data_dim = data_dim
        self._betas = self._cosine_betas(self.T, s)
        self._build()

    @staticmethod
    def _cosine_betas(timesteps: int, s: float) -> np.ndarray:
        steps = np.arange(timesteps + 1, dtype=np.float64) / timesteps
        f = np.cos((steps + s) / (1.0 + s) * math.pi * 0.5) ** 2
        bar = f / f[0]
        beta = 1.0 - bar[1:] / bar[:-1]
        return np.clip(beta, 0.0, _COSINE_MAX)

    def _build(self) -> None:
        self._alphas = 1.0 - self._betas
        self._acp = np.cumprod(self._alphas)
        self._sqrt_acp = np.sqrt(self._acp)
        self._sqrt_1macp = np.sqrt(1.0 - self._acp)
        acp_prev = np.concatenate([[1.0], self._acp[:-1]])
        self._post_var = (1.0 - acp_prev) / (1.0 - self._acp) * self._betas
        self._post_var = np.clip(self._post_var, 0.0, None)

    def betas(self) -> np.ndarray:
        return self._betas

    def alphas(self) -> np.ndarray:
        return self._alphas

    def alphas_cumprod(self) -> np.ndarray:
        return self._acp

    def sqrt_alphas_cumprod(self) -> np.ndarray:
        return self._sqrt_acp

    def sqrt_one_minus_alphas_cumprod(self) -> np.ndarray:
        return self._sqrt_1macp

    def posterior_variance(self) -> np.ndarray:
        return self._post_var

    def add_noise(self, x0: np.ndarray, t: int, noise: np.ndarray) -> np.ndarray:
        return self._sqrt_acp[t] * x0 + self._sqrt_1macp[t] * noise

    def extract(self, arr: np.ndarray, t: int, batch: int) -> np.ndarray:
        return np.full((batch, 1), arr[t], dtype=arr.dtype)


_SCHEDULERS = {"linear": LinearSchedule, "cosine": CosineSchedule}


def build_scheduler(kind: str, timesteps: int = 1000, data_dim: int = 2):
    """按 kind 构造调度器（工厂）。"""
    if kind not in _SCHEDULERS:
        raise ValueError(f"unknown schedule kind: {kind}")
    return _SCHEDULERS[kind](timesteps=timesteps, data_dim=data_dim)


def check_schedule_invariants(sched) -> None:
    """对调度做数值不变量断言，破坏即抛 NumericalError。"""
    acp = sched.alphas_cumprod()
    if acp[0] > 1.0 + 1e-9 or acp[-1] < -1e-9:
        raise NumericalError(f"alphas_cumprod out of range: acp[0]={acp[0]:.3e} acp[-1]={acp[-1]:.3e}")
    if not np.all(np.diff(acp) <= 1e-9):
        raise NumericalError("alphas_cumprod not monotonically non-increasing")
    if sched.posterior_variance().min() < -1e-12:
        raise NumericalError("posterior_variance has negative entry")
    # 闭式加噪正确性：x_t == sqrt_acp[t]·x0 + sqrt_1macp[t]·ε（对任意 T 恒成立）
    x0 = np.ones((4, sched.data_dim))
    eps = np.zeros((4, sched.data_dim))
    x_t0 = sched.add_noise(x0, 0, eps)
    if not np.allclose(x_t0, sched.sqrt_alphas_cumprod()[0] * x0, atol=1e-9):
        raise NumericalError("add_noise closed-form mismatch at t=0")
    noise = np.full((4, sched.data_dim), 5.0)
    x_tl = sched.add_noise(np.zeros((4, sched.data_dim)), sched.T - 1, noise)
    if not np.allclose(x_tl, sched.sqrt_one_minus_alphas_cumprod()[sched.T - 1] * noise, atol=1e-9):
        raise NumericalError("add_noise closed-form mismatch at t=T-1")
    # t=0 近似恒等（beta_start 很小 → 偏差 ~1e-4）
    if not np.allclose(x_t0, x0, atol=1e-3):
        raise NumericalError("add_noise at t=0 deviates too far from identity")
