"""DiffusionForge — 世界顶级扩散模型（Diffusion Models）系统.

纯 numpy 手写 DDPM/DDIM 引擎（零下载离线兜底） + 可选 SOTA 后端 diffusers/torch
（可用性探测降级）。评测使用 MMD / Wasserstein-2 / 模式覆盖 / 能量距离，旗舰
DiffuseFuse 跨调度×采样器按验证 MMD 自适应选优。作者：晨星。
"""

__version__ = "1.0.0"
__author__ = "晨星"

from .core.types import (
    ScheduleConfig,
    ModelConfig,
    TrainingConfig,
    SamplingConfig,
    EvalResult,
    BenchmarkRow,
)
from .schedules.schedule import build_scheduler
from .models.numpy_mlp import NumpyMLP, available_numpy
from .models.torch_mlp import available_torch, available_diffusers

__all__ = [
    "ScheduleConfig",
    "ModelConfig",
    "TrainingConfig",
    "SamplingConfig",
    "EvalResult",
    "BenchmarkRow",
    "build_scheduler",
    "NumpyMLP",
    "available_numpy",
    "available_torch",
    "available_diffusers",
]
