"""配置与结果数据类型（dataclass，契约先行）。

所有跨模块通信都通过这些数据结构，避免散落的 dict 与隐式字段。语义统一：
- 分数类指标里 MMD / Wasserstein-2 / 能量距离：越小越好（越接近目标分布）。
- mode_coverage：越大越好（∈[0,1]，越接近 1 模式覆盖越全）。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScheduleConfig:
    """噪声调度配置。"""

    kind: str = "cosine"  # "linear" | "cosine"
    timesteps: int = 1000

    def validate(self) -> "ScheduleConfig":
        if self.kind not in ("linear", "cosine"):
            raise ValueError(f"unknown schedule kind: {self.kind}")
        if self.timesteps < 2:
            raise ValueError("timesteps must be >= 2")
        return self


@dataclass
class ModelConfig:
    """ε-预测模型配置。"""

    kind: str = "numpy_mlp"  # "numpy_mlp" | "torch_mlp"
    hidden: int = 128
    depth: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5

    def validate(self) -> "ModelConfig":
        if self.kind not in ("numpy_mlp", "torch_mlp"):
            raise ValueError(f"unknown model kind: {self.kind}")
        if self.hidden < 1 or self.depth < 1:
            raise ValueError("hidden/depth must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        return self


@dataclass
class TrainingConfig:
    """训练配置。"""

    n_train: int = 6000
    epochs: int = 120
    batch_size: int = 256
    grad_clip: float = 5.0
    seed: int = 42

    def validate(self) -> "TrainingConfig":
        if self.n_train < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("n_train/epochs/batch_size must be >= 1")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be > 0")
        return self


@dataclass
class SamplingConfig:
    """采样配置。"""

    method: str = "ddpm"  # "ddpm" | "ddim"
    steps: int = 1000  # ddpm 用全 T；ddim 可远小于 T（如 100）
    eta: float = 0.0  # ddim 随机性：0=确定性，1=等价 ddpm

    def validate(self) -> "SamplingConfig":
        if self.method not in ("ddpm", "ddim"):
            raise ValueError(f"unknown sampler method: {self.method}")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.eta < 0 or self.eta > 1:
            raise ValueError("eta must be in [0,1]")
        return self


@dataclass
class EvalResult:
    """单次生成质量评测结果（越小越好者已注明）。"""

    mmd: float  # 最大均值差异，越小越好
    wasserstein2: float  # 2D 经验 Wasserstein-2，越小越好
    mode_coverage: float  # 模式覆盖比例 ∈[0,1]，越大越好
    energy_distance: float  # 能量距离，越小越好
    n_samples: int
    has_modes: bool = True
    kl_emp: Optional[float] = None  # 经验直方图 KL（可选，越小越好）


@dataclass
class BenchmarkRow:
    """基准表一行，跨 (数据集 × 调度 × 模型 × 采样器) 的公平评测。"""

    dataset: str
    schedule: str
    model: str
    sampler: str
    steps: int
    mmd: float
    wasserstein2: float
    mode_coverage: float
    energy_distance: float
    train_time_s: float
    sample_time_s: float
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "schedule": self.schedule,
            "model": self.model,
            "sampler": self.sampler,
            "steps": self.steps,
            "mmd": round(self.mmd, 6),
            "wasserstein2": round(self.wasserstein2, 6),
            "mode_coverage": round(self.mode_coverage, 4),
            "energy_distance": round(self.energy_distance, 6),
            "train_time_s": round(self.train_time_s, 2),
            "sample_time_s": round(self.sample_time_s, 2),
            "notes": self.notes,
        }
