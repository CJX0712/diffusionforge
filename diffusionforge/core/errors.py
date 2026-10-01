"""异常层级（E100~E400）。

代码约定：所有 DiffusionForge 抛出的业务错误都继承自 DiffusionError，便于调用方
按层级捕获；单一职责，不混入无关运行时异常。
"""


class DiffusionError(Exception):
    """DiffusionForge 所有业务异常的基类。"""


class ConfigError(DiffusionError):
    """E100 — 配置非法（字段越界 / 类型错误 / 未知枚举值）。"""


class DataError(DiffusionError):
    """E200 — 数据生成或载入失败（未知分布 / 样本形状错误）。"""


class BackendError(DiffusionError):
    """E300 — 后端不可用或探测失败（torch/diffusers 缺失且被强制要求）。"""


class NumericalError(DiffusionError):
    """E400 — 数值不变量被破坏（调度非单调 / 方差为负 / 样本含 NaN）。"""
