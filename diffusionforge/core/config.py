"""配置解析与环境变量覆盖。

支持以 ``DIFFUSION_<FIELD>`` 形式覆盖任意配置字段，便于无代码调参（容器 /
CI 场景）。覆盖仅作用于基础配置（Schedule/Model/Training/Sampling），且对类型做
宽松转换并复用各自的 ``validate()``，非法值原样抛出 ConfigError。
"""

import os
from dataclasses import dataclass, fields, is_dataclass
from typing import Any


def _coerce(value: str, target_type: type) -> Any:
    if target_type is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def config_from_env(base: Any, prefix: str = "DIFFUSION_") -> Any:
    """用 ``<prefix><FIELD>`` 环境变量覆盖 dataclass 配置 ``base`` 的字段。

    返回新的 dataclass 实例（不修改入参）。未知字段 / 子类字段会被忽略。
    """
    if not is_dataclass(base):
        return base
    overrides = {}
    field_names = {f.name: f.type for f in fields(base)}
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        field_name = env_key[len(prefix):].lower()
        if field_name not in field_names:
            continue
        target_type = field_names[field_name]
        try:
            overrides[field_name] = _coerce(env_val, target_type)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"cannot coerce env {env_key}={env_val!r} to {target_type}"
            ) from exc
    if not overrides:
        return base
    new = dataclass_replace(base, **overrides)
    if hasattr(new, "validate"):
        new.validate()
    return new


def dataclass_replace(obj: Any, **changes) -> Any:
    """轻量 replace，避免循环依赖 dataclasses.replace 的字段完备性约束。"""
    result = type(obj)(**{f.name: getattr(obj, f.name) for f in fields(obj)})
    for k, v in changes.items():
        setattr(result, k, v)
    return result
