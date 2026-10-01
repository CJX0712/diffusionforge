"""模型子包：工厂 + 可用性探测。"""

from ..core.errors import BackendError
from .numpy_mlp import NumpyMLP, available_numpy
from .torch_mlp import TorchMLP, available_torch, available_diffusers


def build_model(cfg, data_dim: int, T: int, seed: int):
    """按 ModelConfig 构造 ε-模型，并绑定调度步数 T。"""
    if cfg.kind == "numpy_mlp":
        model = NumpyMLP(data_dim=data_dim, hidden=cfg.hidden, depth=cfg.depth, seed=seed,
                         lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.kind == "torch_mlp":
        if not available_torch():
            raise BackendError("torch_mlp requested but torch unavailable")
        model = TorchMLP(data_dim=data_dim, hidden=cfg.hidden, depth=cfg.depth,
                         lr=cfg.lr, seed=seed)
    else:
        raise ValueError(f"unknown model kind: {cfg.kind}")
    model.T = T
    return model


__all__ = [
    "NumpyMLP",
    "TorchMLP",
    "available_numpy",
    "available_torch",
    "available_diffusers",
    "build_model",
]
