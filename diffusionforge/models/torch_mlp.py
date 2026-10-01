"""可选 SOTA 后端：PyTorch MLP（diffusers/torch 生态的轻量代表）。

本文件演示「复用顶级开源」原则：当 torch 可用时，提供与纯 numpy 引擎同契约的
``TorchMLP``（正弦时间嵌入 + ReLU MLP + Adam），用于与离线引擎做对拍验证；
不可用时仅暴露 ``available_torch()/available_diffusers()`` 探测，不抛出导入错误。
"""

import numpy as np

from ..core.errors import BackendError

try:  # pragma: no cover - 依赖可选
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

try:  # pragma: no cover - 仅用于声明「复用 diffusers 生态」
    import diffusers  # noqa: F401

    _DIFFUSERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _DIFFUSERS_AVAILABLE = False


def available_torch() -> bool:
    return _TORCH_AVAILABLE


def available_diffusers() -> bool:
    return _DIFFUSERS_AVAILABLE


class _TorchMLPImpl:
    """真实 PyTorch 实现（仅在 torch 可用时构造）。"""

    def __init__(self, data_dim: int = 2, hidden: int = 128, depth: int = 4, lr: float = 1e-3, seed: int = 0):
        torch.manual_seed(seed)
        self.data_dim = int(data_dim)
        self.T = None
        inp = self.data_dim + 16
        sizes = [inp] + [hidden] * (depth - 1) + [data_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _emb(self, t):
        t = t.float() / self.T
        half = 8
        freqs = torch.exp(torch.linspace(np.log(1.0), np.log(16.0), half))
        ang = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self._emb(t)], dim=1))

    def predict_noise(self, x, t):
        self.net.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float32)
            tt = torch.tensor(np.asarray(t).reshape(-1), dtype=torch.long)
            out = self.forward(xt, tt)
        return out.cpu().numpy().astype(np.float64)

    def fit(self, samples, scheduler, cfg) -> None:
        self.T = scheduler.T
        T = self.T
        data = torch.tensor(samples, dtype=torch.float32)
        N = data.shape[0]
        acp = scheduler.alphas_cumprod()
        sqrt_acp = scheduler.sqrt_alphas_cumprod()
        sqrt_1macp = scheduler.sqrt_one_minus_alphas_cumprod()
        rng = np.random.default_rng(cfg.seed)
        n_batches = max(1, N // cfg.batch_size)
        self.net.train()
        for _ in range(cfg.epochs):
            perm = rng.permutation(N)
            for b in range(n_batches):
                idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                x0 = data[idx]
                t_int = torch.tensor(rng.integers(0, T, size=idx.size), dtype=torch.long)
                noise = torch.randn(idx.size, self.data_dim)
                sa = torch.tensor(sqrt_acp[t_int.numpy()], dtype=torch.float32).unsqueeze(1)
                s1 = torch.tensor(sqrt_1macp[t_int.numpy()], dtype=torch.float32).unsqueeze(1)
                x_t = sa * x0 + s1 * noise
                pred = self.forward(x_t, t_int)
                loss = ((pred - noise) ** 2).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


if _TORCH_AVAILABLE:

    class TorchMLP(_TorchMLPImpl):
        """与 NumpyMLP 同契约的 PyTorch 后端。"""
else:

    class TorchMLP:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise BackendError("torch is not installed; numpy engine is the offline fallback")
