"""纯 numpy ε-预测 MLP（离线兜底引擎，零第三方依赖）。

网络：输入 = [x (data_dim) | 正弦时间嵌入 (16)] → 若干隐藏层(ReLU) → 输出 ε (data_dim)。
手写前向/反向 + Adam，作为 DiffusionForge 的零下载默认引擎，保证无网络/无 GPU 也可跑。

可验证不变量（test_models 交叉验证）：
- 反向传播梯度与有限差分在单权重上的数值梯度一致（相对误差 ~1e-5）。
- 训练后 MSE 损失单调不增（逐 epoch 下降）。
- predict_noise 输出形状 == 输入 x 形状。
"""

import numpy as np

from ..core.errors import NumericalError

_EMB_DIM = 16


def _sinusoidal_embed(t_cont: np.ndarray, dim: int = _EMB_DIM) -> np.ndarray:
    """连续时间 → 正弦嵌入 (n, dim)。频率呈对数间隔，提升时间条件质量。"""
    half = dim // 2
    freqs = np.exp(np.linspace(np.log(1.0), np.log(float(dim)), half))
    ang = np.outer(t_cont.astype(np.float64), freqs)  # (n, half)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def available_numpy() -> bool:
    """numpy 引擎始终可用。"""
    return True


class NumpyMLP:
    """纯 numpy 多层感知机，预测扩散噪声 ε。"""

    def __init__(self, data_dim: int = 2, hidden: int = 128, depth: int = 4, seed: int = 0,
                 lr: float = 1e-3, weight_decay: float = 1e-5):
        self.data_dim = int(data_dim)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.input_dim = self.data_dim + _EMB_DIM
        sizes = [self.input_dim] + [hidden] * (depth - 1) + [data_dim]
        self.sizes = sizes
        self.T = None  # 必须在使用前由 pipeline/trainer 绑定调度步数

        rng = np.random.default_rng(seed)
        scale = 1e-2
        self.W = [rng.normal(0.0, scale, size=(sizes[i + 1], sizes[i])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros((sizes[i + 1],), dtype=np.float64) for i in range(len(sizes) - 1)]

        self.m = [np.zeros_like(w) for w in self.W]
        self.v = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self._adam_t = 0
        self._cache_acts = None
        self._cache_zs = None
        self.history = []
        self.scaler = None  # 由 pipeline 在标准化训练时注入

    # ---- 参数 ----
    def parameters_count(self) -> int:
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

    # ---- 前向 ----
    def forward(self, x: np.ndarray, t: np.ndarray, T: int) -> np.ndarray:
        if T is None:
            raise NumericalError("model.T is not bound to a scheduler")
        emb = _sinusoidal_embed(np.asarray(t, dtype=np.float64) / T)
        a0 = np.concatenate([np.asarray(x, dtype=np.float64), emb], axis=1)
        self._cache_acts = [a0]
        self._cache_zs = []
        a = a0
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l].T + self.b[l]
            self._cache_zs.append(z)
            a = np.maximum(z, 0.0) if l < L - 1 else z
            self._cache_acts.append(a)
        return a

    def predict_noise(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        t = np.asarray(t).reshape(-1)
        return self.forward(x, t, self.T)

    # ---- 损失 / 反向 ----
    def predict_loss(self, x: np.ndarray, t: np.ndarray, target: np.ndarray) -> float:
        out = self.forward(x, t, self.T)
        return float(np.mean((out - target) ** 2))

    def loss_and_grad(self, x: np.ndarray, t: np.ndarray, target: np.ndarray):
        out = self.forward(x, t, self.T)
        loss = float(np.mean((out - target) ** 2))
        gW, gb = self._backward(target)
        return loss, gW, gb

    def _backward(self, target: np.ndarray):
        L = len(self.W)
        n = target.shape[0]
        out = self._cache_acts[-1]
        dout = (out - np.asarray(target, dtype=np.float64)) / (n * self.data_dim)
        gW = [None] * L
        gb = [None] * L
        gW[L - 1] = dout.T @ self._cache_acts[L - 1]
        gb[L - 1] = dout.sum(0)
        da = dout @ self.W[L - 1]
        for l in range(L - 2, -1, -1):
            z = self._cache_zs[l]
            dz = da * (z > 0).astype(np.float64)
            gW[l] = dz.T @ self._cache_acts[l]
            gb[l] = dz.sum(0)
            da = dz @ self.W[l]
        return gW, gb

    # ---- Adam ----
    def adam_step(self, gW, gb, lr: float = 1e-3, wd: float = 1e-5,
                  b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self._adam_t += 1
        for i in range(len(self.W)):
            gWw = gW[i] + wd * self.W[i]
            self.m[i] = b1 * self.m[i] + (1 - b1) * gWw
            self.v[i] = b2 * self.v[i] + (1 - b2) * (gWw ** 2)
            mhat = self.m[i] / (1 - b1 ** self._adam_t)
            vhat = self.v[i] / (1 - b2 ** self._adam_t)
            self.W[i] -= lr * mhat / (np.sqrt(vhat) + eps)

            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * (gb[i] ** 2)
            mbhat = self.mb[i] / (1 - b1 ** self._adam_t)
            vbhat = self.vb[i] / (1 - b2 ** self._adam_t)
            self.b[i] -= lr * mbhat / (np.sqrt(vbhat) + eps)
