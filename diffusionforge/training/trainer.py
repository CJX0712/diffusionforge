"""DDPM 训练（ε-预测目标）。

目标：给定 x0∼数据、t∼Uniform(0..T)、ε∼N(0,I)，构造 x_t = sqrt(acp_t)·x0 +
sqrt(1−acp_t)·ε，令模型预测 ε，最小化 MSE(ε_pred, ε)。这是 DDPM 的标准简单目标，
训练稳定、与采样严格对齐。
"""

import time

import numpy as np

from ..core.types import TrainingConfig
from ..models.numpy_mlp import NumpyMLP


def _clip_grads(gW, gb, max_norm: float):
    sq = sum(float(np.sum(g ** 2)) for g in gW) + sum(float(np.sum(g ** 2)) for g in gb)
    norm = np.sqrt(sq)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        gW = [g * scale for g in gW]
        gb = [g * scale for g in gb]
    return gW, gb


def _train_numpy(model: NumpyMLP, scheduler, samples: np.ndarray, cfg: TrainingConfig, verbose: bool) -> float:
    rng = np.random.default_rng(cfg.seed)
    N = samples.shape[0]
    T = scheduler.T
    model.T = T
    acp = scheduler.alphas_cumprod()
    sqrt_acp = scheduler.sqrt_alphas_cumprod()
    sqrt_1macp = scheduler.sqrt_one_minus_alphas_cumprod()

    n_batches = max(1, N // cfg.batch_size)
    for epoch in range(cfg.epochs):
        perm = rng.permutation(N)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            x0 = samples[idx]
            t_int = rng.integers(0, T, size=idx.size)
            noise = rng.normal(0.0, 1.0, size=x0.shape)
            sa = sqrt_acp[t_int].reshape(-1, 1)
            s1 = sqrt_1macp[t_int].reshape(-1, 1)
            x_t = sa * x0 + s1 * noise
            loss, gW, gb = model.loss_and_grad(x_t, t_int, noise)
            # 梯度裁剪：防止小尺度分布训练发散
            gW, gb = _clip_grads(gW, gb, cfg.grad_clip)
            model.adam_step(gW, gb, lr=model.lr, wd=model.weight_decay)
            epoch_loss += loss
        epoch_loss /= n_batches
        model.history.append(epoch_loss)
        if verbose and (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            print(f"  [epoch {epoch+1:>4}/{cfg.epochs}] loss={epoch_loss:.4f}")
    return sum(model.history)  # placeholder; real time measured by train_model


def train_model(model, scheduler, samples: np.ndarray, cfg: TrainingConfig, verbose: bool = True) -> float:
    """训练模型；返回训练耗时（秒）。"""
    t0 = time.time()
    if isinstance(model, NumpyMLP):
        _train_numpy(model, scheduler, samples, cfg, verbose)
    else:
        # torch 后端自带 fit（含内部循环）
        model.fit(samples, scheduler, cfg)
    return time.time() - t0
