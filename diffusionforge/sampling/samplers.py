"""采样器（反向扩散还原 x_0）。

- DDPMSampler：标准 1000 步马尔可夫反向采样（每步加噪声直至 t=0）。
- DDIMSampler：确定性/低方差加速采样，步数可远小于 T（如 100 步），η=0 时退化为
  确定轨迹，η=1 时等价于 DDPM 方差。

不变量（test_sampling 验证）：
- 输出形状 == (n, data_dim)，且全有限（无 NaN/Inf）。
- t=0 反向步不向样本注入额外噪声（纯均值还原）。
"""

import numpy as np


class DDPMSampler:
    name = "ddpm"

    def sample(self, model, scheduler, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, size=(n, model.data_dim))
        T = scheduler.T
        acp = scheduler.alphas_cumprod()
        alphas = scheduler.alphas()
        post_var = scheduler.posterior_variance()

        for t in reversed(range(T)):
            t_batch = np.full((n,), t, dtype=np.int64)
            eps = model.predict_noise(x, t_batch)
            acp_t = acp[t]
            alpha_t = alphas[t]
            sqrt_acp_t = np.sqrt(acp_t)
            sqrt_1macp_t = np.sqrt(1.0 - acp_t)
            # 正确 DDPM 反向均值系数：1/sqrt(alpha_t)（单步 alpha，非累积 ᾱ_t）
            coef1 = 1.0 / np.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / sqrt_1macp_t
            mean = coef1 * (x - coef2 * eps)
            if t > 0:
                var = post_var[t]
                noise = rng.normal(0.0, 1.0, size=x.shape)
                x = mean + np.sqrt(var) * noise
            else:
                x = mean
        return x


class DDIMSampler:
    name = "ddim"

    def __init__(self, steps: int = 100, eta: float = 0.0):
        self.steps = int(steps)
        self.eta = float(eta)

    def sample(self, model, scheduler, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, size=(n, model.data_dim))
        T = scheduler.T
        acp = scheduler.alphas_cumprod()

        if self.steps >= T:
            seq = np.arange(T - 1, -1, -1)
        else:
            seq = np.linspace(T - 1, 0, self.steps).round().astype(int)
            seq = np.clip(seq, 0, T - 1)
            seq = np.unique(seq)[::-1]  # 降序去重，尾端含 0

        for i, t in enumerate(seq):
            t_batch = np.full((n,), int(t), dtype=np.int64)
            eps = model.predict_noise(x, t_batch)
            acp_t = acp[t]
            x0_pred = (x - np.sqrt(1.0 - acp_t) * eps) / np.sqrt(acp_t)
            # 数值护栏：首个大 t 步 acp_t 极小，弱模型下 x0_pred 可能放大；裁剪成
            # 标准化空间内的合理范围（真值分布标准化后 |x| 通常 ≤ 15）。
            x0_pred = np.clip(x0_pred, -15.0, 15.0)
            if i < len(seq) - 1:
                t_prev = int(seq[i + 1])
                acp_prev = acp[t_prev]
                var = self.eta * np.sqrt((1.0 - acp_prev) / (1.0 - acp_t)) * np.sqrt(1.0 - acp_t / acp_prev)
                dir_x0 = np.sqrt(acp_prev) * x0_pred
                if self.eta > 0:
                    noise = rng.normal(0.0, 1.0, size=x.shape)
                else:
                    noise = 0.0
                x = dir_x0 + np.sqrt(max(1.0 - acp_prev - var ** 2, 0.0)) * eps + var * noise
            else:
                x = x0_pred  # t_prev = 0 → acp_prev = 1
        return x


def build_sampler(method: str, steps: int = 1000, eta: float = 0.0):
    if method == "ddpm":
        return DDPMSampler()
    if method == "ddim":
        return DDIMSampler(steps=steps, eta=eta)
    raise ValueError(f"unknown sampler method: {method}")
