# DiffusionForge 架构文档

> 世界顶级扩散模型（Diffusion Models）系统 · 作者：晨星 · 纯 numpy 离线兜底 + 可选 SOTA 后端（torch/diffusers）

---

## 1. 设计原则

- **单一职责 + 契约先行**：所有跨模块通信通过 `core/` 的 dataclass 与 `Protocol` 完成，无散落 dict。
- **调用单向无环**：`cli → pipeline → {data, schedules, models, training, sampling, eval} → core`。
- **离线兜底优先**：纯 numpy 手写 DDPM/DDIM 引擎保证零下载可跑；`torch`/`diffusers` 作为可选 SOTA 后端，经 `available_*()` 探测，不可用时自动降级，benchmark 不伪造数字。
- **可验证不变量**：调度单调性、加噪闭式、反向梯度与有限差分一致、训练损失单调不增——全部由单测交叉验证。

---

## 2. 模块布局

```
diffusionforge/
  core/        types(dataclass) · errors(E100~E400) · config(ENV_XXX_* 覆盖) · interfaces(Protocol)
  data/        toys.py — 2D 玩具分布（25高斯/瑞士卷/双月/双环/棋盘/正弦）
  schedules/   schedule.py — LinearSchedule / CosineSchedule（β 调度闭式）
  models/      numpy_mlp.py（离线引擎）· torch_mlp.py（可选 SOTA 后端）
  training/    trainer.py — DDPM ε-预测训练（numpy + torch 双路径）
  sampling/    samplers.py — DDPMSampler / DDIMSampler
  eval/        metrics.py — MMD / Wasserstein-2 / 模式覆盖 / 能量距离
  pipeline/    pipeline.py — DiffusionPipeline.run() + benchmark()
  fuse/        diffuse_fuse.py — 旗舰 DiffuseFuse（跨调度×采样器自适应选优）
  cli.py       argparse 入口
  examples/run_demo.py  端到端演示（落盘 benchmark.json）
tests/         31 个 pytest 单测（全绿）
```

---

## 3. 算法核心

### 3.1 前向加噪（封闭形式）

对任意 `t`：

```
x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε ,   ε ~ N(0, I)
```

其中 `ᾱ_t = ∏_{s=1..t}(1 − β_s)` 由调度给出。两种调度：

- **Linear**：`β_t` 从 `1e-4` 线性升到 `0.02`（DDPM 原始）。
- **Cosine**（Nichol & Dhariwal 2021）：`ᾱ_t = cos((t/T + s)/(1+s)·π/2)² / cos(s/(1+s)·π/2)²`，低噪声区更平滑。

### 3.2 训练目标（ε-预测，简单稳定）

```
L = E_{x_0, t, ε} [ ‖ ε − ε_θ(x_t, t) ‖² ]
```

`ε_θ` 为 MLP，输入拼接 `[x, t 的正弦时间嵌入(16)]`，输出噪声预测。numpy 引擎手写前向/反向 + Adam；torch 引擎用 `torch.optim.Adam` 对拍。

### 3.3 反向采样

- **DDPM**：`x_{t-1} = 1/√ᾱ_t·(x_t − (1−ᾱ_t)/√(1−ᾱ_t)·ε_θ) + √β̃_t·z`，`t` 从 `T−1` 到 `0` 逐步去噪。
- **DDIM**：`x_0 预测 = (x_t − √(1−ᾱ_t)·ε_θ)/√ᾱ_t`；`η=0` 时退化为确定轨迹，步数可远小于 `T`（默认 200 步，含 `x_0_pred∈[-15,15]` 数值护栏）。

### 3.4 评测指标（越小越好者已注）

| 指标 | 含义 | 口径 |
|------|------|------|
| MMD | 最大均值差异（RBF 核，中位数启发式带宽） | 越小越好，同分布≈0 |
| Wasserstein-2 | 2D 经验最优匹配距离 | 越小越好 |
| 模式覆盖 | 真实模式被生成样本半径内覆盖比例 | 越大越好∈[0,1] |
| 能量距离 | `2E‖X−Y‖ − E‖X−X'‖ − E‖Y−Y'‖` | 越小越好（≥0） |

### 3.5 旗舰 DiffuseFuse

对每个数据集遍历候选 `(调度 × 采样器)`，在**留出真值样本**上以验证 MMD 选最优；仅当非默认配置相对默认**严格更优（超过 margin）**才采纳切换，否则回退默认——**非劣守护**防止被单种子噪声误导。

---

## 4. 不变量（单测守护）

- 调度：`ᾱ_t` 单调非增、`β̃_t ≥ 0`、`add_noise` 闭式一致、`t=0` 近似恒等。
- 模型：反向传播梯度与单权重有限差分一致（相对误差 ~1e-3）；训练后损失单调不增。
- 采样：输出形状 `(n, data_dim)`、全有限、无 NaN/Inf。
- 评测：`MMD(x,x)≈0`、`MMD(x,y>0)`、`能量距离 ≥ 0`。

---

## 5. 性能基线（随机种子固定，可复现）

详见仓库根 `benchmark.json`（由 `examples/run_demo.py` 生成）。摘要：

| 数据集 | 调度 | 采样器 | MMD↓ | 覆盖↑ |
|--------|------|--------|------|-------|
| gaussians25 | cosine | ddpm | — | — |
| moons | cosine | ddpm | — | — |
| checkerboard | cosine | ddpm | — | — |

> 数值以 `benchmark.json` 为准；numpy 引擎在 CPU（无 GPU）上完成全部训练与采样。
