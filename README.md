# DiffusionForge

> 世界顶级**扩散模型（Diffusion Models）**系统 · 作者：**晨星**（GitHub: [CJX0712](https://github.com/CJX0712)）
> 纯 numpy 离线兜底引擎 + 可选 SOTA 后端（torch / diffusers）· 31 个 pytest 单测全绿 · 模块化 + 契约先行

DiffusionForge 是一套**端到端可运行**的扩散模型研究与采样系统：从 2D 玩具分布出发，覆盖
**前向加噪（闭式）→ ε-预测训练 → DDPM/DDIM 反向采样 → 多维分布评测 → 跨调度×采样器自适应选优**
的完整链路，并以 `benchmark.json` 形式交付可复现的性能基线。

- 离线兜底引擎**纯 numpy 手写**（MLP 前向/反向 + Adam），零下载即可训练与采样；
- 可选 `torch` / `diffusers` 后端经 `available_*()` 探测，不可用时自动降级（**不伪造任何数字**）；
- 全部不变量由单测交叉验证：调度单调性、加噪闭式一致性、反向梯度 vs 有限差分、训练损失单调不增。

---

## 1. 特性速览

| 能力 | 说明 |
|------|------|
| 噪声调度 | `LinearSchedule`（DDPM 原始 β 线性）/ `CosineSchedule`（Nichol & Dhariwal 2021） |
| 模型 | `NumpyMLP`（离线引擎）/ `TorchMLP`（可选 SOTA 后端，自动对拍） |
| 采样器 | `DDPMSampler`（1000 步）/ `DDIMSampler`（200 步，η=0 确定性，数值护栏） |
| 评测 | MMD（RBF 核）/ Wasserstein-2 / 模式覆盖 / 能量距离 |
| 旗舰 | `DiffuseFuse`：跨 (调度×采样器) 留出验证选优 + 非劣守护 |
| 玩具数据 | 25 高斯 / 瑞士卷 / 双月 / 双环 / 棋盘 / 正弦（共 6 种） |
| 数据标准化 | `StandardScaler` + 梯度裁剪，防止小尺度分布训练发散 |
| 测试 | 31 个 pytest 单测，全部通过 |

---

## 2. 一键复现

```bash
# 1) 创建隔离 venv（Python 3.13）
python -m venv envs/diffusionforge
envs/diffusionforge/Scripts/python.exe -m pip install -r requirements.txt

# 2) 运行端到端 demo（训练 + 多配置基准 + 旗舰选优 → benchmark.json）
envs/diffusionforge/Scripts/python.exe diffusionforge/examples/run_demo.py

# 3) 运行全部单测
envs/diffusionforge/Scripts/python.exe -m pytest -q -W ignore::UserWarning
```

可选 SOTA 后端（非必须；不装也完整可跑）：

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install diffusers==0.40.0
```

---

## 3. 命令行（CLI）

```bash
# 列出内置玩具分布
python -m diffusionforge.cli --list-toys

# 跑基准（指定数据集，落盘 JSON）
python -m diffusionforge.cli --run --datasets gaussians25 moons checkerboard --out benchmark.json

# 单数据集采样示例
python -m diffusionforge.cli --sample --dataset moons --schedule cosine --sampler ddpm --steps 1000 --n 2000 --out samples.npy
```

环境变量覆盖（`core/config.py` 的 `config_from_env`）：
`DIFFUSION_TIMESTEPS`、`DIFFUSION_HIDDEN`、`DIFFUSION_LR`、`DIFFUSION_EPOCHS`、
`DIFFUSION_BATCH`、`DIFFUSION_GRAD_CLIP` 等。

---

## 4. 作为库调用

```python
from diffusionforge.pipeline import DiffusionPipeline
from diffusionforge.core.config import ScheduleConfig, ModelConfig, TrainingConfig, SamplingConfig

pipe = DiffusionPipeline(
    schedule=ScheduleConfig(kind="cosine", timesteps=1000),
    model=ModelConfig(kind="numpy_mlp", hidden=[64, 64], lr=1e-3),
    training=TrainingConfig(epochs=4000, batch=256, grad_clip=5.0),
    sampling=SamplingConfig(kind="ddpm", steps=1000),
)
result = pipe.run(dataset="moons")          # 训练 + 采样
print(result.metrics)                        # MMD / W2 / 覆盖率 / 能量距离
```

---

## 5. 架构

调用单向无环：`cli → pipeline → {data, schedules, models, training, sampling, eval} → core`。

```
diffusionforge/
  core/        types(dataclass) · errors(E100~E400) · config(ENV_XXX_* 覆盖) · interfaces(Protocol)
  data/        toys.py（6 种 2D 玩具分布）· scaler.py（StandardScaler）
  schedules/   schedule.py — LinearSchedule / CosineSchedule（β 调度闭式 + 不变量校验）
  models/      numpy_mlp.py（离线引擎）· torch_mlp.py（可选 SOTA 后端）
  training/    trainer.py — DDPM ε-预测训练（numpy + torch 双路径，梯度裁剪）
  sampling/    samplers.py — DDPMSampler / DDIMSampler（数值护栏）
  eval/        metrics.py — MMD / Wasserstein-2 / 模式覆盖 / 能量距离
  pipeline/    pipeline.py — DiffusionPipeline.run() + benchmark()
  fuse/        diffuse_fuse.py — 旗舰 DiffuseFuse（跨调度×采样器自适应选优）
  cli.py       argparse 入口
  examples/run_demo.py  端到端演示（落盘 benchmark.json）
tests/         31 个 pytest 单测（全绿）
docs/architecture.md   架构与算法细节
```

详见 [`docs/architecture.md`](docs/architecture.md)。

---

## 6. 算法核心

- **前向加噪（闭式）**：`x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε`
- **训练目标（ε-预测）**：`L = E[ ‖ε − ε_θ(x_t, t)‖² ]`，MLP 输入拼接 `[x, 16 维正弦时间嵌入]`
- **DDPM 反向**：`x_{t-1} = 1/√α_t·(x_t − (1−α_t)/√(1−ᾱ_t)·ε_θ) + √β̃_t·z`（注意 `α_t` 为单步）
- **DDIM**：`x_0 预测 = (x_t − √(1−ᾱ_t)·ε_θ)/√ᾱ_t`；η=0 确定性，步数可远小于 T
- **旗舰 DiffuseFuse**：在留出真值样本上以验证 MMD 选最优，仅当严格更优（超 margin）才切换，否则回退默认——非劣守护

---

## 7. 性能基线（可复现）

随机种子固定（seed=42），全部数值由 `examples/run_demo.py` 生成并落盘 `benchmark.json`。
CPU（无 GPU）下 numpy 引擎完成训练与采样。下表为**旗舰 DiffuseFuse 选出的最优组合**（详见 `benchmark.json`）：

| 数据集 | 选优调度 | 选优采样器 | MMD↓ | Wasserstein-2↓ | 模式覆盖↑ |
|--------|----------|------------|------|----------------|-----------|
| gaussians25 | cosine | ddpm (1000) | 0.0007 | 0.929 | 1.0 |
| moons | linear | ddpm (1000) | 0.0028 | 0.225 | 1.0 |
| checkerboard | linear | ddpm (1000) | 0.0010 | 0.156 | 1.0 |

- DDPM 在三个数据集上均达到 **mmd≈0.001~0.003、模式覆盖 1.0**；DDIM（200 步）稳定但 MMD 略高于 DDPM。
- 可选 SOTA 后端 torch 对拍（gaussians25）：mmd=0.0272、cov=0.92——纯 numpy 引擎质量对标 torch 后端。
- 实际完整 12 行基准 + 旗舰选优 + torch 对拍，见仓库根 `benchmark.json`。

---

## 8. 质量保障（DoD）

- [x] 隔离 venv + 锁定依赖（`requirements.lock.txt`）
- [x] 31 个 pytest 单测全绿（调度/模型/采样/评测/pipeline 不变量）
- [x] 端到端 demo 落盘 `benchmark.json`，数值合理（MMD 小、覆盖率高）
- [x] 可选 torch/diffusers 后端与 numpy 引擎对拍（不可用时自动降级）
- [x] 架构文档 + 一键复现命令 + LICENSE

---

## 9. License

MIT · 作者：晨星
