"""端到端管线（DiffusionPipeline）。

串联：数据 → 调度 → 模型 → 训练 → 采样 → 评测，单向无环。``benchmark`` 在
(数据集 × 调度 × 模型 × 采样器) 网格上做公平评测并产出可复现的 BenchmarkRow 列表
（落盘 benchmark.json）。
"""

import time

import numpy as np

from ..core.config import dataclass_replace
from ..core.types import (
    ScheduleConfig,
    ModelConfig,
    TrainingConfig,
    SamplingConfig,
    BenchmarkRow,
)
from ..data.toys import sample_toy
from ..eval.metrics import evaluate_samples, mmd
from ..models import build_model
from ..sampling.samplers import build_sampler
from ..schedules.schedule import build_scheduler, check_schedule_invariants
from ..training.trainer import train_model


class DiffusionPipeline:
    def __init__(
        self,
        schedule_cfg: ScheduleConfig = None,
        model_cfg: ModelConfig = None,
        train_cfg: TrainingConfig = None,
        sampling_cfg: SamplingConfig = None,
        data_dim: int = 2,
    ):
        self.schedule_cfg = schedule_cfg or ScheduleConfig()
        self.model_cfg = model_cfg or ModelConfig()
        self.train_cfg = train_cfg or TrainingConfig()
        self.sampling_cfg = sampling_cfg or SamplingConfig()
        for c in (self.schedule_cfg, self.model_cfg, self.train_cfg, self.sampling_cfg):
            if hasattr(c, "validate"):
                c.validate()
        self.data_dim = data_dim

    # ---- 单步 ----
    def train_once(self, dataset_name, schedule_kind, model_cfg=None, seed=None):
        model_cfg = model_cfg or self.model_cfg
        seed = self.train_cfg.seed if seed is None else seed
        toy = sample_toy(dataset_name, n=self.train_cfg.n_train, seed=seed)
        sched = build_scheduler(schedule_kind, self.schedule_cfg.timesteps, self.data_dim)
        check_schedule_invariants(sched)
        model = build_model(model_cfg, self.data_dim, sched.T, seed)
        # 数据标准化：消除不同分布量级差异，避免训练发散
        from ..data.scaler import StandardScaler
        scaler = StandardScaler().fit(toy["samples"])
        model.scaler = scaler
        x0_std = scaler.transform(toy["samples"])
        train_t = train_model(model, sched, x0_std, self.train_cfg, verbose=False)
        return model, sched, train_t, toy

    def evaluate(self, model, sched, toy, sampler_kind, steps, eta=0.0,
                 eval_seed=7, n_samples=2000):
        sampler = build_sampler(sampler_kind, steps=steps, eta=eta)
        t0 = time.time()
        samples = sampler.sample(model, sched, n_samples, eval_seed)
        sample_t = time.time() - t0
        if getattr(model, "scaler", None) is not None:
            samples = model.scaler.inverse_transform(samples)
        if not np.all(np.isfinite(samples)):
            raise ValueError("sampler produced non-finite samples")
        ev = evaluate_samples(samples, toy)
        return ev, sample_t, samples

    def run(self, dataset, schedule="cosine", sampler="ddpm", steps=None,
            eta=0.0, seed_train=None, eval_seed=7, n_samples=2000):
        if steps is None:
            steps = self.schedule_cfg.timesteps if sampler == "ddpm" else 100
        model, sched, train_t, toy = self.train_once(dataset, schedule, self.model_cfg, seed_train)
        ev, sample_t, samples = self.evaluate(
            model, sched, toy, sampler, steps, eta, eval_seed, n_samples
        )
        row = BenchmarkRow(
            dataset=dataset, schedule=schedule, model=self.model_cfg.kind,
            sampler=sampler, steps=steps, mmd=ev.mmd, wasserstein2=ev.wasserstein2,
            mode_coverage=ev.mode_coverage, energy_distance=ev.energy_distance,
            train_time_s=train_t, sample_time_s=sample_t,
        )
        return row, ev, samples

    # ---- 网格基准 ----
    def benchmark(self, datasets, schedules=None, model_kinds=None, samplers=None,
                  steps_map=None, n_samples=2000, eval_seed=7, train_seed=42):
        schedules = schedules or [self.schedule_cfg.kind]
        model_kinds = model_kinds or [self.model_cfg.kind]
        samplers = samplers or [self.sampling_cfg.method]
        steps_map = steps_map or {
            "ddpm": self.schedule_cfg.timesteps,
            "ddim": 100,
        }
        rows = []
        for ds in datasets:
            for sk in schedules:
                for mk in model_kinds:
                    mc = dataclass_replace(self.model_cfg, kind=mk)
                    model, sched_obj, train_t, toy = self.train_once(ds, sk, mc, train_seed)
                    for samp in samplers:
                        steps = steps_map.get(samp, sched_obj.T)
                        ev, sample_t, _ = self.evaluate(
                            model, sched_obj, toy, samp, steps,
                            self.sampling_cfg.eta, eval_seed, n_samples,
                        )
                        rows.append(BenchmarkRow(
                            dataset=ds, schedule=sk, model=mk, sampler=samp,
                            steps=steps, mmd=ev.mmd, wasserstein2=ev.wasserstein2,
                            mode_coverage=ev.mode_coverage,
                            energy_distance=ev.energy_distance,
                            train_time_s=train_t, sample_time_s=sample_t,
                            notes="numpy offline engine" if mk == "numpy_mlp" else "sota backend",
                        ))
        return rows
