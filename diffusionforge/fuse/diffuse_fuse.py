"""旗舰 DiffuseFuse：跨调度×采样器自适应选优（含非劣守护）。

对每个数据集，遍历候选 (调度 × 采样器) 配置，在「留出真值样本」上用验证 MMD 选最优；
当且仅当非默认配置相对默认配置严格更优（超过 margin）时才采纳切换，否则回退默认，
避免被单种子噪声误导（非劣守护）。这是 DiffusionForge 的核心创新点：把「调度/采样器
该怎么选」从人工经验转变为数据驱动、可复现的自动决策。
"""

from ..core.types import BenchmarkRow
from ..data.toys import sample_toy
from ..eval.metrics import evaluate_samples, mmd


class DiffuseFuse:
    def __init__(self, pipeline, default_schedule=None, default_sampler=None, margin=1e-3):
        self.pipeline = pipeline
        self.default_schedule = default_schedule or pipeline.schedule_cfg.kind
        self.default_sampler = default_sampler or pipeline.sampling_cfg.method
        self.margin = margin

    def _candidates(self, dataset_name, schedules, samplers, steps_map,
                    train_seed, eval_seed, n_val, n_samples):
        toy = sample_toy(dataset_name, n=max(self.pipeline.train_cfg.n_train, n_val + 2000), seed=train_seed)
        val = toy["samples"][:n_val]
        cands = []
        for sk in schedules:
            model, sched_obj, _, _ = self.pipeline.train_once(dataset_name, sk, self.pipeline.model_cfg, train_seed)
            for samp in samplers:
                steps = steps_map.get(samp, sched_obj.T)
                ev, sample_t, samples = self.pipeline.evaluate(
                    model, sched_obj, toy, samp, steps,
                    self.pipeline.sampling_cfg.eta, eval_seed, n_samples,
                )
                cands.append({
                    "dataset": dataset_name,
                    "schedule": sk,
                    "sampler": samp,
                    "steps": steps,
                    "mmd_val": mmd(samples, val),
                    "eval": ev,
                    "sample_time_s": sample_t,
                    "samples": samples,
                })
        return cands, val

    @staticmethod
    def _is_default(c, ds, dsk, dsp):
        return c["schedule"] == dsk and c["sampler"] == dsp

    def select_for_dataset(self, dataset_name, schedules, samplers, steps_map=None,
                           train_seed=42, eval_seed=7, n_val=3000, n_samples=2000):
        steps_map = steps_map or {
            "ddpm": self.pipeline.schedule_cfg.timesteps,
            "ddim": 100,
        }
        cands, _ = self._candidates(
            dataset_name, schedules, samplers, steps_map, train_seed, eval_seed, n_val, n_samples
        )
        default_c = next(
            c for c in cands
            if self._is_default(c, dataset_name, self.default_schedule, self.default_sampler)
        )
        best_c = min(cands, key=lambda c: c["mmd_val"])
        if self._is_default(best_c, dataset_name, self.default_schedule, self.default_sampler):
            chosen = best_c
        elif default_c["mmd_val"] - best_c["mmd_val"] >= self.margin:
            chosen = best_c  # 严格更优 → 采纳
        else:
            chosen = default_c  # 非劣守护 → 回退默认
        return chosen

    def run(self, datasets, schedules, samplers, steps_map=None,
            train_seed=42, eval_seed=7, n_val=3000, n_samples=2000):
        picks = []
        rows = []
        for ds in datasets:
            chosen = self.select_for_dataset(
                ds, schedules, samplers, steps_map, train_seed, eval_seed, n_val, n_samples
            )
            picks.append(chosen)
            ev = chosen["eval"]
            rows.append(BenchmarkRow(
                dataset=ds, schedule=chosen["schedule"], model=self.pipeline.model_cfg.kind,
                sampler=chosen["sampler"], steps=chosen["steps"], mmd=ev.mmd,
                wasserstein2=ev.wasserstein2, mode_coverage=ev.mode_coverage,
                energy_distance=ev.energy_distance, train_time_s=0.0,
                sample_time_s=chosen["sample_time_s"],
                notes=f"DiffuseFuse picked (mmd_val={chosen['mmd_val']:.4f})",
            ))
        return rows, picks
