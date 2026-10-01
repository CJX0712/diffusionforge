"""DiffusionForge 命令行入口（argparse）。

用法：
  python -m diffusionforge.cli run [--datasets NAME...] [--out benchmark.json]
  python -m diffusionforge.cli sample --dataset gaussians25 --n 2000 --out samples.npy
  python -m diffusionforge.cli list-toys
"""

import argparse
import json
import sys

from .core.types import (
    ScheduleConfig,
    ModelConfig,
    TrainingConfig,
    SamplingConfig,
    BenchmarkRow,
)
from .data.toys import list_toys, sample_toy
from .fuse import DiffuseFuse
from .models import available_torch, available_diffusers
from .pipeline import DiffusionPipeline


DEMO_DATASETS = ["gaussians25", "moons", "checkerboard"]


def _print_rows(title, rows):
    print(f"\n=== {title} ===")
    hdr = f"{'dataset':<14}{'sched':<9}{'model':<11}{'sampler':<8}{'steps':>6}{'mmd':>10}{'w2':>9}{'cov':>7}{'E-Dist':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.dataset:<14}{r.schedule:<9}{r.model:<11}{r.sampler:<8}"
            f"{r.steps:>6}{r.mmd:>10.4f}{r.wasserstein2:>9.4f}{r.mode_coverage:>7.3f}{r.energy_distance:>10.4f}"
        )


def run_benchmark(datasets=None, schedules=("cosine", "linear"),
                  samplers=("ddpm", "ddim"), out_path="benchmark.json",
                  use_torch=True):
    datasets = datasets or DEMO_DATASETS
    sched_cfg = ScheduleConfig(kind="cosine", timesteps=1000)
    model_cfg = ModelConfig(kind="numpy_mlp", hidden=96, depth=4)
    train_cfg = TrainingConfig(n_train=4000, epochs=100, batch_size=256, seed=42)
    samp_cfg = SamplingConfig(method="ddpm", steps=1000, eta=0.0)
    pipe = DiffusionPipeline(sched_cfg, model_cfg, train_cfg, samp_cfg, data_dim=2)

    steps_map = {"ddpm": 1000, "ddim": 200}
    print(">> numpy 离线引擎基准（数据集 × 调度 × 采样器）")
    rows = pipe.benchmark(
        datasets=datasets, schedules=list(schedules), samplers=list(samplers),
        steps_map=steps_map, n_samples=2000,
    )
    _print_rows("NumPy Benchmark", rows)

    print("\n>> 旗舰 DiffuseFuse（跨调度×采样器自适应选优 + 非劣守护）")
    fuse = DiffuseFuse(pipe, default_schedule="cosine", default_sampler="ddpm", margin=1e-3)
    fuse_rows, picks = fuse.run(
        datasets=datasets, schedules=list(schedules), samplers=list(samplers),
        steps_map=steps_map,
    )
    _print_rows("DiffuseFuse Picks", fuse_rows)

    report = {
        "system": "DiffusionForge",
        "version": "1.0.0",
        "author": "晨星",
        "backend": {
            "numpy": True,
            "torch": available_torch(),
            "diffusers": available_diffusers(),
        },
        "config": {
            "schedule": sched_cfg.__dict__,
            "model": model_cfg.__dict__,
            "training": train_cfg.__dict__,
            "sampling": samp_cfg.__dict__,
        },
        "numpy_benchmark": [r.as_dict() for r in rows],
        "diffuse_fuse": [r.as_dict() for r in fuse_rows],
    }

    if use_torch and available_torch():
        print("\n>> 可选 SOTA 后端 torch 对拍（gaussians25）")
        tc = ModelConfig(kind="torch_mlp", hidden=64, depth=3, lr=1e-3)
        tpipe = DiffusionPipeline(sched_cfg, tc, TrainingConfig(n_train=3000, epochs=40, batch_size=256, seed=42), samp_cfg)
        trow, tev, _ = tpipe.run("gaussians25", "cosine", "ddim", steps=100, n_samples=2000)
        print(f"   torch DDPM/DDIM → mmd={tev.mmd:.4f}  cov={tev.mode_coverage:.3f}")
        report["torch_crosscheck"] = {
            "dataset": "gaussians25", "mmd": tev.mmd,
            "mode_coverage": tev.mode_coverage, "wasserstein2": tev.wasserstein2,
        }
    else:
        report["torch_crosscheck"] = None

    # 汇总：numpy 基准平均 MMD / 覆盖
    if rows:
        avg_mmd = sum(r.mmd for r in rows) / len(rows)
        avg_cov = sum(r.mode_coverage for r in rows) / len(rows)
        report["summary"] = {
            "avg_mmd": round(avg_mmd, 4),
            "avg_mode_coverage": round(avg_cov, 4),
            "n_rows": len(rows),
        }

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 已落盘 {out_path}")
    return report


def cmd_run(args):
    run_benchmark(datasets=args.datasets, out_path=args.out, use_torch=not args.no_torch)


def cmd_sample(args):
    sched_cfg = ScheduleConfig(kind=args.schedule, timesteps=1000)
    model_cfg = ModelConfig(kind=args.model, hidden=96, depth=4)
    train_cfg = TrainingConfig(n_train=4000, epochs=max(20, args.epochs), batch_size=256, seed=42)
    samp_cfg = SamplingConfig(method=args.sampler, steps=args.steps, eta=0.0)
    pipe = DiffusionPipeline(sched_cfg, model_cfg, train_cfg, samp_cfg)
    model, sched, _, toy = pipe.train_once(args.dataset, args.schedule, model_cfg)
    _, _, samples = pipe.evaluate(model, sched, toy, args.sampler, args.steps, n_samples=args.n)
    npmod = __import__("numpy")
    npmod.save(args.out, samples)
    print(f"✅ 已生成 {args.n} 条样本 → {args.out}  (mmd vs true≈{_quick_mmd(samples, toy):.4f})")


def _quick_mmd(samples, toy):
    from .eval.metrics import mmd
    return mmd(samples, toy["samples"])


def cmd_list_toys(args):
    print("可用玩具分布:", ", ".join(list_toys()))


def build_parser():
    p = argparse.ArgumentParser(prog="diffusionforge", description="DiffusionForge CLI")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="运行端到端基准并落盘 benchmark.json")
    pr.add_argument("--datasets", nargs="+", default=None)
    pr.add_argument("--out", default="benchmark.json")
    pr.add_argument("--no-torch", action="store_true", help="跳过 torch 对拍")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("sample", help="训练并采样生成分布")
    ps.add_argument("--dataset", default="gaussians25")
    ps.add_argument("--schedule", default="cosine")
    ps.add_argument("--model", default="numpy_mlp")
    ps.add_argument("--sampler", default="ddpm")
    ps.add_argument("--steps", type=int, default=1000)
    ps.add_argument("--epochs", type=int, default=100)
    ps.add_argument("--n", type=int, default=2000)
    ps.add_argument("--out", default="samples.npy")
    ps.set_defaults(func=cmd_sample)

    pl = sub.add_parser("list-toys", help="列出可用玩具分布")
    pl.set_defaults(func=cmd_list_toys)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        # 默认跑基准
        run_benchmark()
        return 0
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
