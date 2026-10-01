import os
import json

from diffusionforge.core.types import (
    ScheduleConfig,
    ModelConfig,
    TrainingConfig,
    SamplingConfig,
)
from diffusionforge.pipeline import DiffusionPipeline
from diffusionforge.fuse import DiffuseFuse


def _tiny_pipeline():
    sched_cfg = ScheduleConfig(kind="cosine", timesteps=50)
    model_cfg = ModelConfig(kind="numpy_mlp", hidden=16, depth=2)
    train_cfg = TrainingConfig(n_train=300, epochs=6, batch_size=64, seed=0)
    samp_cfg = SamplingConfig(method="ddpm", steps=50, eta=0.0)
    return DiffusionPipeline(sched_cfg, model_cfg, train_cfg, samp_cfg, data_dim=2)


def test_pipeline_run_returns_row_and_eval():
    pipe = _tiny_pipeline()
    row, ev, samples = pipe.run("moons", "cosine", "ddpm", steps=50, n_samples=200)
    assert row.mmd >= 0
    assert samples.shape == (200, 2)
    assert row.mode_coverage >= 0


def test_pipeline_benchmark_small_grid():
    pipe = _tiny_pipeline()
    rows = pipe.benchmark(
        datasets=["moons"], schedules=["cosine"], samplers=["ddpm"], n_samples=100,
    )
    assert len(rows) == 1
    assert rows[0].dataset == "moons"


def test_diffuse_fuse_picks():
    pipe = _tiny_pipeline()
    fuse = DiffuseFuse(pipe, default_schedule="cosine", default_sampler="ddpm", margin=1e-3)
    rows, picks = fuse.run(
        datasets=["moons"], schedules=["cosine", "linear"], samplers=["ddpm", "ddim"],
        steps_map={"ddpm": 50, "ddim": 20},
    )
    assert len(rows) == 1
    assert picks[0]["dataset"] == "moons"
    assert "schedule" in picks[0] and "sampler" in picks[0]
