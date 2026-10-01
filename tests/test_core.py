import os

import numpy as np
import pytest

from diffusionforge.core.config import config_from_env, dataclass_replace
from diffusionforge.core.errors import ConfigError, DataError, DiffusionError
from diffusionforge.core.types import (
    ScheduleConfig,
    ModelConfig,
    TrainingConfig,
    SamplingConfig,
)
from diffusionforge.data.toys import sample_toy, list_toys


def test_schedule_validate_rejects_bad_kind():
    with pytest.raises(ValueError):
        ScheduleConfig(kind="bad").validate()


def test_config_from_env_override(monkeypatch):
    monkeypatch.setenv("DIFFUSION_EPOCHS", "7")
    monkeypatch.setenv("DIFFUSION_HIDDEN", "11")
    cfg = config_from_env(TrainingConfig())
    assert cfg.epochs == 7


def test_config_from_env_unknown_field_ignored(monkeypatch):
    monkeypatch.setenv("DIFFUSION_NOPE", "1")
    cfg = config_from_env(TrainingConfig())
    assert cfg.epochs == TrainingConfig().epochs


def test_dataclass_replace():
    cfg = dataclass_replace(ModelConfig(), hidden=5, depth=2)
    assert cfg.hidden == 5 and cfg.depth == 2


def test_error_hierarchy():
    assert issubclass(ConfigError, DiffusionError)
    assert issubclass(DataError, DiffusionError)


def test_list_toys_and_sample():
    names = list_toys()
    assert "gaussians25" in names and "moons" in names
    toy = sample_toy("gaussians25", n=200, seed=0)
    assert toy["samples"].shape == (200, 2)
    assert toy["modes"].shape[0] == 25
    assert toy["has_modes"] is True


def test_sample_unknown_raises():
    with pytest.raises(DataError):
        sample_toy("does_not_exist", n=10)
