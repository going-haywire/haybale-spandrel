"""Loading model files onto a device, and the model types."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import spandrel
import torch

from haybale_spandrel.runtime.loader import load_model, release_model

CPU = torch.device("cpu")


def test_safetensors_loads_on_cpu_in_fp32(tiny_sr_path: Path) -> None:
    model = load_model(tiny_sr_path, CPU, "auto")
    assert model.scale == 2
    assert model.dtype == torch.float32
    assert model.purpose == "SR"
    assert model.name == "tiny-2x"
    assert model.status() == "RealESRGAN Compact · 2x · RGB · cpu fp32"
    assert next(model.descriptor.model.parameters()).device == CPU


def test_pth_loads(tiny_sr4_path: Path) -> None:
    assert load_model(tiny_sr4_path, CPU, "fp32").scale == 4


def test_explicit_half_on_cpu_is_honoured(tiny_sr_path: Path) -> None:
    model = load_model(tiny_sr_path, CPU, "fp16")
    assert model.dtype == torch.float16
    assert next(model.descriptor.model.parameters()).dtype == torch.float16


def test_notes_are_kept_and_extended(tiny_sr_path: Path) -> None:
    model = load_model(tiny_sr_path, CPU, "nonsense", notes=("cuda:1 not available — running on cpu",))
    assert model.notes[0] == "cuda:1 not available — running on cpu"
    assert "nonsense" in model.notes[1]


def test_garbage_file_is_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "junk.safetensors"
    from safetensors.torch import save_file

    save_file({"not.a.model": torch.zeros(3)}, str(path))
    with pytest.raises(spandrel.UnsupportedModelError):
        load_model(path, CPU, "auto")


def test_release_runs(tiny_sr_path: Path) -> None:
    release_model(load_model(tiny_sr_path, CPU, "auto"))


def test_gray_model_reports_gray(tiny_gray_path: Path) -> None:
    model = load_model(tiny_gray_path, CPU, "auto")
    assert model.input_channels == 1
    assert "GRAY" in model.status()


@pytest.mark.integration
def test_model_type_never_serializes(library_system: Any, tiny_sr_path: Path) -> None:
    MODEL_BASE = importlib.import_module("haybale_spandrel.types.model").MODEL_BASE
    value = MODEL_BASE(load_model(tiny_sr_path, CPU, "auto"))
    assert value.to_dict() == {"value": {}}
    assert MODEL_BASE.from_dict({"value": {}}) is None
    assert MODEL_BASE().value is None
