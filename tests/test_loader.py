"""Loading model files onto a device, and the model types."""

from __future__ import annotations

import importlib
import subprocess
import sys
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


def test_extra_arch_loads_when_on(srformer_path: Path) -> None:
    model = load_model(srformer_path, CPU, "auto", noncommercial=True)
    assert model.architecture == "SRFormer"
    assert model.licence == "CC BY-NC 4.0"
    assert model.noncommercial
    assert model.status().endswith(" · non-commercial")
    assert any("CC BY-NC 4.0" in n and "non-commercial" in n for n in model.notes)


def test_unsupported_while_off_names_the_setting(srformer_path: Path) -> None:
    with pytest.raises(spandrel.UnsupportedModelError, match="Non-commercial architectures"):
        load_model(srformer_path, CPU, "auto")


def test_unsupported_says_what_failed(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    path = tmp_path / "junk.safetensors"
    save_file({"not.a.model": torch.zeros(3)}, str(path))
    with pytest.raises(spandrel.UnsupportedModelError, match=r"^junk\.safetensors holds no architecture"):
        load_model(path, CPU, "auto")
    with pytest.raises(spandrel.UnsupportedModelError, match=r"^junk\.safetensors holds no architecture"):
        load_model(path, CPU, "auto", noncommercial=True)


def test_main_arch_is_not_marked(tiny_sr_path: Path) -> None:
    model = load_model(tiny_sr_path, CPU, "auto", noncommercial=True)
    assert model.licence is None
    assert not model.noncommercial
    assert model.status() == "RealESRGAN Compact · 2x · RGB · cpu fp32"


def test_off_never_imports_extras(tiny_sr_path: Path) -> None:
    code = (
        "import sys; from pathlib import Path; import torch;"
        "from haybale_spandrel.runtime.loader import load_model;"
        f"load_model(Path({str(tiny_sr_path)!r}), torch.device('cpu'), 'auto');"
        "assert not any(m.startswith('spandrel_extra_arches') for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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
