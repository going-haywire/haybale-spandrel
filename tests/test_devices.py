"""Device, precision and memory budget resolution."""

from __future__ import annotations

import pytest
import torch

from haybale_spandrel.runtime import devices
from haybale_spandrel.runtime.devices import memory_budget_bytes, resolve_device, resolve_dtype

CPU = torch.device("cpu")
MPS = torch.device("mps")
CUDA = torch.device("cuda")


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.mark.usefixtures("no_gpu")
def test_auto_without_a_gpu_is_the_cpu() -> None:
    assert resolve_device("auto") == (CPU, None)
    assert resolve_device("") == (CPU, None)


def test_auto_prefers_cuda_then_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == (MPS, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == (CUDA, None)


@pytest.mark.usefixtures("no_gpu")
def test_unavailable_device_falls_back_to_cpu_with_a_note() -> None:
    device, note = resolve_device("cuda:1")
    assert device == CPU
    assert note == "cuda:1 not available — running on cpu"


def test_unknown_device_name_falls_back_to_cpu_with_a_note() -> None:
    device, note = resolve_device("banana")
    assert device == CPU
    assert note is not None
    assert "banana" in note


def test_cpu_is_always_available() -> None:
    assert resolve_device("cpu") == (CPU, None)


@pytest.mark.parametrize(
    ("precision", "device", "half", "bf16", "expected"),
    [
        ("auto", CPU, True, True, torch.float32),
        ("auto", MPS, True, True, torch.float16),
        ("auto", MPS, False, True, torch.float32),
        ("auto", CUDA, True, True, torch.float16),
        ("fp16", CPU, True, False, torch.float16),
        ("fp32", MPS, True, True, torch.float32),
        ("bf16", CPU, False, True, torch.bfloat16),
    ],
)
def test_precision_without_fallback(
    precision: str, device: torch.device, half: bool, bf16: bool, expected: torch.dtype
) -> None:
    assert resolve_dtype(precision, device, supports_half=half, supports_bfloat16=bf16) == (expected, None)


def test_auto_on_cuda_uses_bf16_when_only_bf16_is_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_dtype("auto", CUDA, supports_half=False, supports_bfloat16=True) == (torch.bfloat16, None)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_dtype("auto", CUDA, supports_half=False, supports_bfloat16=True) == (torch.float32, None)


def test_unsupported_explicit_precision_runs_fp32_with_a_note() -> None:
    dtype, note = resolve_dtype("bf16", MPS, supports_half=True, supports_bfloat16=False)
    assert dtype == torch.float32
    assert note == "bf16 not supported by this model — running fp32"


def test_memory_budget_from_the_setting() -> None:
    assert memory_budget_bytes(CPU, 2.0) == 2 * 1024**3


def test_memory_budget_derived_on_cpu() -> None:
    assert memory_budget_bytes(CPU, 0.0) == devices.CPU_BUDGET_BYTES == 4 * 1024**3


def test_memory_budget_derived_on_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 10_000)
    assert memory_budget_bytes(MPS, 0.0) == 5_000
