"""Turn the Device, Precision and Memory budget settings into a torch device, dtype and byte count."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

#: Memory budget on the CPU when the setting is 0.
CPU_BUDGET_BYTES = 4 * 1024**3
_CUDA_FREE_SHARE = 0.8
_MPS_RECOMMENDED_SHARE = 0.5

_DTYPE_NAMES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def resolve_device(name: str) -> tuple[torch.device, str | None]:
    """Return the torch device for a Device setting, and a note when it falls back to the CPU.

    Args:
        name: ``auto`` (CUDA, then Apple GPU, then CPU), ``cpu``, ``mps``, ``cuda`` or
            ``cuda:N``; blank reads as ``auto``.

    Returns:
        The device, and ``None`` or a sentence for the card when the asked device is
        not available or not a device name.
    """
    import torch

    wanted = (name or "auto").strip().lower()
    if wanted == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), None
        if torch.backends.mps.is_available():
            return torch.device("mps"), None
        return torch.device("cpu"), None
    try:
        device = torch.device(wanted)
    except (RuntimeError, ValueError):
        return torch.device("cpu"), f"{name} is not a device — running on cpu"
    if _available(device):
        return device, None
    return torch.device("cpu"), f"{name} not available — running on cpu"


def _available(device: torch.device) -> bool:
    import torch

    if device.type == "cpu":
        return True
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return False
        return device.index is None or device.index < torch.cuda.device_count()
    if device.type == "mps":
        return bool(torch.backends.mps.is_available())
    return False


def resolve_dtype(
    precision: str, device: torch.device, *, supports_half: bool, supports_bfloat16: bool
) -> tuple[torch.dtype, str | None]:
    """Return the dtype a model runs in, and a note when an explicit choice falls back to fp32.

    ``auto`` picks fp16 on CUDA and Apple GPUs when the model supports it, bf16 on a
    CUDA card with bf16 when only that is supported, else fp32; the CPU always gets
    fp32. An explicit ``fp16`` or ``bf16`` the model does not support runs fp32.

    Args:
        precision: ``auto``, ``fp32``, ``fp16`` or ``bf16``; blank reads as ``auto``.
        device: The device the model runs on.
        supports_half: The model's ``supports_half``.
        supports_bfloat16: The model's ``supports_bfloat16``.
    """
    import torch

    wanted = (precision or "auto").strip().lower()
    if wanted == "auto":
        if device.type in ("cuda", "mps") and supports_half:
            return torch.float16, None
        if device.type == "cuda" and supports_bfloat16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16, None
        return torch.float32, None
    if wanted not in _DTYPE_NAMES:
        return torch.float32, f"{precision} is not a precision — running fp32"
    supported = {"fp32": True, "fp16": supports_half, "bf16": supports_bfloat16}[wanted]
    if not supported:
        return torch.float32, f"{wanted} not supported by this model — running fp32"
    return getattr(torch, _DTYPE_NAMES[wanted]), None


def dtype_name(dtype: torch.dtype) -> str:
    """Return ``fp32``, ``fp16`` or ``bf16`` for a torch dtype."""
    for short, long in _DTYPE_NAMES.items():
        if str(dtype) == f"torch.{long}":
            return short
    return str(dtype).removeprefix("torch.")


def memory_budget_bytes(device: torch.device, setting_gb: float) -> int:
    """Return the bytes one model run may use on *device*.

    Args:
        device: The device the model runs on.
        setting_gb: The Memory budget setting in GB; ``0`` derives the budget: 80 % of
            the free CUDA memory, half the Apple GPU's recommended working set, or
            4 GB on the CPU.
    """
    import torch

    if setting_gb > 0:
        return int(setting_gb * 1024**3)
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free * _CUDA_FREE_SHARE)
    if device.type == "mps":
        return int(torch.mps.recommended_max_memory() * _MPS_RECOMMENDED_SHARE)
    return CPU_BUDGET_BYTES
