"""Load a model file with spandrel onto a device in a precision, and free it again."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from haybale_spandrel.runtime.devices import resolve_dtype
from haybale_spandrel.value.spandrel_model import SpandrelModel

if TYPE_CHECKING:
    import torch


def load_model(
    path: Path, device: torch.device, precision: str, *, notes: tuple[str, ...] = ()
) -> SpandrelModel:
    """Load the model file at *path* onto *device* in the dtype *precision* resolves to.

    The file is read on the CPU, so a file that is not a model never touches the GPU.
    ``.pth`` files are unpickled with spandrel's restricted unpickler.

    Args:
        path: A ``.pth``, ``.pt``, ``.ckpt`` or ``.safetensors`` file.
        device: Where the model runs.
        precision: ``auto``, ``fp32``, ``fp16`` or ``bf16`` (see ``resolve_dtype``).
        notes: Notes from resolving the device, kept with the model for the card.

    Raises:
        spandrel.UnsupportedModelError: The file holds no architecture spandrel knows.
        ValueError: The file's extension is not a model format.
    """
    from spandrel import ModelLoader

    descriptor = ModelLoader(device="cpu").load_from_file(path)
    descriptor.eval()
    dtype, note = resolve_dtype(
        precision,
        device,
        supports_half=descriptor.supports_half,
        supports_bfloat16=descriptor.supports_bfloat16,
    )
    descriptor.to(device)
    descriptor.to(dtype)
    all_notes = notes + ((note,) if note else ())
    return SpandrelModel(descriptor=descriptor, device=device, dtype=dtype, name=path.stem, notes=all_notes)


def release_model(model: SpandrelModel) -> None:
    """Return the device's cached memory once the pool lets go of *model*.

    The model itself is left as it is: a port may still hold it and run it, and its
    memory is freed when the last such reference goes.
    """
    import torch

    if model.device.type == "cuda":
        torch.cuda.empty_cache()
    elif model.device.type == "mps":
        torch.mps.empty_cache()
