"""Load a model file with spandrel onto a device in a precision, and free it again."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from haybale_spandrel.runtime.devices import resolve_dtype
from haybale_spandrel.runtime.noncommercial import LICENSES, licence_of, registry
from haybale_spandrel.value.spandrel_model import SpandrelModel

if TYPE_CHECKING:
    import torch

_NAMES = sorted(LICENSES)

#: Added to "unsupported model" while the non-commercial architectures are off.
UNSUPPORTED_HINT = (
    f"if it is one of {', '.join(_NAMES[:-1])} or {_NAMES[-1]}, "
    "turn on 'Non-commercial architectures' in the Spandrel settings"
)


def load_model(
    path: Path,
    device: torch.device,
    precision: str,
    *,
    notes: tuple[str, ...] = (),
    noncommercial: bool = False,
) -> SpandrelModel:
    """Load the model file at *path* onto *device* in the dtype *precision* resolves to.

    The file is read on the CPU, so a file that is not a model never touches the GPU.
    ``.pth`` files are unpickled with spandrel's restricted unpickler.

    Args:
        path: A ``.pth``, ``.pt``, ``.ckpt`` or ``.safetensors`` file.
        device: Where the model runs.
        precision: ``auto``, ``fp32``, ``fp16`` or ``bf16`` (see ``resolve_dtype``).
        notes: Notes from resolving the device, kept with the model for the card.
        noncommercial: Also detect the non-commercial architectures; a model of one
            carries its licence and a note saying so.

    Raises:
        spandrel.UnsupportedModelError: The file holds no architecture spandrel knows;
            while *noncommercial* is off, the message points at the setting.
        ValueError: The file's extension is not a model format.
    """
    from spandrel import ModelLoader, UnsupportedModelError

    try:
        descriptor = ModelLoader(device="cpu", registry=registry(noncommercial)).load_from_file(path)
    except UnsupportedModelError as exc:
        message = f"{path.name} holds no architecture spandrel knows"
        if not noncommercial:
            message += f"; {UNSUPPORTED_HINT}"
        raise UnsupportedModelError(message) from exc
    descriptor.eval()
    dtype, note = resolve_dtype(
        precision,
        device,
        supports_half=descriptor.supports_half,
        supports_bfloat16=descriptor.supports_bfloat16,
    )
    descriptor.to(device)
    descriptor.to(dtype)
    licence = licence_of(descriptor)
    all_notes = notes + ((note,) if note else ())
    if licence:
        all_notes += (f"{descriptor.architecture.name} is non-commercial only ({licence})",)
    return SpandrelModel(
        descriptor=descriptor, device=device, dtype=dtype, name=path.stem, notes=all_notes, licence=licence
    )


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
