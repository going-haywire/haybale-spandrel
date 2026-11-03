"""SpandrelModel — a spandrel model loaded on a device in a precision, ready to run."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from spandrel import ImageModelDescriptor, MaskedImageModelDescriptor, ModelTiling, SizeRequirements

_CHANNEL_NAMES = {1: "GRAY", 3: "RGB", 4: "RGBA"}


@dataclass(frozen=True, eq=False)
class SpandrelModel:
    """A loaded model on its device and dtype, shared by every node that runs it.

    Runs of the model go through ``lock`` so two flows never drive one model at once.
    """

    descriptor: ImageModelDescriptor[Any] | MaskedImageModelDescriptor[Any]
    device: torch.device
    dtype: torch.dtype
    name: str
    notes: tuple[str, ...] = ()
    licence: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def architecture(self) -> str:
        """The architecture's name, e.g. ``ESRGAN`` or ``RealESRGAN Compact``."""
        return str(self.descriptor.architecture.name)

    @property
    def purpose(self) -> str:
        """``SR``, ``FaceSR``, ``Inpainting`` or ``Restoration``."""
        return str(self.descriptor.purpose)

    @property
    def scale(self) -> int:
        """How many times larger the output is than the input."""
        return int(self.descriptor.scale)

    @property
    def input_channels(self) -> int:
        """Channels the model takes: 1, 3 or 4."""
        return int(self.descriptor.input_channels)

    @property
    def output_channels(self) -> int:
        """Channels the model gives: 1, 3 or 4."""
        return int(self.descriptor.output_channels)

    @property
    def tiling(self) -> ModelTiling:
        """Whether the model may be run tile by tile."""
        return self.descriptor.tiling

    @property
    def size_requirements(self) -> SizeRequirements:
        """Input sizes the model accepts (minimum, multiple of, square)."""
        return self.descriptor.size_requirements

    @property
    def noncommercial(self) -> bool:
        """Whether the model's architecture is licensed for non-commercial use only."""
        return self.licence is not None

    @property
    def is_inpaint(self) -> bool:
        """Whether the model fills a masked region instead of mapping image to image."""
        return self.purpose == "Inpainting"

    @property
    def parameters(self) -> int:
        """Number of weights in the model."""
        return sum(p.numel() for p in self.descriptor.model.parameters())

    def status(self) -> str:
        """One line for the card, e.g. ``ESRGAN · 4x · RGB · mps fp16``.

        A non-commercial architecture adds `` · non-commercial``.
        """
        from haybale_spandrel.runtime.devices import dtype_name

        channels = _CHANNEL_NAMES.get(self.input_channels, f"{self.input_channels}ch")
        line = f"{self.architecture} · {self.scale}x · {channels} · {self.device} {dtype_name(self.dtype)}"
        return f"{line} · non-commercial" if self.noncommercial else line
