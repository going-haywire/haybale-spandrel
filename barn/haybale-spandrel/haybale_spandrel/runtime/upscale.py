"""Run an image-to-image model (super-resolution or 1x restoration) on an ``ImageValue``."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from haybale_image.value import ImageValue

from haybale_spandrel.runtime.tiling import choose_tile, run_tiled

if TYPE_CHECKING:
    from torch import Tensor

    from haybale_spandrel.value.spandrel_model import SpandrelModel

#: Values of the Alpha setting.
ALPHA_MODES = ("resize", "model")

_LUMA = (0.299, 0.587, 0.114)


def upscale_image(
    image: ImageValue,
    model: SpandrelModel,
    *,
    scale: float = 0.0,
    tile_setting: int | None = None,
    alpha: str = "resize",
    budget_bytes: int,
    cancel: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[ImageValue, list[str]]:
    """Run *model* on *image* and return the result with notes for the card.

    The result keeps the input's layout: a gray image through a colour model comes back
    gray; a colour image through a gray model comes back gray, with a note. It stays on
    the model's device.

    Args:
        image: The picture to process.
        model: An image-to-image model.
        scale: Size of the result relative to the input; ``0`` is the model's own scale.
            The model runs as many times as it takes to reach it, then the result is
            scaled down to the exact size.
        tile_setting: See ``choose_tile``.
        alpha: For an RGBA image and a model without an alpha channel: ``resize`` scales
            the alpha channel smoothly, ``model`` runs it through the model as well.
        budget_bytes: Memory one model run may use, for the tile size.
        cancel: Checked between tiles.
        on_progress: Called with ``(done, total)`` tiles of the current pass.

    Raises:
        ValueError: The image is empty, the model is an inpainting model, *scale* is
            negative, or *alpha* is not one of ``ALPHA_MODES``.
        Cancelled: *cancel* was set between two tiles.
    """
    import torch

    if image.is_empty:
        raise ValueError("empty image")
    if model.is_inpaint:
        raise ValueError(f"{model.name} is an inpainting model — use Inpaint")
    if scale < 0:
        raise ValueError(f"scale must be 0 or more, got {scale}")
    if alpha not in ALPHA_MODES:
        raise ValueError(f"alpha must be one of {', '.join(ALPHA_MODES)}, got {alpha!r}")

    notes: list[str] = []
    target = scale if scale > 0 else float(model.scale)
    size = (round(image.height * target), round(image.width * target))
    if 0 in size:
        raise ValueError(f"scale {scale} makes the image empty")

    pixels = image.tensor(model.device, model.dtype)
    color, alpha_channel = pixels, None
    if pixels.shape[1] == 4 and model.input_channels != 4:
        color, alpha_channel = pixels[:, :3], pixels[:, 3:4]
    if model.input_channels == 1 and color.shape[1] == 3:
        notes.append("model is grayscale — colour is dropped")

    def run(t: Tensor) -> Tensor:
        return _run_passes(
            model,
            _to_channels(t, model.input_channels),
            target=target,
            tile_setting=tile_setting,
            budget_bytes=budget_bytes,
            cancel=cancel,
            on_progress=on_progress,
        )

    with torch.inference_mode():
        out = _resize(run(color), size)
        out = _to_channels(out, min(color.shape[1], model.output_channels))
        if alpha_channel is not None:
            a = run(alpha_channel) if alpha == "model" else alpha_channel
            out = torch.cat(
                [out if out.shape[1] == 3 else out.expand(-1, 3, -1, -1), _alpha_of(_resize(a, size))], 1
            )
        out = out.clamp(0, 1).to(model.dtype)
    # A clone outside inference mode: later in-place conversions refuse inference tensors.
    result = ImageValue.from_tensor(out.clone(), source_depth=image.source_depth)
    return image.derive(result), notes


def _run_passes(
    model: SpandrelModel,
    t: Tensor,
    *,
    target: float,
    tile_setting: int | None,
    budget_bytes: int,
    cancel: threading.Event | None,
    on_progress: Callable[[int, int], None] | None,
) -> Tensor:
    reached = 1.0
    while True:
        tile = choose_tile(
            model, t.shape[3], t.shape[2], tile_setting=tile_setting, budget_bytes=budget_bytes
        )
        t = run_tiled(locked(model), t, scale=model.scale, tile=tile, cancel=cancel, on_tile=on_progress)
        reached *= model.scale
        if model.scale == 1 or reached >= target - 1e-9:
            return t


def locked(model: SpandrelModel) -> Callable[[Tensor], Tensor]:
    """Return *model*'s forward pass, run under the model's lock."""

    def forward(t: Tensor) -> Tensor:
        with model.lock:
            return model.descriptor(t)  # type: ignore[call-arg]

    return forward


def _to_channels(t: Tensor, channels: int) -> Tensor:
    """Convert a ``1×C×H×W`` tensor between gray (1), RGB (3) and RGBA (4)."""
    import torch

    have = int(t.shape[1])
    if have == channels:
        return t
    if channels == 1:
        rgb = t[:, :3] if have >= 3 else t
        weights = torch.tensor(_LUMA, device=t.device, dtype=t.dtype).view(1, 3, 1, 1)
        return (rgb * weights).sum(1, keepdim=True) if rgb.shape[1] == 3 else rgb
    if channels == 3:
        return t.expand(-1, 3, -1, -1) if have == 1 else t[:, :3]
    rgb = _to_channels(t, 3)
    return torch.cat([rgb, torch.ones_like(rgb[:, :1])], 1)


def _alpha_of(t: Tensor) -> Tensor:
    return t if t.shape[1] == 1 else t.mean(1, keepdim=True)


def _resize(t: Tensor, size: tuple[int, int]) -> Tensor:
    """Scale *t* to ``(height, width)`` with antialiased bicubic, in float32."""
    import torch.nn.functional as F

    if (int(t.shape[2]), int(t.shape[3])) == size:
        return t
    out = F.interpolate(t.float(), size=size, mode="bicubic", antialias=True, align_corners=False)
    return out.clamp_(0, 1).to(t.dtype)
