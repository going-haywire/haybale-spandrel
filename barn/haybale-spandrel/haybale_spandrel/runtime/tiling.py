"""Split an image into overlapping tiles so a model runs within a memory budget.

``choose_tile`` picks a tile side from the budget; ``run_tiled`` runs a model over the
tiles and blends their overlaps, halving the tile and starting over when the device
runs out of memory or a tensor grows too large for it.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from haybale_spandrel.value.spandrel_model import SpandrelModel

#: Pixels each tile reaches into its neighbours; seams blend across this band.
OVERLAP = 16
#: Smallest tile side an automatic or out-of-memory choice goes down to.
MIN_TILE = 32
#: Memory per input pixel and channel, in units of the dtype's size, that a model run needs.
BYTES_PER_PIXEL_FACTOR = 64
#: Largest tensor, in elements, a convolution may produce (MPS refuses 2**31 and more).
MAX_ELEMENTS = 2**31 - 1
#: Feature channels assumed at the model's output resolution when capping a tile by ``MAX_ELEMENTS``.
OUTPUT_FEATURES = 64


class Cancelled(Exception):
    """The run was cancelled between two tiles."""


def choose_tile(
    model: SpandrelModel, width: int, height: int, *, tile_setting: int | None, budget_bytes: int
) -> int | None:
    """Return the tile side for a run, or ``None`` to run the whole image at once.

    Args:
        model: The model that runs.
        width: Input width in pixels.
        height: Input height in pixels.
        tile_setting: ``None`` picks from *budget_bytes*, small enough that no output-size
            tensor reaches ``MAX_ELEMENTS``; ``0`` never tiles, a positive
            number is the side in pixels. A model that tiles internally is never
            tiled; one that discourages tiling is tiled only for a positive number.
        budget_bytes: Memory one run may use.
    """
    from spandrel import ModelTiling

    if model.tiling == ModelTiling.INTERNAL or tile_setting == 0:
        return None
    req = model.size_requirements
    if tile_setting is not None:
        side = max(tile_setting, req.minimum)
    else:
        if model.tiling == ModelTiling.DISCOURAGED:
            return None
        element = model.dtype.itemsize
        per_pixel = element * (model.input_channels + model.output_channels * model.scale**2)
        side = math.isqrt(max(budget_bytes, 0) // max(per_pixel * BYTES_PER_PIXEL_FACTOR, 1))
        side = min(side, math.isqrt(MAX_ELEMENTS // OUTPUT_FEATURES) // model.scale)
        side = max(side, MIN_TILE, req.minimum)
    side = max(side - side % req.multiple_of, req.multiple_of)
    if side >= max(width, height):
        return None
    return side


def run_tiled(
    forward: Callable[[Tensor], Tensor],
    image: Tensor,
    *,
    scale: int,
    tile: int | None,
    cancel: threading.Event | None = None,
    on_tile: Callable[[int, int], None] | None = None,
) -> Tensor:
    """Run *forward* over *image* tile by tile and return the blended result.

    Args:
        forward: The model: ``1×C×h×w`` in, ``1×C'×h*scale×w*scale`` out, on the
            input's device and dtype.
        image: ``1×C×H×W``.
        scale: The model's scale.
        tile: Tile side in pixels, or ``None`` for the whole image.
        cancel: Checked before each tile.
        on_tile: Called after each tile with ``(done, total)``.

    Returns:
        ``1×C'×H*scale×W*scale`` on the input's device and dtype.

    Raises:
        Cancelled: *cancel* was set before a tile.
        torch.OutOfMemoryError: The device ran out of memory with tiles of
            ``MIN_TILE`` pixels; other errors meaning "too large" pass on the same way.
    """
    height, width = int(image.shape[2]), int(image.shape[3])
    while True:
        try:
            return _run(forward, image, scale=scale, tile=tile, cancel=cancel, on_tile=on_tile)
        except Exception as exc:
            if not _is_too_large(exc):
                raise
            side = tile if tile is not None else max(width, height)
            if side // 2 < MIN_TILE:
                raise
            tile = side // 2
            _empty_cache(image)


def _run(
    forward: Callable[[Tensor], Tensor],
    image: Tensor,
    *,
    scale: int,
    tile: int | None,
    cancel: threading.Event | None,
    on_tile: Callable[[int, int], None] | None,
) -> Tensor:
    import torch

    height, width = int(image.shape[2]), int(image.shape[3])
    if tile is None or tile >= max(width, height):
        _check(cancel)
        out = forward(image)
        if on_tile is not None:
            on_tile(1, 1)
        return out

    ys = range(0, height, tile)
    xs = range(0, width, tile)
    total = len(ys) * len(xs)
    acc: Tensor | None = None
    weights: Tensor | None = None
    done = 0
    for y0 in ys:
        for x0 in xs:
            _check(cancel)
            y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
            py0, px0 = max(y0 - OVERLAP, 0), max(x0 - OVERLAP, 0)
            py1, px1 = min(y1 + OVERLAP, height), min(x1 + OVERLAP, width)
            out = forward(image[:, :, py0:py1, px0:px1])
            if acc is None or weights is None:
                acc = torch.zeros(
                    (1, out.shape[1], height * scale, width * scale), device=out.device, dtype=torch.float32
                )
                weights = torch.zeros((1, 1, height * scale, width * scale), device=out.device)
            w = _feather(
                (py1 - py0) * scale,
                (px1 - px0) * scale,
                top=(y0 - py0) * scale,
                left=(x0 - px0) * scale,
                bottom=(py1 - y1) * scale,
                right=(px1 - x1) * scale,
                device=out.device,
            )
            region = (
                slice(None),
                slice(None),
                slice(py0 * scale, py1 * scale),
                slice(px0 * scale, px1 * scale),
            )
            acc[region] += out.float() * w
            weights[region] += w
            done += 1
            if on_tile is not None:
                on_tile(done, total)
    assert acc is not None
    assert weights is not None
    return (acc / weights).to(image.dtype)


def _feather(
    h: int, w: int, *, top: int, left: int, bottom: int, right: int, device: torch.device
) -> Tensor:
    """A ``1×1×h×w`` weight: 1 over the tile's own area, falling to 0 across the inner half of each margin.

    The outer half of a margin gets no weight: those pixels sit within the model's
    reach of the cut edge, where the tile saw padding instead of the neighbour.
    """
    import torch

    def edge(m: int) -> Tensor:
        """Weights across a margin of *m* pixels, from the tile's own area outward."""
        inner = m - m // 2
        w = torch.zeros(m, device=device)
        w[:inner] = torch.arange(inner, 0, -1, device=device) / (inner + 1)
        return w

    def ramp(n: int, before: int, after: int) -> Tensor:
        r = torch.ones(n, device=device)
        if before:
            r[:before] = edge(before).flip(0)
        if after:
            r[n - after :] = edge(after)
        return r

    return (ramp(h, top, bottom)[:, None] * ramp(w, left, right)[None, :])[None, None]


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _is_too_large(exc: Exception) -> bool:
    """Whether *exc* says the input was too large for the device, so a smaller tile may work."""
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    # MPS reports running out of memory as a plain RuntimeError, and a convolution over
    # 2**31 elements or more as a NotImplementedError naming convolution_overrideable.
    return (isinstance(exc, RuntimeError) and "out of memory" in text) or (
        isinstance(exc, NotImplementedError) and "convolution_overrideable" in text
    )


def _empty_cache(image: Tensor) -> None:
    import torch

    if image.device.type == "cuda":
        torch.cuda.empty_cache()
    elif image.device.type == "mps":
        torch.mps.empty_cache()
