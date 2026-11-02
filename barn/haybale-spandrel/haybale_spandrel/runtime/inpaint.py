"""Run an inpainting model on an ``ImageValue`` and a mask."""

from __future__ import annotations

import threading

from haybale_image.value import ImageValue

from haybale_spandrel.runtime.tiling import Cancelled
from haybale_spandrel.runtime.upscale import _to_channels
from haybale_spandrel.value.spandrel_model import SpandrelModel


def inpaint_image(
    image: ImageValue, mask: ImageValue, model: SpandrelModel, *, cancel: threading.Event | None = None
) -> ImageValue:
    """Fill the parts of *image* that *mask* marks, and return the result on the model's device.

    The whole image runs at once. An alpha channel is kept as it was; a gray image
    comes back gray.

    Args:
        image: The picture to fill.
        mask: Where to fill: values above 0.5 are filled. Scaled to the image's size
            (nearest neighbour) when the sizes differ; a colour mask counts by brightness.
        model: An inpainting model.
        cancel: Checked once before the model runs.

    Raises:
        ValueError: The image or mask is empty, or the model is not an inpainting model.
        Cancelled: *cancel* was set.
    """
    import torch
    import torch.nn.functional as F

    if image.is_empty:
        raise ValueError("empty image")
    if mask.is_empty:
        raise ValueError("empty mask")
    if not model.is_inpaint:
        raise ValueError(f"{model.name} is a {model.purpose} model, not an inpainting model")
    if cancel is not None and cancel.is_set():
        raise Cancelled()

    pixels = image.tensor(model.device, model.dtype)
    color, alpha = (pixels[:, :3], pixels[:, 3:4]) if pixels.shape[1] == 4 else (pixels, None)
    with torch.inference_mode():
        m = _to_channels(mask.tensor(model.device, torch.float32), 1)
        if m.shape[2:] != color.shape[2:]:
            m = F.interpolate(m, size=tuple(color.shape[2:]), mode="nearest")
        m = (m > 0.5).to(model.dtype)
        with model.lock:
            out = model.descriptor(_to_channels(color, model.input_channels), m)  # type: ignore[call-arg]
        out = _to_channels(out, color.shape[1])
        if alpha is not None:
            out = torch.cat([out, alpha], 1)
    # A clone outside inference mode: later in-place conversions refuse inference tensors.
    return image.derive(ImageValue.from_tensor(out.clone(), source_depth=image.source_depth))
