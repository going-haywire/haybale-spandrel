"""Upscale Image — run a super-resolution or restoration model on an image."""

from __future__ import annotations

from haybale_image.value import ImageValue

from haywire.barn.builtin.progress import ProgressSettings, progress_field
from haywire.core.execution.execution_context import ExecutionContext
from haywire.core.node import BaseNode, NodeType, node
from haywire.core.node.node_warning import NodeWarning
from haywire.core.settings import bag

from haybale_spandrel.runtime.devices import memory_budget_bytes
from haybale_spandrel.runtime.tiling import Cancelled
from haybale_spandrel.runtime.upscale import ALPHA_MODES, upscale_image
from haybale_spandrel.settings.spandrel_settings import SpandrelSettings
from haybale_spandrel.value.spandrel_model import SpandrelModel

#: Suggested tile sizes; any number of pixels can be typed in.
TILE_SIZES = ("auto", "none", "128", "256", "512", "1024")

#: Warning kind of the notes on the card.
KIND = "spandrel"


class UpscaleProgress(ProgressSettings):
    """The tiles of the current model pass."""

    main = progress_field("Tiles")


def parse_tile_size(text: str) -> int | None:
    """Return the tile setting for ``choose_tile``: ``None`` for auto, ``0`` for none, else pixels.

    Raises:
        ValueError: *text* is not ``auto``, ``none`` or a positive whole number.
    """
    value = (text or "auto").strip().lower()
    if value == "auto":
        return None
    if value == "none":
        return 0
    if value.isdigit() and int(value) > 0:
        return int(value)
    raise ValueError(f"tile size {text!r}: use auto, none or a number of pixels")


@node(
    label="Upscale Image",
    description="Enlarges or restores an image with a loaded model",
    menu="image/models",
    search_tags=["upscale", "super resolution", "sr", "enlarge", "restore", "esrgan", "model", "spandrel"],
    node_type=NodeType.DATA,
)
class UpscaleImageNode(BaseNode):
    """Enlarges or restores an image with a loaded model.

    Connect `Model` to Load Spandrel Model. The result stays on the model's device, so a
    following model step runs without copying.

    - `Scale`: size of the result relative to the image; `0` uses the model's own
      scale. A larger scale runs the model again (a 4x model twice for 8x or 16x);
      a scale between steps is reached by scaling the result down smoothly.
    - `Tile Size`: large images run in tiles that fit the memory budget (*Spandrel*
      settings). `auto` picks the size, `none` runs the whole image at once, a number
      sets the tile side in pixels. A model that tiles itself is never tiled.
    - `Alpha`: for images with transparency. `resize` scales the transparency
      smoothly; `model` runs it through the model too, which takes twice as long and
      keeps edges sharper.

    A gray image comes back gray. A colour image through a grayscale model comes
    back gray, with a note on the card.

    Fields stamped on the image carry over unchanged, so a field holding a position or
    a size in pixels no longer matches the result.
    """

    progress = bag(UpscaleProgress)

    def init(self) -> None:
        from haybale_image.types.image import IMAGE

        from haywire.barn.builtin.types import FLOAT, STRING
        from haywire.barn.builtin.widgets import SelectWidget

        from haybale_spandrel.types.model import IMAGE_MODEL

        self.add(IMAGE.as_inlet("image", label="Image", description="Image to process"))
        self.add(IMAGE_MODEL.as_inlet("model", label="Model", description="A loaded image-to-image model"))
        self.add(
            FLOAT.as_inlet(
                "scale",
                label="Scale",
                default=0.0,
                description="Size of the result relative to the input; 0 is the model's own scale",
            )
        )
        self.add(
            STRING.as_config(
                "tile_size",
                default="auto",
                label="Tile Size",
                widget=SelectWidget.config(
                    properties={"options": list(TILE_SIZES), "new_value_mode": "add-unique"}
                ),
                description="Tile edge in pixels; auto fits the memory, none runs the whole image at once",
            )
        )
        self.add(
            STRING.as_config(
                "alpha",
                default="resize",
                label="Alpha",
                widget=SelectWidget.config(properties={"options": list(ALPHA_MODES)}),
                description="Alpha channel: resize scales it smoothly, model runs it through the model",
            )
        )
        self.add(IMAGE.as_outlet("out_image", label="Image", description="The processed image"))

    def worker(
        self,
        context: ExecutionContext,
        image: ImageValue,
        model: SpandrelModel | None,
        scale: float,
        tile_size: str,
        alpha: str,
    ) -> None:
        if model is None:
            raise ValueError("no model — connect Load Spandrel Model")
        tile_setting = parse_tile_size(tile_size)
        budget = memory_budget_bytes(model.device, float(SpandrelSettings().memory_budget))
        with self.progress.busy("main", "upscaling") as span:
            try:
                result, notes = upscale_image(
                    image,
                    model,
                    scale=float(scale),
                    tile_setting=tile_setting,
                    alpha=alpha,
                    budget_bytes=budget,
                    cancel=context.cancel,
                    on_progress=lambda done, total: span.step(done / total, f"tile {done} of {total}"),
                )
            except Cancelled:
                return None
        self.wrapper.replace_warnings(KIND, [NodeWarning(n, source_version=None, kind=KIND) for n in notes])
        self.out("out_image", result)
        return None
