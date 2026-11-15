"""Inpaint — fill a masked region of an image with a loaded inpainting model."""

from __future__ import annotations

from haybale_image.value import ImageValue

from haywire.api import BaseNode, ExecutionContext, NodeType, node

from haybale_spandrel.runtime.inpaint import inpaint_image
from haybale_spandrel.runtime.tiling import Cancelled
from haybale_spandrel.value.spandrel_model import SpandrelModel


@node(
    label="Inpaint",
    description="Fills the masked part of an image with a loaded inpainting model",
    menu="image/models",
    search_tags=["inpaint", "fill", "remove", "erase", "mask", "lama", "model", "spandrel"],
    node_type=NodeType.DATA,
)
class InpaintNode(BaseNode):
    """Fills the masked part of an image with a loaded inpainting model, such as LaMa.

    Connect `Model` to Load Spandrel Model. White in `Mask` marks what to fill; a mask of a
    different size is stretched to the image. The whole image runs at once, and
    transparency is kept as it was. A super-resolution model here is an error. Fields
    stamped on the image carry over.
    """

    def init(self) -> None:
        from haybale_image.types.image import IMAGE, IMAGE_GRAY

        from haybale_spandrel.types.model import INPAINT_MODEL

        self.add(IMAGE.as_inlet("image", label="Image", description="Image to fill"))
        self.add(IMAGE_GRAY.as_inlet("mask", label="Mask", description="White marks what to fill"))
        self.add(INPAINT_MODEL.as_inlet("model", label="Model", description="A loaded inpainting model"))
        self.add(IMAGE.as_outlet("out_image", label="Image", description="The filled image"))

    def worker(
        self, context: ExecutionContext, image: ImageValue, mask: ImageValue, model: SpandrelModel | None
    ) -> None:
        if model is None:
            raise ValueError("no model — connect Load Spandrel Model")
        try:
            result = inpaint_image(image, mask, model, cancel=context.cancel)
        except Cancelled:
            return None
        self.out("out_image", result)
        return None
