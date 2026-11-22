"""Spandrel Model Info — the facts of a loaded model: architecture, scale, channels, device, precision."""

from __future__ import annotations

from haywire.api import BOOL, INT, STRING, BaseNode, ExecutionContext, NodeType, node

from haybale_spandrel.runtime.devices import dtype_name
from haybale_spandrel.value.spandrel_model import SpandrelModel


@node(
    label="Spandrel Model Info",
    description="Tells what a loaded spandrel model is: architecture, purpose, scale, device, precision",
    menu="models",
    search_tags=["model", "info", "scale", "architecture", "spandrel"],
    node_type=NodeType.DATA,
)
class SpandrelModelInfoNode(BaseNode):
    """Tells what a loaded model is.

    - `Architecture`: the network design, e.g. `ESRGAN` or `RealESRGAN Compact`.
    - `Purpose`: `SR` (enlarges), `Restoration` (same size), `FaceSR` or `Inpainting`.
    - `Scale`: how many times larger the result is.
    - `Input Channels`, `Output Channels`: 1 (gray), 3 (RGB) or 4 (RGBA).
    - `Tiling`: `SUPPORTED`, `DISCOURAGED` (runs whole unless a tile size is set) or
      `INTERNAL` (the model tiles itself).
    - `fp16`, `bf16`: whether the model can run in that precision.
    - `Min Size`, `Multiple Of`: input sizes the model needs; smaller or odd sizes
      are padded for it.
    - `Non-commercial`: whether the architecture's code is licensed for
      non-commercial use only.
    - `Device`, `Precision`: where and how this copy runs.
    - `Parameters`: the number of weights.
    """

    def init(self) -> None:
        from haybale_spandrel.types.model import LOADED_MODEL

        self.add(LOADED_MODEL.as_inlet("model", label="Model", description="Loaded model to describe"))
        self.add(
            STRING.as_outlet(
                "architecture", label="Architecture", description="Architecture name, such as ESRGAN"
            )
        )
        self.add(
            STRING.as_outlet(
                "purpose", label="Purpose", description="What the model does, such as SR or Inpainting"
            )
        )
        self.add(INT.as_outlet("scale", label="Scale", description="Upscale factor; 1 for restoration"))
        self.add(
            INT.as_outlet(
                "input_channels", label="Input Channels", description="Channels the model takes: 1, 3 or 4"
            )
        )
        self.add(
            INT.as_outlet(
                "output_channels", label="Output Channels", description="Channels the model returns"
            )
        )
        self.add(
            STRING.as_outlet(
                "tiling",
                label="Tiling",
                description="Whether it can run in tiles: SUPPORTED, DISCOURAGED or INTERNAL",
            )
        )
        self.add(
            BOOL.as_outlet(
                "supports_fp16", label="fp16", description="Whether it runs in half precision (fp16)"
            )
        )
        self.add(BOOL.as_outlet("supports_bf16", label="bf16", description="Whether it runs in bfloat16"))
        self.add(INT.as_outlet("min_size", label="Min Size", description="Smallest input side in pixels"))
        self.add(
            INT.as_outlet(
                "multiple_of", label="Multiple Of", description="Input sides must be a multiple of this"
            )
        )
        self.add(
            BOOL.as_outlet(
                "noncommercial",
                label="Non-commercial",
                description="Whether the model's architecture is licensed for non-commercial use only",
            )
        )
        self.add(
            STRING.as_outlet("device", label="Device", description="Where it runs, such as mps or cuda:0")
        )
        self.add(
            STRING.as_outlet(
                "precision", label="Precision", description="Number format it runs in, such as fp16"
            )
        )
        self.add(INT.as_outlet("parameters", label="Parameters", description="Number of weights"))

    def worker(self, context: ExecutionContext, model: SpandrelModel | None) -> None:
        if model is None:
            raise ValueError("no model — connect Load Spandrel Model")
        req = model.size_requirements
        self.out("architecture", model.architecture)
        self.out("purpose", model.purpose)
        self.out("scale", model.scale)
        self.out("input_channels", model.input_channels)
        self.out("output_channels", model.output_channels)
        self.out("tiling", model.tiling.name)
        self.out("supports_fp16", bool(model.descriptor.supports_half))
        self.out("supports_bf16", bool(model.descriptor.supports_bfloat16))
        self.out("min_size", int(req.minimum))
        self.out("multiple_of", int(req.multiple_of))
        self.out("noncommercial", model.noncommercial)
        self.out("device", str(model.device))
        self.out("precision", dtype_name(model.dtype))
        self.out("parameters", model.parameters)
        return None
