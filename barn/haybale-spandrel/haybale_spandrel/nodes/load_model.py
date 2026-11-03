"""Load Model — load a picked model onto a device, ready for Upscale Image, Inpaint and Model Info."""

from __future__ import annotations

from typing import Any

from haybale_models.consumer import models_state, release_node, resolve_port
from haybale_models.value.model_ref import ModelRef

from haywire.barn.builtin.types import STRING
from haywire.barn.builtin.widgets import SimpleLabelWidget
from haywire.core.execution.execution_context import ExecutionContext
from haywire.core.node import BaseNode, NodeType, node
from haywire.core.node.node_warning import NodeWarning
from haywire.core.settings import NodeSettings, Promotable, bag, setting, shadow
from haywire.core.types.enums import PortType

from haybale_spandrel.runtime.devices import resolve_device
from haybale_spandrel.runtime.loader import load_model, release_model
from haybale_spandrel.settings.spandrel_settings import SpandrelSettings

#: Warning kind of the device and precision notes on the card.
KIND = "spandrel"

#: The card line while no model is loaded.
NOT_LOADED = "< shows after the first run >"

#: Variants Load Model prefers from a catalog entry: PyTorch weights, safetensors first.
PREFER = [{"platform": "pytorch", "type": "safetensors"}, {"platform": "pytorch"}]


class LoadModelSettings(NodeSettings):
    """Device and precision of one Load Model node."""

    device = shadow(SpandrelSettings.device)
    precision = shadow(SpandrelSettings.precision)


class LoadModelStatus(NodeSettings):
    """What the node loaded, shown on its card."""

    loaded = setting[STRING](
        NOT_LOADED,
        label="Loaded",
        description="Architecture, scale, channels, device and precision of the loaded model",
        widget=SimpleLabelWidget.config(),
        promotable=Promotable.CONFIG,
        promote_default=PortType.CONFIG,
    )


@node(
    label="Load Model",
    description="Loads a model onto the GPU or CPU, ready for Upscale Image, Inpaint and Model Info",
    menu="models",
    search_tags=["model", "load", "spandrel", "upscale", "esrgan", "pytorch", "torch", "gpu"],
    node_type=NodeType.DATA,
)
class LoadModelNode(BaseNode):
    """Loads a model onto the GPU or CPU, ready for Upscale Image, Inpaint and Model Info.

    Pick the model on the card: a catalog entry, or a file from the Models folder
    or the Hugging Face cache. A model that is not on this machine is an error
    until it is fetched in the Models editor. The architecture is detected from
    the file (`.pth`, `.safetensors`, `.ckpt`, `.pt`).

    - `Device`: where the model runs. `auto` uses a CUDA GPU, else the Apple GPU,
      else the CPU. A device this machine lacks runs on the CPU, with a note.
    - `Precision`: `auto` runs in half precision on a GPU when the model supports
      it, else full. A precision the model cannot run falls back to fp32, with a
      note.

    Both default to the library's *Spandrel* settings. Load Model nodes that pick
    the same model with the same device and precision share one loaded copy. The
    model is loaded when the graph runs and freed when it stops.
    """

    settings = bag(LoadModelSettings)
    status = bag(LoadModelStatus)

    def init(self) -> None:
        from haybale_models.types.model_ref import MODEL_REF

        from haybale_spandrel.types.model import MODEL

        self.add(
            MODEL_REF.as_inlet(
                "model",
                label="Model",
                widget_config={"properties": {"filter": {}, "prefer": PREFER}},
                description="Model to load: a catalog entry or a file",
            )
        )
        self.add(
            MODEL.as_outlet(
                "loaded",
                label="Model",
                description="The loaded model, for Upscale Image, Inpaint and Model Info",
            )
        )

    def post_init(self) -> None:
        self.settings._subscribe(self._on_settings_changed)

    def _on_settings_changed(self, name: str, value: Any, old: Any) -> None:
        """A new device or precision needs another load: run at the next execution."""
        self.run_again()

    def worker(self, context: ExecutionContext, model: ModelRef) -> None:
        device, device_note = resolve_device(str(self.settings.device))
        precision = str(self.settings.precision)
        try:
            resolved = resolve_port(self, "model")
        except Exception:
            self._let_go()
            raise
        key = (resolved.key, (str(device), precision))
        if getattr(self.cache, "key", None) != key:
            self._let_go()
        loaded = models_state().pool.acquire(
            key,
            self,
            load=lambda: load_model(resolved.path, device, precision),
            release=release_model,
        )
        self.cache.key = key
        notes = ([device_note] if device_note else []) + list(loaded.notes)
        self.wrapper.replace_warnings(KIND, [NodeWarning(n, source_version=None, kind=KIND) for n in notes])
        self.status.loaded = loaded.status()
        self.out("loaded", loaded)
        return None

    def on_shutdown(self, context: ExecutionContext) -> None:
        self._let_go()
        # Drop the wire's reference too, so the model's memory is freed while the graph is stopped.
        self.out("loaded", None)
        self.run_again()

    def on_teardown(self) -> None:
        self._let_go()

    def _let_go(self) -> None:
        """Release this node's hold on its loaded model, if any."""
        self.cache.key = None
        try:
            release_node(self)
        except RuntimeError:
            pass  # haybale-models already disabled: its pool released everything
        try:
            self.status.loaded = NOT_LOADED
        except Exception:  # noqa: BLE001
            pass

    def model_preference(self, port_id: str) -> Any:
        """The catalog variants this node prefers: PyTorch weights, safetensors first."""
        return PREFER
