"""Load Spandrel Model — load a picked spandrel model onto a device, for Upscale Image and Inpaint."""

from __future__ import annotations

from typing import Any

from haybale_models.consumer import models_state, release_node, resolve_port
from haybale_models.value.model_ref import ModelRef

from haywire.api import (
    ADD,
    STRING,
    BaseNode,
    ExecutionContext,
    NodeSettings,
    NodeType,
    NodeWarning,
    PortType,
    Promotable,
    SimpleLabelWidget,
    bag,
    node,
    setting,
    shadow,
)

from haybale_spandrel.runtime.devices import resolve_device
from haybale_spandrel.runtime.loader import load_model, release_model
from haybale_spandrel.settings.spandrel_settings import SpandrelSettings

#: Warning kind of the device and precision notes on the card.
KIND = "spandrel"

#: The card line while no model is loaded.
NOT_LOADED = "< shows after the first run >"

#: Variants Load Spandrel Model prefers from a catalog entry: PyTorch weights, safetensors first.
PREFER = [{"platform": "pytorch", "type": "safetensors"}, {"platform": "pytorch"}]


class LoadModelSettings(NodeSettings):
    """Device and precision of one Load Spandrel Model node."""

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
    label="Load Spandrel Model",
    description="Loads a spandrel (PyTorch image) model onto the GPU or CPU, for Upscale Image or Inpaint",
    menu="models",
    search_tags=["model", "load", "load model", "spandrel", "upscale", "esrgan", "pytorch", "torch", "gpu"],
    node_type=NodeType.DATA,
)
class LoadModelNode(BaseNode):
    """Loads a spandrel (PyTorch image) model onto the GPU or CPU, for Upscale Image or Inpaint.

    Connect the `Model` outlet first, to the node that runs the model: the outlet
    takes the kind of model that node needs (an image model for Upscale Image, an
    inpainting model for Inpaint, any kind for Spandrel Model Info), and the
    `Model` picker appears, offering only models of that kind. The outlet keeps its
    kind until the node is reset. A model of another kind, picked as a file, is
    reported when it loads.

    Pick the model on the card: a catalog entry, or a file from the Models folder
    or the Hugging Face cache. A model that is not on this machine is an error
    until it is fetched in the Models editor. The architecture is detected from
    the file (`.pth`, `.safetensors`, `.ckpt`, `.pt`).

    - `Device`: where the model runs. `auto` uses a CUDA GPU, else the Apple GPU,
      else the CPU. A device this machine lacks runs on the CPU, with a note.
    - `Precision`: `auto` runs in half precision on a GPU when the model supports
      it, else full. A precision the model cannot run falls back to fp32, with a
      note.

    - Models of the non-commercial architectures (CodeFormer, SRFormer, Restormer
      and others) load only with the library setting *Non-commercial architectures*
      on, and say so on the card: their code is licensed for non-commercial use only.

    Device and precision default to the library's *Spandrel* settings. Nodes that pick the same
    model with the same device and precision share one loaded copy. The model is
    loaded when the graph runs and freed when it stops.
    """

    settings = bag(LoadModelSettings)
    status = bag(LoadModelStatus)

    def init(self) -> None:
        self.add(
            ADD.as_outlet(
                "loaded",
                label="Model",
                description="Connect it to the node that runs the model; it takes that node's kind of model",
                on_connect="_on_outlet_connected",
            )
        )

    def _on_outlet_connected(self, port: Any, edge_wrapper: Any) -> None:
        """Give the open outlet the kind of model the connected inlet takes, once.

        An inlet outside the model family makes it ``MODEL_BASE``, so that edge shows as
        invalid instead of carrying a model where none fits.
        """

        from haybale_spandrel.types.model import MODEL_BASE

        if port.type_cls is not ADD:
            return
        sink = edge_wrapper._inlet_port.stored_type
        kind = sink if isinstance(sink, type) and issubclass(sink, MODEL_BASE) else MODEL_BASE
        with self.rejig(include=[port.id, "model"]):
            self._add_model_inlet(kind)
            self.add(
                kind.as_outlet(
                    port.id,
                    label="Model",
                    description=f"The loaded model: {kind.class_identity.label.lower()}",
                    on_connect="_on_outlet_connected",
                )
            )
        self.wrapper.redraw()

    def _add_model_inlet(self, kind: Any) -> None:
        """Add the model picker, offering the catalog entries of *kind*."""
        from haybale_models.types.model_ref import MODEL_REF

        properties = {"filter": dict(kind.catalog_filter or {}), "prefer": PREFER}
        self.add(
            MODEL_REF.as_inlet(
                "model",
                label="Model",
                widget_config={"properties": properties},
                description="Model to load: a catalog entry or a file",
            )
        )

    def _outlet_kind(self) -> Any:
        """The model type the outlet resolved to, or ``None`` while it is open."""
        from haybale_spandrel.types.model import MODEL_BASE

        kind = self.ports["loaded"].type_cls
        return kind if isinstance(kind, type) and issubclass(kind, MODEL_BASE) else None

    def post_init(self) -> None:
        self.settings._subscribe(self._on_settings_changed)

    def _on_settings_changed(self, name: str, value: Any, old: Any) -> None:
        """A new device or precision needs another load: run at the next execution."""
        self.run_again()

    def worker(self, context: ExecutionContext, model: ModelRef) -> None:
        kind = self._outlet_kind()
        if kind is None:
            raise ValueError("connect the Model outlet first: it decides which kind of model to load")
        device, device_note = resolve_device(str(self.settings.device))
        precision = str(self.settings.precision)
        noncommercial = bool(SpandrelSettings().noncommercial_architectures)
        try:
            resolved = resolve_port(self, "model")
        except Exception:
            self._let_go()
            raise
        key = (resolved.key, (str(device), precision, noncommercial))
        if getattr(self.cache, "key", None) != key:
            self._let_go()
        loaded = models_state().pool.acquire(
            key,
            self,
            load=lambda: load_model(resolved.path, device, precision, noncommercial=noncommercial),
            release=release_model,
        )
        self.cache.key = key
        if not kind.fits(loaded):
            self._let_go()
            raise ValueError(
                f"{loaded.name} is a {loaded.purpose} model, which does not fit this outlet "
                f"({kind.class_identity.label.lower()})"
            )
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
