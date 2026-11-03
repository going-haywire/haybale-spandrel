"""MODEL_BASE and its kinds: a spandrel model loaded on a device, ready to run."""

from typing import Any, ClassVar

from haywire.barn.builtin.wires import DataWire
from haywire.core.types import PrimitiveField, PrimitiveType, StoreStrategy
from haywire.core.types import type as type_decorator

from haybale_spandrel.value.spandrel_model import SpandrelModel


@type_decorator(
    wire_type=DataWire,
    label="Loaded model",
    description="A model loaded on a device, ready to run. Never saved with the graph.",
    color="#d97ab0",
    store_strategy=StoreStrategy.NEVER,
    default={"value": {}},
)
class MODEL_BASE(PrimitiveType[SpandrelModel | None]):
    """A model loaded on a device in a precision, ready to run, of any kind.

    Spandrel Model Info takes any loaded model through it. Upscale Image and Inpaint take
    one kind each (`IMAGE_MODEL`, `INPAINT_MODEL`), so a Load Spandrel Model wired to them
    becomes that kind and offers only the models that fit.
    """

    #: The catalog fields a model of this kind must match, for the model picker;
    #: ``None`` offers every entry.
    catalog_filter: ClassVar[dict[str, list[Any]] | None] = None

    def __init__(self, value: Any = None, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self._value = _as_model(self._value)  # the type's default arrives as JSON

    @classmethod
    def fits(cls, model: SpandrelModel) -> bool:
        """Return whether *model* is of this kind."""
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"value": {}}``: a loaded model is never serialized."""
        return {"value": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> None:
        """Return ``None``: a loaded model never round-trips."""
        return None


@type_decorator(
    wire_type=DataWire,
    label="Image model",
    description="A loaded image-to-image model: upscaling and restoration",
    color="#d97ab0",
    store_strategy=StoreStrategy.NEVER,
    default={"value": {}},
)
class IMAGE_MODEL(MODEL_BASE):
    """A loaded model that turns an image into an image.

    Super-resolution, face super-resolution and restoration models are of this
    kind. Upscale Image takes it.
    """

    catalog_filter = {"purpose": ["SR", "FaceSR", "Restoration"]}

    @classmethod
    def fits(cls, model: SpandrelModel) -> bool:
        """Return whether *model* turns an image into an image."""
        return not model.is_inpaint


@type_decorator(
    wire_type=DataWire,
    label="Inpainting model",
    description="A loaded model that fills a masked part of an image",
    color="#c46aa0",
    store_strategy=StoreStrategy.NEVER,
    default={"value": {}},
)
class INPAINT_MODEL(MODEL_BASE):
    """A loaded model that fills the masked part of an image.

    Inpaint takes it.
    """

    catalog_filter = {"purpose": ["Inpainting"]}

    @classmethod
    def fits(cls, model: SpandrelModel) -> bool:
        """Return whether *model* is an inpainting model."""
        return model.is_inpaint


class MODELField(PrimitiveField):
    """DataField for the model types: a ``SpandrelModel``, or ``None`` until Load Spandrel Model ran."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._default = _as_model(self._default)
        self._value = self._default

    def _set_own(self, value: Any) -> None:
        self._value = _as_model(value)


def _as_model(value: Any) -> SpandrelModel | None:
    return value if isinstance(value, SpandrelModel) else None


MODEL_BASE.field_class = MODELField
