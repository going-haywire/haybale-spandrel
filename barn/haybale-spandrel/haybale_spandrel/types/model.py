"""MODEL: a spandrel model loaded on a device, ready to run."""

from typing import Any

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
class MODEL(PrimitiveType[SpandrelModel | None]):
    """A model loaded on a device in a precision, ready to run.

    Load Model makes one from a model you pick; Upscale Image, Inpaint and Model Info
    take it. Several nodes can share one: the model is loaded once.
    """

    def __init__(self, value: Any = None, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self._value = _as_model(self._value)  # the type's default arrives as JSON

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"value": {}}``: a loaded model is never serialized."""
        return {"value": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> None:
        """Return ``None``: a loaded model never round-trips."""
        return None


class MODELField(PrimitiveField):
    """DataField for MODEL: holds a ``SpandrelModel``, or ``None`` until Load Model has run."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._default = _as_model(self._default)
        self._value = self._default

    def _set_own(self, value: Any) -> None:
        self._value = _as_model(value)


def _as_model(value: Any) -> SpandrelModel | None:
    return value if isinstance(value, SpandrelModel) else None


MODEL.field_class = MODELField
