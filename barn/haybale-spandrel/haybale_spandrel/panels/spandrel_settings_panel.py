"""Library settings panel for the Spandrel library, on the ``Barns`` surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from haywire.barn.builtin.surfaces import Barns
from haywire.ui import elements as hui
from haywire.ui.panel import BasePanel, PanelLayout
from haywire.ui.panel.decorator import panel
from haywire.ui.panel.render_utils import render_schema

from haybale_spandrel.settings.spandrel_settings import SpandrelSettings

if TYPE_CHECKING:
    from haywire.core.session.context import SessionContext


@panel(
    surface=Barns,
    label="Spandrel",
    description="Device, precision and memory budget for spandrel models.",
    icon=hui.icon.models,
    order=20,
    default_open=False,
)
class SpandrelSettingsPanel(BasePanel):
    """Device, precision and memory budget for spandrel models."""

    def draw(self, ctx: "SessionContext", layout: PanelLayout) -> None:
        registry = ctx.app.library_service.get_settings_registry()
        render_schema(ctx, SpandrelSettings, registry)
