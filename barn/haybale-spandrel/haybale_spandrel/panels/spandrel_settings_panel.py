"""Library settings panel for the Spandrel library, on the ``BarnsSurface`` surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from haywire.api.workbench import BarnsSurface, BasePanel, PanelLayout, hui, panel, render_schema

from haybale_spandrel.settings.spandrel_settings import SpandrelSettings

if TYPE_CHECKING:
    from haywire.api.workbench import SessionContext


@panel(
    surface=BarnsSurface,
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
