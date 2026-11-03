"""The library loads and registers its settings and panel."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


def test_library_settings_have_their_defaults(library_system: Any) -> None:
    settings_mod = importlib.import_module("haybale_spandrel.settings.spandrel_settings")
    bag = settings_mod.SpandrelSettings()
    assert bag.device == "auto"
    assert bag.precision == "auto"
    assert bag.memory_budget == 0.0
    assert bag.noncommercial_architectures is False
    description = type(bag).noncommercial_architectures._description
    assert description.endswith("Applies from the next graph start.")


def test_panel_is_registered_under_barns(library_system: Any) -> None:
    from haywire.barn.builtin.surfaces import Barns
    from haywire.ui.panel.registry import PanelRegistry

    panel_cls = importlib.import_module(
        "haybale_spandrel.panels.spandrel_settings_panel"
    ).SpandrelSettingsPanel
    assert panel_cls.class_identity.surface is Barns
    panels = library_system.injector.get(PanelRegistry)
    assert panels.get(panel_cls.class_identity.registry_key) is not None


def test_panel_draws_the_spandrel_settings(monkeypatch: pytest.MonkeyPatch, library_system: Any) -> None:
    mod = importlib.import_module("haybale_spandrel.panels.spandrel_settings_panel")
    calls = []
    monkeypatch.setattr(mod, "render_schema", lambda ctx, schema_cls, registry: calls.append(schema_cls))
    panel = mod.SpandrelSettingsPanel.__new__(mod.SpandrelSettingsPanel)
    panel.draw(MagicMock(), MagicMock())
    assert calls == [mod.SpandrelSettings]
