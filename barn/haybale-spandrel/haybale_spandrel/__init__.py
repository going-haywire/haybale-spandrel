"""Haybale-Spandrel: upscale, restore and inpaint images with spandrel PyTorch models.

Component folders (registered in ``register_components``):
  - ``settings``, ``types``, ``nodes``, ``panels``, ``macros``, ``catalog``.

``runtime`` holds the torch code the nodes call; it imports torch only when a model runs.
"""

from pathlib import Path

from haywire.api import (
    BaseLibrary,
    CatalogRegistry,
    MacroRegistry,
    NodeRegistry,
    SettingsRegistry,
    TypeRegistry,
    library,
)

from haywire.api.workbench import PanelRegistry


@library(
    file_watcher=True,
)
class Library(BaseLibrary):
    """Spandrel library — load image models and run them on images."""

    def register_components(self):
        """Register the component folders with their registries."""
        base_path = Path(__file__).parent

        self.add_folder_to_registry(folder_path=str(base_path / "settings"), registry_cls=SettingsRegistry)
        self.add_folder_to_registry(folder_path=str(base_path / "types"), registry_cls=TypeRegistry)
        self.add_folder_to_registry(folder_path=str(base_path / "nodes"), registry_cls=NodeRegistry)
        self.add_folder_to_registry(folder_path=str(base_path / "panels"), registry_cls=PanelRegistry)
        self.add_folder_to_registry(folder_path=str(base_path / "macros"), registry_cls=MacroRegistry)
        self.add_folder_to_registry(folder_path=str(base_path / "catalog"), registry_cls=CatalogRegistry)

    def validate(self) -> bool:
        return True
