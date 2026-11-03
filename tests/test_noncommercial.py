"""The non-commercial architectures: their licences and the registry the loader uses."""

from __future__ import annotations

from pathlib import Path

import spandrel

from haybale_spandrel.runtime.noncommercial import LICENSES, licence_of, registry


def test_every_extra_arch_has_a_licence() -> None:
    from spandrel_extra_arches import EXTRA_REGISTRY

    assert set(LICENSES) == {str(a.architecture.id) for a in EXTRA_REGISTRY}


def test_off_is_the_main_registry() -> None:
    assert registry(False) is spandrel.MAIN_REGISTRY


def test_on_adds_the_extras_to_a_copy() -> None:
    before = len(spandrel.MAIN_REGISTRY)
    combined = registry(True)
    assert combined is not spandrel.MAIN_REGISTRY
    assert len(combined) == before + len(LICENSES)
    assert "SRFormer" in combined
    assert "SRFormer" not in spandrel.MAIN_REGISTRY
    assert registry(True) is combined


def test_licence_of_main_arch_is_none(tiny_sr_path: Path) -> None:
    assert licence_of(spandrel.ModelLoader().load_from_file(tiny_sr_path)) is None
