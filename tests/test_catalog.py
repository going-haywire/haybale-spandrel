"""The shipped catalog: every entry registers, is PyTorch, pinned and permissively licensed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

CATALOG = Path(__file__).parent.parent / "barn/haybale-spandrel/haybale_spandrel/catalog"
PERMISSIVE = {"MIT", "BSD-3-Clause", "BSD-2-Clause", "Apache-2.0", "WTFPL", "CC0-1.0", "CC-BY-4.0"}


def test_every_entry_registers_pinned_and_permissive(library_system: Any) -> None:
    from haybale_models.consumer import models_state

    catalog = models_state().catalog
    stems = sorted(p.stem for p in CATALOG.glob("*.json"))
    assert len(stems) == 4
    for stem in stems:
        entry: Any = catalog.get(f"haybale-spandrel:catalog:{stem}")
        assert entry is not None, stem
        assert entry.license in PERMISSIVE, (stem, entry.license)
        pytorch = [r for r in entry.resources if r.platform == "pytorch"]
        assert pytorch, stem
        assert all(len(r.sha256) == 64 for r in pytorch), stem
