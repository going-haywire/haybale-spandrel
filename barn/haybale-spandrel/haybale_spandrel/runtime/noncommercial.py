"""The non-commercial architectures of ``spandrel_extra_arches``: their licences and the registry.

Their code is licensed for non-commercial use only. The package is installed with this
library but imported only when the Spandrel setting *Non-commercial architectures* is on.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spandrel import ArchRegistry

#: Licence of each architecture's code, from its ``__arch/LICENSE`` in the spandrel repo.
#: The package's own metadata says MIT, which covers only its wrapper.
LICENSES: dict[str, str] = {
    "AdaCode": "CC BY-NC-SA 4.0",
    "CodeFormer": "S-Lab License 1.0",
    "DDColor": "Alibaba, academic and non-commercial use only",
    "FeMaSR": "CC BY-NC-SA 4.0",
    "M3SNet": "Academic Public License",
    "MAT": "CC BY-NC 4.0",
    "MIRNet2": "Academic Public License",
    "MPRNet": "Academic Public License",
    "Restormer": "Academic Public License",
    "SRFormer": "CC BY-NC 4.0",
}

_EXTRA_MODULE = "spandrel_extra_arches."


def registry(noncommercial: bool) -> ArchRegistry:
    """The architectures a model may be detected as.

    Off, spandrel's own ``MAIN_REGISTRY``; on, a copy of it with the non-commercial
    architectures added. The global registry is never changed.
    """
    from spandrel import MAIN_REGISTRY

    return _combined() if noncommercial else MAIN_REGISTRY


@functools.cache
def _combined() -> ArchRegistry:
    from spandrel import MAIN_REGISTRY
    from spandrel_extra_arches import EXTRA_REGISTRY

    combined = MAIN_REGISTRY.copy()
    combined.add(*EXTRA_REGISTRY)
    return combined


def licence_of(descriptor: Any) -> str | None:
    """The licence of *descriptor*'s architecture if it is a non-commercial one, else ``None``."""
    architecture = descriptor.architecture
    if not type(architecture).__module__.startswith(_EXTRA_MODULE):
        return None
    return LICENSES.get(str(architecture.id), "non-commercial use only")
