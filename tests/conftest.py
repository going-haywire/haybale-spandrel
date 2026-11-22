"""Fixtures for this repo's own suite.

Mirrors the monorepo's ``library_system`` fixture: the installed haywire libraries
(haybale-core, -image, -models, ...) load through their entry points, this repo's
library from ``barn/``. Model files are tiny random-weight spandrel architectures
built on the fly, so no test touches the network.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from haywire.core.volumes import VolumeStore, get_volume_store, set_volume_store


@pytest.fixture(scope="session")
def project_root() -> Path:
    """The repo root (the directory holding ``barn/``)."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session", autouse=True)
def _isolated_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point HOME at a temp dir so library enables never write into the real ``~/.haywire``."""
    home = tmp_path_factory.mktemp("home")
    previous = os.environ.get("HOME")
    os.environ["HOME"] = str(home)
    yield home
    if previous is not None:
        os.environ["HOME"] = previous


@pytest.fixture(scope="session")
def library_system(project_root: Path, _isolated_home: Path) -> Iterator[Any]:
    """A fully initialised library system with this repo's library loaded."""
    from haywire.core.di.config import set_global_injector, set_library_system
    from haywire.core.di.test_config import create_test_library_system

    service = create_test_library_system(
        workspace_root=str(project_root),
        library_paths=[str(project_root / "barn")],
        load_libraries=True,
        enable_file_watching=False,
    )
    set_library_system(service)
    set_global_injector(service.injector)

    yield service

    lib_registry = service.get_library_registry()
    if hasattr(lib_registry, "stop_file_watching"):
        lib_registry.stop_file_watching()
    set_library_system(None)


@pytest.fixture
def graph(library_system: Any) -> Any:
    """An empty graph on the loaded library system, validating inline."""
    from haywire.api.workbench import BaseGraph
    from haywire.core.graph.scheduler import SyncScheduler

    return BaseGraph(filestem="Spandrel Test Graph", validation_scheduler=SyncScheduler())


@pytest.fixture
def models_env(tmp_path: Path) -> Iterator[VolumeStore]:
    """A temp VolumeStore with both model volumes pointed at tmp_path subfolders."""
    from haybale_models.keys import HF_VOLUME, MODELS_VOLUME

    previous = get_volume_store()
    store = VolumeStore(tmp_path / "models-volumes.toml")
    set_volume_store(store)
    store.ensure(MODELS_VOLUME, label="Models", description="", location=tmp_path / "models")
    store.ensure(HF_VOLUME, label="Hugging Face cache", description="", location=tmp_path / "hf")
    (tmp_path / "models").mkdir()
    (tmp_path / "hf").mkdir()
    yield store
    set_volume_store(previous)


def _save_compact(path: Path, **kwargs: Any) -> Path:
    import torch
    from spandrel.architectures.Compact import Compact

    torch.manual_seed(0)
    model = Compact(num_feat=8, num_conv=2, **kwargs)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(model.state_dict(), str(path))
    else:
        torch.save(model.state_dict(), path)
    return path


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A session folder holding the generated model files."""
    return tmp_path_factory.mktemp("spandrel-models")


@pytest.fixture(scope="session")
def tiny_sr_path(model_dir: Path) -> Path:
    """A 2x Compact (RGB → RGB) as ``.safetensors``."""
    return _save_compact(model_dir / "tiny-2x.safetensors", upscale=2)


@pytest.fixture(scope="session")
def tiny_sr4_path(model_dir: Path) -> Path:
    """A 4x Compact (RGB → RGB) as ``.pth``."""
    return _save_compact(model_dir / "tiny-4x.pth", upscale=4)


@pytest.fixture(scope="session")
def tiny_1x_path(model_dir: Path) -> Path:
    """A 1x Compact (restoration-shaped) as ``.safetensors``."""
    return _save_compact(model_dir / "tiny-1x.safetensors", upscale=1)


@pytest.fixture(scope="session")
def tiny_gray_path(model_dir: Path) -> Path:
    """A 2x Compact with one input and one output channel."""
    return _save_compact(model_dir / "tiny-gray-2x.safetensors", upscale=2, num_in_ch=1, num_out_ch=1)


@pytest.fixture(scope="session")
def lama_path(model_dir: Path) -> Path:
    """A random-weight LaMa (about 200 MB) as ``.safetensors``."""
    import torch
    from safetensors.torch import save_file
    from spandrel.architectures.LaMa import LaMa

    torch.manual_seed(0)
    path = model_dir / "lama.safetensors"
    # in_nc counts the mask channel, as in the published big-lama weights.
    save_file({k: v.contiguous() for k, v in LaMa(in_nc=4).state_dict().items()}, str(path))
    return path


@pytest.fixture(scope="session")
def srformer_path(model_dir: Path) -> Path:
    """A tiny SRFormer (non-commercial architecture, 1x restoration) as ``.safetensors``."""
    import importlib

    import torch
    from safetensors.torch import save_file

    # The package folder is named ``__arch``, which ``from ... import`` cannot spell.
    arch = importlib.import_module("spandrel_extra_arches.architectures.SRFormer.__arch.SRFormer")
    torch.manual_seed(0)
    path = model_dir / "srformer.safetensors"
    model = arch.SRFormer(embed_dim=12, depths=[1], num_heads=[1])
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    return path


def add_node(graph: Any, module: str, name: str) -> Any:
    """Place the node class *name* from *module* in *graph* and return the node."""
    import importlib

    cls = getattr(importlib.import_module(module), name)
    wrapper = graph.create_node_wrapper(cls.class_identity.registry_key, position=(0, 0))
    assert wrapper is not None, f"node creation failed for {name}"
    assert isinstance(wrapper.node, cls), "an unknown key yields an ErrorNode"
    return wrapper.node


def add_loader(
    graph: Any,
    feeds: tuple[str, str] = ("haybale_spandrel.nodes.spandrel_model_info", "SpandrelModelInfoNode"),
) -> Any:
    """Place a Load Model in *graph* with its outlet wired to a new *feeds* node, which sets its kind."""
    load = add_node(graph, "haybale_spandrel.nodes.load_spandrel_model", "LoadSpandrelModelNode")
    consumer = add_node(graph, *feeds)
    graph.create_edge_wrapper(load.node_id, "loaded", consumer.node_id, "model")
    return load


def run_from(graph: Any, entry: Any) -> None:
    """Assemble *graph* and run the flow that starts at the event node *entry*, startup to shutdown."""
    import threading

    from haywire.core.assembly.flow_assembly_manager import FlowAssemblyManager
    from haywire.core.execution.event_source import Trigger
    from haywire.core.execution.vm import HaywireVM

    graph.force_validation()
    flows = FlowAssemblyManager().assemble_graph(graph)
    flow = next(f for f in flows if f.get_entry_node_id() == entry.node_id)
    trigger = Trigger(source_key=flow.get_subscription_key(), payload={}, timestamp=0.0)
    vm = HaywireVM()
    nodes = list(flow.get_all_nodes())

    def run() -> None:
        vm.call_startup(nodes, flow)
        try:
            vm.execute_control_flow(flow, trigger)
        finally:
            vm.call_shutdown(nodes, flow)

    runner = threading.Thread(target=run)
    runner.start()
    runner.join()


def place_model(source: Path, name: str | None = None) -> Any:
    """Copy *source* into the Models volume, index it, and return a pinned ``ModelRef``."""
    import hashlib
    import shutil

    from haybale_models.consumer import models_state
    from haybale_models.keys import MODELS_VOLUME
    from haybale_models.value.model_ref import ModelRef
    from haywire.api import volume_location

    target = volume_location(MODELS_VOLUME) / (name or source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    state = models_state()
    state.index.scan()
    state.index.hash_pending()
    data = target.read_bytes()
    return ModelRef(sha256=hashlib.sha256(data).hexdigest(), size=len(data), label=target.name)


@pytest.fixture(autouse=True)
def _empty_pool(request: pytest.FixtureRequest) -> Iterator[None]:
    """Release every pooled model after a test that loaded the library system."""
    yield
    if "library_system" in request.fixturenames:
        from haybale_models.consumer import models_state

        models_state().pool.release_all()
