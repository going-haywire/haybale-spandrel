"""Load Model and Model Info: graph runs, pool holds, device notes, release."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from haywire.core.volumes import VolumeStore

from tests.conftest import add_node, place_model, run_from

pytestmark = pytest.mark.integration

LOAD = ("haybale_spandrel.nodes.load_model", "LoadModelNode")
INFO = ("haybale_spandrel.nodes.model_info", "ModelInfoNode")


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def _pool() -> Any:
    from haybale_models.consumer import models_state

    return models_state().pool


def _graph_with_info(graph: Any) -> tuple[Any, Any, Any]:
    begin = add_node(graph, "haybale_core.nodes.events.begin_play", "BeginPlayNode")
    printer = add_node(graph, "haybale_core.nodes.print", "PrintNode")
    load = add_node(graph, *LOAD)
    info = add_node(graph, *INFO)
    graph.create_edge_wrapper(begin.node_id, "exec", printer.node_id, "exec")
    graph.create_edge_wrapper(load.node_id, "loaded", info.node_id, "model")
    graph.create_edge_wrapper(info.node_id, "architecture", printer.node_id, "message")
    return begin, load, info


@pytest.mark.usefixtures("cpu_only")
def test_model_info_after_a_run(graph: Any, models_env: VolumeStore, tiny_sr_path: Path) -> None:
    begin, load, info = _graph_with_info(graph)
    load.ports["model"].set_value(place_model(tiny_sr_path))
    run_from(graph, begin)

    out = {pid: info.ports[pid].get_value() for pid in ("architecture", "scale", "device", "precision")}
    assert out == {"architecture": "RealESRGAN Compact", "scale": 2, "device": "cpu", "precision": "fp32"}
    assert info.ports["purpose"].get_value() == "SR"
    assert info.ports["parameters"].get_value() > 0
    assert _pool().entries() == []  # released when the run ended


@pytest.mark.usefixtures("cpu_only")
def test_a_second_run_loads_again(
    graph: Any, models_env: VolumeStore, tiny_sr_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("haybale_spandrel.nodes.load_model")
    loads: list[Path] = []
    real = mod.load_model

    def counting(path: Path, *args: Any, **kwargs: Any) -> Any:
        loads.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(mod, "load_model", counting)
    begin, load, info = _graph_with_info(graph)
    load.ports["model"].set_value(place_model(tiny_sr_path))
    run_from(graph, begin)
    assert load.ports["loaded"].get_value() is None  # the wire lets go when the run ends
    info.ports["scale"].set_value(0)
    run_from(graph, begin)
    assert len(loads) == 2
    assert info.ports["scale"].get_value() == 2


@pytest.mark.usefixtures("cpu_only")
def test_same_model_and_settings_share_one_load(
    graph: Any, models_env: VolumeStore, tiny_sr_path: Path
) -> None:
    ref = place_model(tiny_sr_path)
    a, b = add_node(graph, *LOAD), add_node(graph, *LOAD)
    for n in (a, b):
        n.ports["model"].set_value(ref)
        n._execute(MagicMock())
    assert a.ports["loaded"].get_value() is b.ports["loaded"].get_value()
    assert [e.holders for e in _pool().entries()] == [2]

    graph.remove_node_wrapper(a.wrapper)
    a.wrapper.cleanup()  # what dropping the node for good, or a rebuild on reload, does
    assert [e.holders for e in _pool().entries()] == [1]
    b.on_shutdown(MagicMock())
    assert _pool().entries() == []


@pytest.mark.usefixtures("cpu_only")
def test_changing_the_model_releases_the_old_one(
    graph: Any, models_env: VolumeStore, tiny_sr_path: Path, tiny_sr4_path: Path
) -> None:
    load = add_node(graph, *LOAD)
    load.ports["model"].set_value(place_model(tiny_sr_path))
    load._execute(MagicMock())
    first_key = _pool().entries()[0].key

    load.ports["model"].set_value(place_model(tiny_sr4_path))
    load._execute(MagicMock())
    keys = [e.key for e in _pool().entries()]
    assert first_key not in keys
    assert len(keys) == 1
    assert load.ports["loaded"].get_value().scale == 4
    load.on_shutdown(MagicMock())


@pytest.mark.usefixtures("cpu_only")
def test_unavailable_device_runs_on_cpu_with_a_warning(
    graph: Any, models_env: VolumeStore, tiny_sr_path: Path
) -> None:
    load = add_node(graph, *LOAD)
    load.settings.device = "cuda:1"
    load.ports["model"].set_value(place_model(tiny_sr_path))
    load._execute(MagicMock())
    assert str(load.ports["loaded"].get_value().device) == "cpu"
    messages = [w.message for w in load.wrapper.state.warnings if w.kind == "spandrel"]
    assert messages == ["cuda:1 not available — running on cpu"]
    assert load.status.loaded == "RealESRGAN Compact · 2x · RGB · cpu fp32"
    load.on_shutdown(MagicMock())
    assert load.status.loaded == importlib.import_module(LOAD[0]).NOT_LOADED


def test_missing_model_is_an_error_with_a_warning(graph: Any, models_env: VolumeStore) -> None:
    from haybale_models.consumer import ModelMissing
    from haybale_models.value.model_ref import ModelRef

    load = add_node(graph, *LOAD)
    load.ports["model"].set_value(ModelRef(sha256="0" * 64, size=1))
    with pytest.raises(ModelMissing):
        load._execute(MagicMock())
    assert any(w.kind == "model" for w in load.wrapper.state.warnings)


def test_model_info_without_a_model_is_an_error(graph: Any) -> None:
    info = add_node(graph, *INFO)
    with pytest.raises(ValueError, match="no model"):
        info._execute(MagicMock())


@pytest.mark.usefixtures("cpu_only")
def test_a_released_model_still_runs_for_whoever_holds_it(
    graph: Any, models_env: VolumeStore, tiny_sr_path: Path
) -> None:
    import numpy as np
    from haybale_image.value import ImageValue

    from haybale_spandrel.runtime.upscale import upscale_image

    load = add_node(graph, *LOAD)
    load.ports["model"].set_value(place_model(tiny_sr_path))
    load._execute(MagicMock())
    held = load.ports["loaded"].get_value()
    load.on_shutdown(MagicMock())
    assert _pool().entries() == []
    assert load.ports["loaded"].get_value() is None

    out, _ = upscale_image(ImageValue.from_array(np.zeros((8, 8, 3), np.float32)), held, budget_bytes=2**30)
    assert out.width == 16


@pytest.mark.usefixtures("cpu_only")
def test_changing_precision_loads_the_model_again(
    graph: Any, models_env: VolumeStore, tiny_sr_path: Path
) -> None:
    load = add_node(graph, *LOAD)
    load.ports["model"].set_value(place_model(tiny_sr_path))
    load._execute(MagicMock())
    assert load.ports["loaded"].get_value().dtype == torch.float32

    load.settings.precision = "fp16"
    load._execute(MagicMock())  # no port is dirty: only the setting changed
    assert load.ports["loaded"].get_value().dtype == torch.float16
    assert load.status.loaded.endswith("cpu fp16")
    assert [e.key[1] for e in _pool().entries()] == [("cpu", "fp16")]
