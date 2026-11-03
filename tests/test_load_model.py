"""Load Model and Model Info: graph runs, pool holds, device notes, release."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from haywire.core.volumes import VolumeStore

from tests.conftest import add_loader, add_node, place_model, run_from

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
    a, b = add_loader(graph), add_loader(graph)
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
    load = add_loader(graph)
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
    load = add_loader(graph)
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

    load = add_loader(graph)
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

    load = add_loader(graph)
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
    load = add_loader(graph)
    load.ports["model"].set_value(place_model(tiny_sr_path))
    load._execute(MagicMock())
    assert load.ports["loaded"].get_value().dtype == torch.float32

    load.settings.precision = "fp16"
    load._execute(MagicMock())  # no port is dirty: only the setting changed
    assert load.ports["loaded"].get_value().dtype == torch.float16
    assert load.status.loaded.endswith("cpu fp16")
    assert [e.key[1] for e in _pool().entries()] == [("cpu", "fp16", False)]


# ---------------------------------------------------------------- non-commercial architectures


@pytest.fixture
def noncommercial(library_system: Any) -> Any:
    """Set the library's *Non-commercial architectures*: call with True/False; reset to off afterwards."""
    settings = importlib.import_module("haybale_spandrel.settings.spandrel_settings").SpandrelSettings

    def set_to(on: bool) -> None:
        settings().noncommercial_architectures = on
        assert settings().noncommercial_architectures is on  # a dropped write fails loudly

    yield set_to
    set_to(False)


@pytest.mark.usefixtures("cpu_only")
def test_toggle_changes_pool_key(
    graph: Any, models_env: VolumeStore, tiny_sr_path: Path, noncommercial: Any
) -> None:
    load = add_loader(graph)
    load.ports["model"].set_value(place_model(tiny_sr_path))
    load._execute(MagicMock())
    key_off = load.cache.key
    load.on_shutdown(MagicMock())  # the setting applies from the next graph start
    noncommercial(True)
    load._execute(MagicMock())
    key_on = load.cache.key
    assert key_off[1][2] is False
    assert key_on[1][2] is True
    assert key_off[0] == key_on[0]
    assert [e.key for e in _pool().entries()] == [key_on]
    load.on_shutdown(MagicMock())


@pytest.mark.usefixtures("cpu_only")
def test_failed_model_loads_after_enabling(
    graph: Any, models_env: VolumeStore, srformer_path: Path, noncommercial: Any
) -> None:
    import spandrel

    load = add_loader(graph)
    load.ports["model"].set_value(place_model(srformer_path))
    with pytest.raises(spandrel.UnsupportedModelError, match="Non-commercial architectures"):
        load._execute(MagicMock())
    noncommercial(True)
    load._execute(MagicMock())  # no graph edit: only the setting changed
    assert load.status.loaded.endswith(" · non-commercial")
    messages = [w.message for w in load.wrapper.state.warnings if w.kind == "spandrel"]
    assert messages == ["SRFormer is non-commercial only (CC BY-NC 4.0)"]
    load.on_shutdown(MagicMock())


@pytest.mark.usefixtures("cpu_only")
def test_model_info_reports_noncommercial(
    graph: Any, models_env: VolumeStore, srformer_path: Path, tiny_sr_path: Path, noncommercial: Any
) -> None:
    noncommercial(True)
    for path, expected in ((srformer_path, True), (tiny_sr_path, False)):
        begin, load, info = _graph_with_info(graph)
        load.ports["model"].set_value(place_model(path))
        run_from(graph, begin)
        assert info.ports["noncommercial"].get_value() is expected


# ---------------------------------------------------------------- the outlet takes its kind


UPSCALE = ("haybale_spandrel.nodes.upscale_image", "UpscaleImageNode")
INPAINT = ("haybale_spandrel.nodes.inpaint", "InpaintNode")


def _kind(load: Any) -> str:
    return load.ports["loaded"].type_cls.__name__


def _picker(load: Any) -> dict[str, Any]:
    return load.ports["model"].widget_config["properties"]


def test_a_new_loader_has_an_open_outlet_and_no_picker(graph: Any) -> None:
    load = add_node(graph, *LOAD)

    assert _kind(load) == "ADD"
    assert "model" not in load.ports


def test_the_outlet_takes_the_kind_its_consumer_needs(graph: Any) -> None:
    load = add_loader(graph, feeds=UPSCALE)

    assert _kind(load) == "IMAGE_MODEL"
    assert _picker(load)["filter"] == {"purpose": ["SR", "FaceSR", "Restoration"]}


def test_model_info_takes_any_kind_and_filters_nothing(graph: Any) -> None:
    load = add_loader(graph)

    assert _kind(load) == "MODEL_BASE"
    assert _picker(load)["filter"] == {}


def test_the_outlet_keeps_its_kind_when_unwired_and_rewired(graph: Any) -> None:
    load = add_loader(graph, feeds=UPSCALE)
    (edge_id,) = list(load.ports["loaded"].outgoing.linked)
    graph.remove_edge_wrapper(edge_id)
    inpaint = add_node(graph, *INPAINT)

    graph.create_edge_wrapper(load.node_id, "loaded", inpaint.node_id, "model")

    assert _kind(load) == "IMAGE_MODEL"


def test_a_non_model_inlet_leaves_an_invalid_edge(graph: Any) -> None:
    load = add_node(graph, *LOAD)
    printer = add_node(graph, "haybale_core.nodes.print", "PrintNode")

    edge = graph.create_edge_wrapper(load.node_id, "loaded", printer.node_id, "message")
    graph.force_validation()

    assert _kind(load) == "MODEL_BASE"
    assert not edge.is_valid()


@pytest.mark.usefixtures("cpu_only")
def test_a_model_of_another_kind_is_an_error_when_it_loads(
    graph: Any, models_env: VolumeStore, tiny_sr_path: Path
) -> None:
    load = add_loader(graph, feeds=INPAINT)
    load.ports["model"].set_value(place_model(tiny_sr_path))

    with pytest.raises(ValueError, match=r"does not fit this outlet \(inpainting model\)"):
        load.worker(MagicMock(), load.ports["model"].get_value())

    assert _pool().entries() == []


def test_wiring_the_outlet_into_an_image_inlet_is_invalid_in_the_studio_scheduler(
    library_system: Any,
) -> None:
    """The studio validates on a timer; the edge must still be rebuilt for the outlet's new kind."""
    from haywire.core.graph.base import BaseGraph
    from haywire.core.graph.scheduler import ThreadingTimerScheduler

    graph = BaseGraph(
        filestem="g", validation_scheduler=ThreadingTimerScheduler(), validation_delay_ms=60_000
    )
    try:
        load = add_node(graph, *LOAD)
        upscale = add_node(graph, *UPSCALE)
        graph._validation.force_immediate_validation()

        to_image = graph.create_edge_wrapper(load.node_id, "loaded", upscale.node_id, "image")
        graph._validation.force_immediate_validation()

        assert _kind(load) == "MODEL_BASE"
        assert not to_image.is_valid()
    finally:
        graph._validation.clear()
        graph.cleanup()
