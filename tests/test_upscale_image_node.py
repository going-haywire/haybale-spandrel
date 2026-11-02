"""Upscale Image and Inpaint nodes in graph runs."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from haybale_image.value import ImageValue, Layout
from haywire.core.volumes import VolumeStore

from tests.conftest import add_node, place_model, run_from

pytestmark = pytest.mark.integration

LOAD = ("haybale_spandrel.nodes.load_model", "LoadModelNode")
UPSCALE = ("haybale_spandrel.nodes.upscale_image", "UpscaleImageNode")
INPAINT = ("haybale_spandrel.nodes.inpaint", "InpaintNode")
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def _image(h: int, w: int, channels: int = 3) -> ImageValue:
    shape = (h, w) if channels == 1 else (h, w, channels)
    return ImageValue.from_array(np.random.default_rng(0).random(shape, dtype=np.float32))


def _upscale_graph(graph: Any, model_path: Path) -> tuple[Any, Any, Any]:
    """BeginPlay → Print, Print's message ← Image Info width ← Upscale ← Load Model."""
    begin = add_node(graph, "haybale_core.nodes.events.begin_play", "BeginPlayNode")
    printer = add_node(graph, "haybale_core.nodes.print", "PrintNode")
    info = add_node(graph, "haybale_image.nodes.image_info", "ImageInfoNode")
    load = add_node(graph, *LOAD)
    upscale = add_node(graph, *UPSCALE)
    graph.create_edge_wrapper(begin.node_id, "exec", printer.node_id, "exec")
    graph.create_edge_wrapper(load.node_id, "loaded", upscale.node_id, "model")
    graph.create_edge_wrapper(upscale.node_id, "out_image", info.node_id, "image")
    graph.create_edge_wrapper(info.node_id, "width", printer.node_id, "message")
    load.ports["model"].set_value(place_model(model_path))
    upscale.ports["image"].set_value(_image(20, 30))
    return begin, upscale, info


def test_upscale_in_a_run(graph: Any, models_env: VolumeStore, tiny_sr_path: Path) -> None:
    begin, upscale, info = _upscale_graph(graph, tiny_sr_path)
    run_from(graph, begin)
    assert (info.ports["width"].get_value(), info.ports["height"].get_value()) == (60, 40)


def test_scale_three(graph: Any, models_env: VolumeStore, tiny_sr_path: Path) -> None:
    begin, upscale, info = _upscale_graph(graph, tiny_sr_path)
    upscale.ports["scale"].set_value(3.0)
    run_from(graph, begin)
    assert (info.ports["width"].get_value(), info.ports["height"].get_value()) == (90, 60)


def test_tiled_equals_whole(graph: Any, models_env: VolumeStore, tiny_sr_path: Path) -> None:
    begin, upscale, _info = _upscale_graph(graph, tiny_sr_path)
    upscale.ports["image"].set_value(_image(37, 53))
    upscale.ports["tile_size"].set_value("none")
    run_from(graph, begin)
    whole = upscale.ports["out_image"].get_value().tensor(CPU)
    upscale.ports["tile_size"].set_value("16")
    run_from(graph, begin)
    tiled = upscale.ports["out_image"].get_value().tensor(CPU)
    torch.testing.assert_close(tiled, whole, atol=1e-3, rtol=0)


def _with_model(graph: Any, model_path: Path) -> tuple[Any, Any]:
    load = add_node(graph, *LOAD)
    load.ports["model"].set_value(place_model(model_path))
    load._execute(MagicMock())
    upscale = add_node(graph, *UPSCALE)
    upscale.ports["model"].set_value(load.ports["loaded"].get_value())
    upscale.ports["image"].set_value(_image(20, 30))
    return load, upscale


def test_bad_tile_size_is_an_error_naming_it(
    graph: Any, models_env: VolumeStore, tiny_sr_path: Path
) -> None:
    _load, upscale = _with_model(graph, tiny_sr_path)
    upscale.ports["tile_size"].set_value("banana")
    with pytest.raises(ValueError, match="banana"):
        upscale._execute(MagicMock(cancel=None))


def test_empty_image_is_an_error(graph: Any, models_env: VolumeStore, tiny_sr_path: Path) -> None:
    _load, upscale = _with_model(graph, tiny_sr_path)
    upscale.ports["image"].set_value(ImageValue.empty(Layout.RGB))
    with pytest.raises(ValueError, match="empty image"):
        upscale._execute(MagicMock(cancel=None))


def test_no_model_is_an_error(graph: Any) -> None:
    upscale = add_node(graph, *UPSCALE)
    upscale.ports["image"].set_value(_image(4, 4))
    with pytest.raises(ValueError, match="no model"):
        upscale._execute(MagicMock(cancel=None))


def test_cancel_leaves_the_outlet_alone(graph: Any, models_env: VolumeStore, tiny_sr_path: Path) -> None:
    _load, upscale = _with_model(graph, tiny_sr_path)
    upscale.ports["tile_size"].set_value("16")
    before = upscale.ports["out_image"].get_value()  # the outlet's default
    cancel = threading.Event()
    cancel.set()
    upscale._execute(MagicMock(cancel=cancel))
    assert upscale.ports["out_image"].get_value() is before


def test_colour_into_gray_model_warns(graph: Any, models_env: VolumeStore, tiny_gray_path: Path) -> None:
    _load, upscale = _with_model(graph, tiny_gray_path)
    upscale._execute(MagicMock(cancel=None))
    assert upscale.ports["out_image"].get_value().layout == Layout.GRAY
    messages = [w.message for w in upscale.wrapper.state.warnings if w.kind == "spandrel"]
    assert messages == ["model is grayscale — colour is dropped"]


@pytest.mark.slow
def test_inpaint_node(graph: Any, models_env: VolumeStore, lama_path: Path) -> None:
    load = add_node(graph, *LOAD)
    load.ports["model"].set_value(place_model(lama_path))
    load._execute(MagicMock())
    inpaint = add_node(graph, *INPAINT)
    inpaint.ports["model"].set_value(load.ports["loaded"].get_value())
    inpaint.ports["image"].set_value(_image(48, 64))
    inpaint.ports["mask"].set_value(_image(48, 64, channels=1))
    inpaint._execute(MagicMock(cancel=None))
    out = inpaint.ports["out_image"].get_value()
    assert (out.height, out.width) == (48, 64)


def test_inpaint_with_an_sr_model_is_an_error(
    graph: Any, models_env: VolumeStore, tiny_sr_path: Path
) -> None:
    load, _upscale = _with_model(graph, tiny_sr_path)
    inpaint = add_node(graph, *INPAINT)
    inpaint.ports["model"].set_value(load.ports["loaded"].get_value())
    inpaint.ports["image"].set_value(_image(8, 8))
    inpaint.ports["mask"].set_value(_image(8, 8, channels=1))
    with pytest.raises(ValueError, match="not an inpainting model"):
        inpaint._execute(MagicMock(cancel=None))
