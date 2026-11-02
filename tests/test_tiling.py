"""Tile size choice and tiled execution."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from spandrel import ModelTiling, SizeRequirements

from haybale_spandrel.runtime.loader import load_model
from haybale_spandrel.runtime.tiling import (
    MAX_ELEMENTS,
    MIN_TILE,
    OUTPUT_FEATURES,
    Cancelled,
    choose_tile,
    run_tiled,
)

CPU = torch.device("cpu")
GIB = 1024**3


class _Fake:
    """Stands in for a SpandrelModel where only tiling facts matter."""

    def __init__(
        self, tiling: ModelTiling = ModelTiling.SUPPORTED, req: SizeRequirements | None = None
    ) -> None:
        self.tiling = tiling
        self.size_requirements = req or SizeRequirements()
        self.dtype = torch.float32
        self.input_channels = 3
        self.output_channels = 3
        self.scale = 4


def _choose(model: object, w: int, h: int, tile: int | None = None, budget: int = GIB) -> int | None:
    return choose_tile(model, w, h, tile_setting=tile, budget_bytes=budget)  # type: ignore[arg-type]


def test_auto_runs_a_small_image_whole() -> None:
    assert _choose(_Fake(), 64, 48) is None


def test_auto_tiles_a_huge_image_within_the_budget() -> None:
    side = _choose(_Fake(req=SizeRequirements(multiple_of=8)), 20000, 20000)
    assert side is not None
    assert side % 8 == 0
    assert side >= MIN_TILE
    assert side < _choose(_Fake(), 20000, 20000, budget=16 * GIB)  # type: ignore[operator]


def test_discouraged_is_whole_unless_a_size_is_set() -> None:
    model = _Fake(ModelTiling.DISCOURAGED)
    assert _choose(model, 20000, 20000) is None
    assert _choose(model, 20000, 20000, tile=128) == 128


def test_internal_never_tiles() -> None:
    assert _choose(_Fake(ModelTiling.INTERNAL), 20000, 20000, tile=128) is None


def test_none_setting_never_tiles() -> None:
    assert _choose(_Fake(), 20000, 20000, tile=0) is None


def test_explicit_size_respects_the_requirements() -> None:
    assert _choose(_Fake(req=SizeRequirements(minimum=64, multiple_of=8)), 1000, 1000, tile=20) == 64
    assert _choose(_Fake(req=SizeRequirements(multiple_of=8)), 1000, 1000, tile=100) == 96


def _nearest(t: torch.Tensor) -> torch.Tensor:
    return F.interpolate(t, scale_factor=2, mode="nearest")


def test_tiles_equal_whole_for_a_pointwise_model() -> None:
    image = torch.rand(1, 3, 37, 53)
    whole = run_tiled(_nearest, image, scale=2, tile=None)
    tiled = run_tiled(_nearest, image, scale=2, tile=16)
    assert tiled.shape == (1, 3, 74, 106)
    torch.testing.assert_close(tiled, whole)


def test_tiles_match_whole_for_a_real_model(tiny_sr_path: Path) -> None:
    model = load_model(tiny_sr_path, CPU, "fp32")
    image = torch.rand(1, 3, 37, 53)
    whole = run_tiled(model.descriptor, image, scale=2, tile=None)
    tiled = run_tiled(model.descriptor, image, scale=2, tile=16)
    torch.testing.assert_close(tiled, whole, atol=1e-3, rtol=0)


def test_image_below_the_minimum_comes_back_at_its_own_size(tiny_sr_path: Path) -> None:
    model = load_model(tiny_sr_path, CPU, "fp32")
    model.descriptor.size_requirements = SizeRequirements(minimum=64, multiple_of=8)
    out = run_tiled(model.descriptor, torch.rand(1, 3, 21, 13), scale=2, tile=None)
    assert out.shape == (1, 3, 42, 26)


def test_cancel_between_tiles() -> None:
    cancel = threading.Event()
    seen: list[tuple[int, int]] = []

    def on_tile(done: int, total: int) -> None:
        seen.append((done, total))
        cancel.set()

    with pytest.raises(Cancelled):
        run_tiled(_nearest, torch.rand(1, 3, 40, 40), scale=2, tile=16, cancel=cancel, on_tile=on_tile)
    assert seen == [(1, 9)]


def test_out_of_memory_halves_the_tile() -> None:
    sides: list[int] = []

    def forward(t: torch.Tensor) -> torch.Tensor:
        sides.append(max(t.shape[2], t.shape[3]))
        if max(t.shape[2], t.shape[3]) > 80:
            raise torch.OutOfMemoryError("out of memory")
        return _nearest(t)

    image = torch.rand(1, 3, 128, 128)
    out = run_tiled(forward, image, scale=2, tile=None)
    torch.testing.assert_close(out, _nearest(image))
    assert sides[0] == 128
    assert max(sides[1:]) <= 80


def test_out_of_memory_below_the_floor_raises() -> None:
    def forward(t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("MPS backend out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        run_tiled(forward, torch.rand(1, 3, 64, 64), scale=2, tile=None)


def test_other_errors_pass_through() -> None:
    def forward(t: torch.Tensor) -> torch.Tensor:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_tiled(forward, torch.rand(1, 3, 64, 64), scale=2, tile=None)


def test_auto_keeps_output_tensors_below_the_element_limit() -> None:
    # 2048×1358 through a 4x model: whole, the last layers would hold 8192×5432×64 elements.
    side = _choose(_Fake(), 2048, 1358, budget=64 * GIB)
    assert side is not None
    assert (side * 4) ** 2 * OUTPUT_FEATURES <= MAX_ELEMENTS


def test_a_convolution_too_large_for_the_device_halves_the_tile() -> None:
    sides: list[int] = []

    def forward(t: torch.Tensor) -> torch.Tensor:
        sides.append(max(t.shape[2], t.shape[3]))
        if max(t.shape[2], t.shape[3]) > 80:
            raise NotImplementedError("convolution_overrideable not implemented")
        return _nearest(t)

    image = torch.rand(1, 3, 128, 128)
    torch.testing.assert_close(run_tiled(forward, image, scale=2, tile=None), _nearest(image))
    assert max(sides[1:]) <= 80
