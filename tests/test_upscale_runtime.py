"""upscale_image: scale, layout, alpha, notes and errors on tiny CPU models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from haybale_image.value import ImageValue, Layout
from haywire.api import FLOAT

from haybale_spandrel.runtime.loader import load_model
from haybale_spandrel.runtime.upscale import upscale_image
from haybale_spandrel.value.spandrel_model import SpandrelModel

CPU = torch.device("cpu")
GIB = 1024**3


def _image(h: int, w: int, channels: int, depth: int = 8) -> ImageValue:
    rng = np.random.default_rng(0)
    shape = (h, w) if channels == 1 else (h, w, channels)
    data = rng.random(shape, dtype=np.float32)
    return ImageValue.from_array(data, source_depth=depth)


def _run(image: ImageValue, model: SpandrelModel, **kwargs: object) -> tuple[ImageValue, list[str]]:
    return upscale_image(image, model, budget_bytes=GIB, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def sr2(tiny_sr_path: Path) -> SpandrelModel:
    return load_model(tiny_sr_path, CPU, "fp32")


def test_native_scale(sr2: SpandrelModel) -> None:
    out, notes = _run(_image(20, 30, 3), sr2)
    assert (out.height, out.width, out.layout) == (40, 60, Layout.RGB)
    assert out.residency == "cpu"
    assert notes == []


def test_fields_pass_through(sr2: SpandrelModel) -> None:
    stamped = _image(20, 30, 3).stamped("camera_timestamp", 1.5, FLOAT)
    out, _ = _run(stamped, sr2)
    assert out.fields["camera_timestamp"].value == 1.5


def test_scale_above_native_runs_twice_then_downsizes(sr2: SpandrelModel) -> None:
    calls: list[tuple[int, ...]] = []
    original = sr2.descriptor.__class__.__call__

    def spy(self: object, t: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(t.shape))
        return original(self, t)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sr2.descriptor.__class__, "__call__", spy)
        out, _ = _run(_image(20, 30, 3), sr2, scale=3)
    assert (out.height, out.width) == (60, 90)
    # The second pass starts from a shrunk image so it lands on 3x, never on 4x.
    assert calls == [(1, 3, 20, 30), (1, 3, 30, 45)]


def test_no_pass_outgrows_the_result(tiny_sr4_path: Path) -> None:
    model = load_model(tiny_sr4_path, CPU, "fp32")
    outputs: list[tuple[int, int]] = []
    original = model.descriptor.__class__.__call__

    def spy(self: object, t: torch.Tensor) -> torch.Tensor:
        out = original(self, t)  # type: ignore[arg-type]
        outputs.append((int(out.shape[2]), int(out.shape[3])))
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(model.descriptor.__class__, "__call__", spy)
        out, _ = _run(_image(20, 30, 3), model, scale=8)
    assert (out.height, out.width) == (160, 240)
    assert outputs == [(80, 120), (160, 240)]


def test_fractional_scale(sr2: SpandrelModel) -> None:
    out, _ = _run(_image(20, 30, 3), sr2, scale=1.5)
    assert (out.height, out.width) == (30, 45)


def test_rgba_alpha_resize_is_bicubic_of_the_input_alpha(sr2: SpandrelModel) -> None:
    image = _image(20, 30, 4)
    out, _ = _run(image, sr2)
    assert out.layout == Layout.RGBA
    expected = F.interpolate(
        image.tensor(CPU)[:, 3:4], size=(40, 60), mode="bicubic", antialias=True, align_corners=False
    ).clamp(0, 1)
    torch.testing.assert_close(out.tensor(CPU)[:, 3:4], expected)


def test_rgba_alpha_model_runs_the_model_twice(sr2: SpandrelModel) -> None:
    calls: list[int] = []
    original = sr2.descriptor.__class__.__call__

    def spy(self: object, t: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return original(self, t)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sr2.descriptor.__class__, "__call__", spy)
        out, _ = _run(_image(20, 30, 4), sr2, alpha="model")
    assert out.layout == Layout.RGBA
    assert len(calls) == 2


def test_gray_into_colour_model_stays_gray(sr2: SpandrelModel) -> None:
    out, notes = _run(_image(20, 30, 1), sr2)
    assert out.layout == Layout.GRAY
    assert notes == []


def test_colour_into_gray_model_is_gray_with_a_note(tiny_gray_path: Path) -> None:
    out, notes = _run(_image(20, 30, 3), load_model(tiny_gray_path, CPU, "fp32"))
    assert out.layout == Layout.GRAY
    assert notes == ["model is grayscale — colour is dropped"]


def test_restoration_model_keeps_the_size(tiny_1x_path: Path) -> None:
    out, _ = _run(_image(20, 30, 3), load_model(tiny_1x_path, CPU, "fp32"))
    assert (out.height, out.width) == (20, 30)


def test_source_depth_is_kept(sr2: SpandrelModel) -> None:
    assert _run(_image(20, 30, 3, depth=16), sr2)[0].source_depth == 16


def test_half_model_gives_a_half_tensor(tiny_sr_path: Path) -> None:
    out, _ = _run(_image(8, 8, 3), load_model(tiny_sr_path, CPU, "fp16"))
    assert out.tensor(CPU, torch.float16).dtype == torch.float16


def test_tiled_run_matches_whole(sr2: SpandrelModel) -> None:
    image = _image(37, 53, 3)
    whole, _ = _run(image, sr2, tile_setting=0)
    tiled, _ = _run(image, sr2, tile_setting=16)
    torch.testing.assert_close(tiled.tensor(CPU), whole.tensor(CPU), atol=1e-3, rtol=0)


def test_empty_image_is_refused(sr2: SpandrelModel) -> None:
    with pytest.raises(ValueError, match="empty image"):
        _run(ImageValue.empty(Layout.RGB), sr2)


def test_unknown_alpha_mode_is_refused(sr2: SpandrelModel) -> None:
    with pytest.raises(ValueError, match="alpha"):
        _run(_image(4, 4, 4), sr2, alpha="blend")


def test_forward_runs_under_the_model_lock(sr2: SpandrelModel) -> None:
    held: list[bool] = []
    original = sr2.descriptor.__class__.__call__

    def spy(self: object, t: torch.Tensor) -> torch.Tensor:
        held.append(sr2.lock.locked())
        return original(self, t)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sr2.descriptor.__class__, "__call__", spy)
        _run(_image(8, 8, 3), sr2)
    assert held == [True]


@pytest.mark.slow
def test_inpainting_model_is_refused(lama_path: Path) -> None:
    with pytest.raises(ValueError, match="inpainting model"):
        _run(_image(8, 8, 3), load_model(lama_path, CPU, "fp32"))
