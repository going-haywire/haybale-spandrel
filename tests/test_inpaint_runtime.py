"""inpaint_image on a random-weight LaMa."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from haybale_image.value import ImageValue, Layout

from haybale_spandrel.runtime.inpaint import inpaint_image
from haybale_spandrel.runtime.loader import load_model
from haybale_spandrel.value.spandrel_model import SpandrelModel

pytestmark = pytest.mark.slow

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def lama(lama_path: Path) -> SpandrelModel:
    return load_model(lama_path, CPU, "fp32")


def _image(h: int, w: int, channels: int) -> ImageValue:
    shape = (h, w) if channels == 1 else (h, w, channels)
    return ImageValue.from_array(np.random.default_rng(0).random(shape, dtype=np.float32))


def _mask(h: int, w: int) -> ImageValue:
    m = np.zeros((h, w), np.float32)
    m[h // 4 : h // 2, w // 4 : w // 2] = 1
    return ImageValue.from_array(m)


def test_fills_and_keeps_the_size(lama: SpandrelModel) -> None:
    out = inpaint_image(_image(48, 64, 3), _mask(48, 64), lama)
    assert (out.height, out.width, out.layout) == (48, 64, Layout.RGB)


def test_mask_is_scaled_to_the_image(lama: SpandrelModel) -> None:
    out = inpaint_image(_image(48, 64, 3), _mask(24, 32), lama)
    assert (out.height, out.width) == (48, 64)


def test_alpha_passes_through(lama: SpandrelModel) -> None:
    image = _image(48, 64, 4)
    out = inpaint_image(image, _mask(48, 64), lama)
    assert out.layout == Layout.RGBA
    torch.testing.assert_close(out.tensor(CPU)[:, 3], image.tensor(CPU)[:, 3])


def test_gray_stays_gray(lama: SpandrelModel) -> None:
    assert inpaint_image(_image(48, 64, 1), _mask(48, 64), lama).layout == Layout.GRAY


def test_sr_model_is_refused(tiny_sr_path: Path) -> None:
    with pytest.raises(ValueError, match="not an inpainting model"):
        inpaint_image(_image(8, 8, 3), _mask(8, 8), load_model(tiny_sr_path, CPU, "fp32"))


def test_empty_image_is_refused(lama: SpandrelModel) -> None:
    with pytest.raises(ValueError, match="empty image"):
        inpaint_image(ImageValue.empty(Layout.RGB), _mask(8, 8), lama)
