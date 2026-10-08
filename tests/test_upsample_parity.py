"""Independent Torch decoder oracles, including the repeated 48-cubed operation."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medmlx_core.runtime import import_mlx

_RAW_LOGIT_RTOL = 1e-4
_RAW_LOGIT_ATOL = 1e-4


def test_renal_deconv2x_matches_torch_repeated() -> None:
    """Exercise the MLX 0.32.2-sensitive 48-cubed decoder operation."""

    from medmlx_core.upsample import deconv2x_ncdhw

    checkpoint_root = os.environ.get("MEDMLX_TEST_CHECKPOINT_ROOT")
    if checkpoint_root is None:
        pytest.skip("set MEDMLX_TEST_CHECKPOINT_ROOT to the staged MONAI bundle directory")
    checkpoint = Path(checkpoint_root) / "renalStructures_CECT_segmentation/models/model.pt"
    if not checkpoint.is_file():
        pytest.skip(f"staged SegResNet checkpoint is missing at {checkpoint}")
    mx = import_mlx()
    torch: Any = pytest.importorskip("torch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    weight = state["up_samples.0.1.deconv.weight"].detach().numpy()
    bias = state["up_samples.0.1.deconv.bias"].detach().numpy()
    rng = np.random.default_rng(617)
    values = rng.normal(size=(1, weight.shape[0], 48, 48, 48)).astype(np.float32)
    expected = torch.nn.functional.conv_transpose3d(
        torch.from_numpy(values),
        weight=torch.from_numpy(weight),
        bias=torch.from_numpy(bias),
        stride=2,
    ).numpy()

    for _ in range(3):
        actual = deconv2x_ncdhw(mx.array(values), mx.array(weight), mx.array(bias), mx=mx)
        mx.eval(actual)
        host = np.asarray(actual)
        assert host.dtype == np.float32
        assert np.allclose(host, expected, rtol=_RAW_LOGIT_RTOL, atol=_RAW_LOGIT_ATOL)
        assert np.array_equal(host >= 0, expected >= 0)


@pytest.mark.parametrize("strided", [False, True])
def test_decoder_fusion_matches_torch_at_borders_and_singleton_axes(strided: bool) -> None:
    from medmlx_core.upsample import upsample_add_ncdhw

    mx = import_mlx()
    torch: Any = pytest.importorskip("torch")
    rng = np.random.default_rng(42)
    values = rng.normal(size=(2, 3, 1, 3, 5)).astype(np.float32)
    skip = rng.normal(size=(2, 3, 2, 6, 10)).astype(np.float32)
    expected = (
        torch.nn.functional.interpolate(
            torch.from_numpy(values.copy()), scale_factor=2, mode="trilinear", align_corners=False
        ).numpy()
        + skip
    )
    inputs = [mx.array(v) for v in (values, skip)]
    if strided:
        inputs = [mx.array(v.swapaxes(1, 4).copy()).swapaxes(1, 4) for v in (values, skip)]
    actual = upsample_add_ncdhw(*inputs, mx=mx)
    mx.eval(actual)
    assert np.allclose(np.asarray(actual), expected, rtol=_RAW_LOGIT_RTOL, atol=_RAW_LOGIT_ATOL)
