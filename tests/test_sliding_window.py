"""Independent MONAI 1.6.0 references for placement, padding, and FP32 blending."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
from monai.data.utils import compute_importance_map as monai_importance  # noqa: E402
from monai.data.utils import dense_patch_slices as monai_slices  # noqa: E402
from monai.inferers.utils import _get_scan_interval as monai_interval  # noqa: E402
from monai.inferers.utils import sliding_window_inference as monai_inference  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from medmlx_core.sliding_window import (  # noqa: E402
    _get_scan_interval,
    _pad_input,
    compute_importance_map,
    dense_patch_slices,
    sliding_window_inference,
)

# Staged bundle settings supplied with this extraction; unspecified options
# retain MONAI 1.6.0 defaults. NV settings come from mlx_worker/streaming.py.
ROUTE_SETTINGS = [
    pytest.param((240, 240, 160), 1, 0.5, "constant", id="brats_segresnet"),
    pytest.param((96, 96, 96), 4, 0.25, "constant", id="renal_cect_segresnet"),
    pytest.param((96, 96, 96), 4, 0.5, "constant", id="renal_unest"),
    pytest.param((128, 128, 128), 1, 0.25, "replicate", id="nv_segment_ct"),
]


def torch_predict(patch):
    anchor = patch[(slice(None), slice(0, 1), *(slice(0, 1),) * (patch.ndim - 2))]
    return torch.cat((patch * 0.5 + anchor * 0.25, patch * -0.25 + 0.125), dim=1)


def mlx_predict(patch):
    anchor = patch[(slice(None), slice(0, 1), *(slice(0, 1),) * (patch.ndim - 2))]
    left, offset, right = patch * 0.5, anchor * 0.25, patch * -0.25
    mx.eval(left, offset, right)  # Separate multiply/add, like Torch eager execution.
    return mx.concatenate((left + offset, right + 0.125), axis=1)


def assert_parity(actual, expected, *, exact=False):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if exact:
        assert np.array_equal(actual, expected)
    else:
        error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
        # NumPy/Torch FP32 exp kernels differ by a rounding unit; weighting,
        # ordered addition, and division propagate that small Gaussian error.
        assert error <= 1e-6, f"max abs difference {error} exceeds 1e-6"


@pytest.mark.parametrize("shape,roi", [((11, 9, 7), (8, 6, 4)), ((3, 5, 2), (8, 8, 6))])
@pytest.mark.parametrize("overlap", [0.25, 0.5])
@pytest.mark.parametrize("mode", ["constant", "gaussian"])
@pytest.mark.parametrize("sw_batch_size", [1, 3])
def test_patch_dependent_blending_matches_monai(shape, roi, overlap, mode, sw_batch_size):
    # Two images expose batch-crossing window groups; two channels expose channel placement.
    inputs = np.random.default_rng(1203).integers(-64, 65, (2, 2, *shape)).astype(np.float32) / 64
    original = inputs.copy()
    options = dict(overlap=overlap, mode=mode, padding_mode="constant", cval=-0.25)
    expected = monai_inference(
        torch.from_numpy(inputs), roi, sw_batch_size, torch_predict, **options
    )
    actual = sliding_window_inference(inputs, roi, sw_batch_size, mlx_predict, **options)
    assert_parity(actual, expected.numpy(), exact=mode == "constant")
    assert np.array_equal(inputs, original)


@pytest.mark.parametrize(
    "mode,cval", [("constant", -0.75), ("reflect", 0), ("replicate", 0), ("circular", 0)]
)
def test_padding_values_and_crop_match_torch(mode, cval):
    inputs = np.arange(60, dtype=np.float32).reshape(1, 2, 3, 5, 2).swapaxes(2, 3)
    roi = (8, 6, 4)
    padded, crop = _pad_input(inputs, roi, mode, cval)
    expected = F.pad(torch.from_numpy(inputs), (1, 1, 1, 2, 1, 2), mode=mode, value=cval)
    assert np.array_equal(padded, expected.numpy())
    assert crop == (slice(1, 6), slice(1, 4), slice(1, 3))
    assert np.array_equal(padded[(slice(None), slice(None), *crop)], inputs)
    expected_logits = monai_inference(
        torch.from_numpy(inputs / 64), roi, 1, torch_predict, padding_mode=mode, cval=cval
    )
    actual = sliding_window_inference(
        inputs / 64, roi, 1, mlx_predict, padding_mode=mode, cval=cval
    )
    assert_parity(actual, expected_logits.numpy(), exact=True)


@pytest.mark.parametrize("overlap,axis0", [(0.25, [0, 6, 11]), (0.5, [0, 4, 8, 11])])
def test_coordinates_match_monai_edge_shift_and_order(overlap, axis0):
    image, roi = (19, 11, 13), (8, 8, 8)
    amounts = (overlap,) * 3
    interval = _get_scan_interval(image, roi, 3, amounts)
    assert interval == monai_interval(image, roi, 3, amounts)
    windows = dense_patch_slices(image, roi, interval)
    assert windows == monai_slices(image, roi, interval)
    assert list(dict.fromkeys(window[0].start for window in windows)) == axis0
    assert windows[-1] == (slice(11, 19), slice(3, 11), slice(5, 13))
    assert dense_patch_slices(image, roi, interval, False) == monai_slices(
        image, roi, interval, False
    )
    assert _get_scan_interval((2, 9), (2, 3), 2, (0.5, 0.99)) == (2, 1)


@pytest.mark.parametrize(
    "patch,scale",
    [((8, 8, 6), 0.125), ((7, 5, 3), (0.2, 0.125, 0.25)), ((1, 4, 5), 0.125), ((8, 6, 4), 0.01)],
)
@pytest.mark.parametrize("mode", ["constant", "gaussian"])
def test_importance_map_matches_monai_center_clamp_and_dtype(patch, scale, mode):
    expected = monai_importance(patch, mode, scale).numpy()
    actual = compute_importance_map(patch, mode, scale)
    assert actual.dtype == np.float32
    if mode == "constant" or scale == 0.01:
        # A sufficiently narrow Gaussian is entirely clamped to 1e-3.
        assert np.array_equal(actual, expected)
    else:
        # Same FP32 separable operations; exp differs between the independent kernels.
        assert float(np.max(np.abs(actual - expected))) <= 1e-7
        assert actual.min() >= np.float32(1e-3)
        assert np.array_equal(actual, np.flip(actual))
    expected_half = monai_importance(patch, mode, scale, dtype=torch.float16).numpy()
    assert np.array_equal(compute_importance_map(patch, mode, scale, np.float16), expected_half)


@pytest.mark.parametrize("full_roi,sw_batch_size,overlap,padding_mode", ROUTE_SETTINGS)
@pytest.mark.parametrize("smaller_than_roi", [False, True], ids=["odd_overlap", "padding"])
def test_scaled_route_settings_match_monai(
    full_roi, sw_batch_size, overlap, padding_mode, smaller_than_roi
):
    roi = tuple(size // 16 for size in full_roi)
    shape = tuple(size - 3 if smaller_than_roi else size * 2 + 1 for size in roi)
    inputs = np.random.default_rng(703).integers(-32, 33, (1, 1, *shape)).astype(np.float32) / 32
    options = dict(
        overlap=overlap,
        mode="constant",
        sigma_scale=0.125,
        padding_mode=padding_mode,
        cval=0,
        sw_device="cpu",
        device="cpu",
        buffer_steps=None,
    )
    expected = monai_inference(
        torch.from_numpy(inputs), roi, sw_batch_size, torch_predict, **options
    ).numpy()
    actual = sliding_window_inference(inputs, roi, sw_batch_size, mlx_predict, **options)
    assert_parity(actual, expected, exact=True)


@pytest.mark.parametrize("roi,sw_batch_size,overlap,padding_mode", ROUTE_SETTINGS)
def test_full_roi_route_settings_match_monai(roi, sw_batch_size, overlap, padding_mode):
    # Exercise the real patch dimensions and padding without a costly network.
    shape = tuple(size - 1 for size in roi)
    inputs = np.random.default_rng(704).integers(-32, 33, (1, 1, *shape), dtype=np.int8)
    inputs = inputs.astype(np.float32) / 32

    def predict_torch(patch):
        assert patch.shape == (1, 1, *roi)
        return patch * 0.5 + 0.125

    def predict_mlx(patch):
        assert patch.shape == (1, 1, *roi)
        result = patch * 0.5
        mx.eval(result)
        return result + 0.125

    options = dict(overlap=overlap, padding_mode=padding_mode, device="cpu", sw_device="cpu")
    expected = monai_inference(
        torch.from_numpy(inputs), roi, sw_batch_size, predict_torch, **options
    ).numpy()
    actual = sliding_window_inference(inputs, roi, sw_batch_size, predict_mlx, **options)
    assert_parity(actual, expected, exact=True)


@pytest.mark.parametrize("overlap", [0.25, 0.5], ids=["renal_cect", "renal_unest"])
def test_radnn_cpu_batch_override_matches_monai(overlap):
    # Both frozen bundle adapters override configured batch size 4 to 1.
    inputs = np.random.default_rng(705).integers(-32, 33, (1, 1, 13, 11, 5))
    inputs = inputs.astype(np.float32) / 32
    options = dict(overlap=overlap, device="cpu", sw_device="cpu")
    expected = monai_inference(torch.from_numpy(inputs), (6, 6, 6), 1, torch_predict, **options)
    actual = sliding_window_inference(inputs, (6, 6, 6), 1, mlx_predict, **options)
    assert_parity(actual, expected.numpy(), exact=True)


def test_cached_gaussian_map_and_coordinate_predictor_are_bitwise_equal():
    inputs = np.random.default_rng(11).integers(-32, 33, (2, 1, 9, 7)).astype(np.float32) / 32
    weight = monai_importance((6, 4), "gaussian", (0.2, 0.15)).numpy()
    expected_coords, actual_coords = [], []

    def predict_torch(patch, coords, scale, *, bias):
        expected_coords.extend(coords)
        # MONAI starts are np.int64, so these expressions are np.float64.
        # Without dtype=patch.dtype, Torch infers FP64 offsets/predictions and
        # weights in FP64 before casting to the FP32 accumulator. MLX uses FP32.
        offsets = torch.tensor(
            [c[2].start / 32 + c[3].start / 64 for c in coords], dtype=patch.dtype
        )[:, None, None, None]
        return patch * scale + offsets + bias

    def predict_mlx(patch, coords, scale, *, bias):
        actual_coords.extend(coords)
        offsets = mx.array([c[2].start / 32 + c[3].start / 64 for c in coords])[:, None, None, None]
        result = patch * scale
        mx.eval(result)
        return result + offsets + bias

    options = dict(overlap=(0.5, 0.25), roi_weight_map=weight, with_coord=True)
    expected = monai_inference(
        torch.from_numpy(inputs),
        (6, 4),
        4,
        predict_torch,
        **{**options, "roi_weight_map": torch.from_numpy(weight)},
        scale=0.5,
        bias=0.125,
    )
    actual = sliding_window_inference(
        mx.array(inputs),
        (6, 4),
        4,
        predict_mlx,
        **options,
        scale=0.5,
        bias=0.125,
        sw_device="cpu",
        device="cpu",
    )
    assert actual_coords == expected_coords
    assert_parity(actual, expected.numpy(), exact=True)


@pytest.mark.parametrize(
    "input_dtype,prediction_dtype", [(np.float32, np.float16), (np.float16, np.float32)]
)
def test_predictor_weighting_precedes_accumulator_dtype_conversion(input_dtype, prediction_dtype):
    inputs = np.random.default_rng(41).integers(-32, 33, (1, 1, 9, 7)).astype(input_dtype) / 32
    weight = monai_importance((6, 4), "gaussian").numpy()
    torch_dtype = torch.float16 if prediction_dtype == np.float16 else torch.float32
    mlx_dtype = mx.float16 if prediction_dtype == np.float16 else mx.float32
    expected = monai_inference(
        torch.from_numpy(inputs),
        (6, 4),
        1,
        lambda x: torch_predict(x).to(torch_dtype),
        roi_weight_map=torch.from_numpy(weight),
    )
    actual = sliding_window_inference(
        inputs, (6, 4), 1, lambda x: mlx_predict(x).astype(mlx_dtype), roi_weight_map=weight
    )
    assert_parity(actual, expected.numpy(), exact=True)


@pytest.mark.parametrize("roi", [4, (None, -1)])
def test_scalar_and_fallback_roi_match_monai(roi):
    inputs = np.arange(35, dtype=np.float32).reshape(1, 1, 5, 7) / 64
    expected = monai_inference(torch.from_numpy(inputs), roi, 2, torch_predict, overlap=0.5)
    actual = sliding_window_inference(inputs, roi, 2, mlx_predict, overlap=0.5)
    assert_parity(actual, expected.numpy(), exact=True)


@pytest.mark.parametrize("mode", ["constant", "gaussian"])
def test_frozen_radnn_rolling_scores_and_labels(mode):
    with np.load(Path(__file__).parent / "fixtures" / f"radnn_rolling_{mode}.npz") as reference:
        assert reference["radnn_commit"].item() == "cbaa1ac"
        inputs, weights, bias = reference["inputs"], reference["weights"], reference["bias"]

        def predict(patch):
            # The fixture predictor is NumPy arithmetic exported to MLX by
            # RadNN's real rolling module; no model/weight converter is moved.
            host = np.asarray(patch)
            return mx.array(host * weights + bias)

        actual = sliding_window_inference(
            inputs, tuple(reference["roi_size"]), 1, predict, mode=mode, padding_mode="replicate"
        )
        assert_parity(actual, reference["scores"], exact=mode == "constant")
        ids = reference["class_ids"]
        labels = ids[np.argmax(actual, axis=1)]
        labels[np.max(actual, axis=1) <= 0] = 0
        assert np.array_equal(labels[:, None], reference["labels"])


@pytest.mark.parametrize(
    "options,error,message",
    [
        ({"overlap": 1}, ValueError, "overlap must"),
        ({"buffer_steps": 2}, NotImplementedError, "buffer_steps"),
        ({"device": "gpu"}, ValueError, "device must"),
    ],
)
def test_invalid_or_unimplemented_options_fail_before_prediction(options, error, message):
    def unexpected_call(_):
        pytest.fail("invalid request reached predictor")

    with pytest.raises(error, match=message):
        sliding_window_inference(
            np.zeros((1, 1, 3, 3), np.float32), 4, 1, unexpected_call, **options
        )


@pytest.mark.parametrize(
    "predictor,error,message",
    [
        (lambda x: (x, x), TypeError, "one MLX array"),
        (lambda x: x[:, :, ::2, ::2], ValueError, "ROI spatial shape"),
    ],
)
def test_unsupported_predictor_outputs_fail_explicitly(predictor, error, message):
    with pytest.raises(error, match=message):
        sliding_window_inference(np.zeros((1, 1, 4, 4), np.float32), 4, 1, predictor)
