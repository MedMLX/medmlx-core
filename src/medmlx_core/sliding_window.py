# Copyright (c) MONAI Consortium
# Portions derived from MONAI, licensed under the Apache License, Version 2.0.
# See licenses/MONAI_LICENSE for the license and THIRD_PARTY_NOTICES.md for sources.

"""MONAI 1.6.0 window placement and host stitching for MLX predictors.

Only NumPy and MLX are required. Inputs and outputs are channel first with a
batch dimension. Stitching uses the input dtype and MONAI's sequential patch
order; only the predictor runs on the MLX device.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from typing import Any

import mlx.core as mx
import numpy as np

__all__ = ["compute_importance_map", "dense_patch_slices", "sliding_window_inference"]


def _tuple_rep(value: Any, dimensions: int) -> tuple:
    if np.isscalar(value) or value is None:
        return (value,) * dimensions
    result = tuple(value)
    if len(result) != dimensions:
        raise ValueError(f"Expected {dimensions} values, got {len(result)}")
    return result


def _get_scan_interval(
    image_size: Sequence[int],
    roi_size: Sequence[int],
    num_spatial_dims: int,
    overlap: Sequence[float],
) -> tuple[int, ...]:
    """Use integer strides, with a minimum of one voxel, as MONAI does."""
    if len(image_size) != num_spatial_dims or len(roi_size) != num_spatial_dims:
        raise ValueError("image_size and roi_size must match num_spatial_dims")
    return tuple(
        int(roi) if roi == size else max(int(roi * (1 - amount)), 1)
        for size, roi, amount in zip(image_size, roi_size, overlap, strict=True)
    )


def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
    return_slice: bool = True,
) -> list[tuple]:
    """Enumerate MONAI's row-major windows, shifting the last to the edge."""
    patch = tuple(
        min(size, roi) if roi else size for size, roi in zip(image_size, patch_size, strict=True)
    )
    starts = []
    for size, roi, step in zip(image_size, patch, scan_interval, strict=True):
        if size < 1 or roi < 1 or step < 0:
            raise ValueError("Image/patch sizes must be positive and strides nonnegative")
        count = (size - roi + step - 1) // step + 1 if step else 1
        starts.append([min(index * step, size - roi) for index in range(count)])
    if return_slice:
        return [
            tuple(slice(start, start + roi) for start, roi in zip(origin, patch, strict=True))
            for origin in product(*starts)
        ]
    return [
        tuple((start, start + roi) for start, roi in zip(origin, patch, strict=True))
        for origin in product(*starts)
    ]


def compute_importance_map(
    patch_size: Sequence[int],
    mode: str = "constant",
    sigma_scale: Sequence[float] | float = 0.125,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Return MONAI's separable FP32 Gaussian, clamped at a minimum of 1e-3.

    The Gaussian is centered at (size - 1) / 2, without peak normalization.
    NumPy and Torch exp kernels can differ by an FP32 rounding unit.
    """
    patch = tuple(patch_size)
    if not patch or any(size < 1 for size in patch):
        raise ValueError("patch_size must contain positive sizes")
    if mode == "constant":
        importance = np.ones(patch, dtype=np.float32)
    elif mode == "gaussian":
        scales = _tuple_rep(sigma_scale, len(patch))
        for axis, (size, scale) in enumerate(zip(patch, scales, strict=True)):
            sigma = size * scale
            x = np.arange(size, dtype=np.float32) - np.float32((size - 1) / 2)
            x = np.exp(x**2 / np.float32(-2 * sigma**2))
            importance = importance[..., None] * x[(None,) * axis] if axis else x
    else:
        raise ValueError(f"Unsupported mode: {mode}; expected constant or gaussian")
    minimum = max(float(importance.min()), 1e-3)
    return np.maximum(importance, np.float32(minimum)).astype(dtype)


def _pad_input(
    inputs: np.ndarray, roi_size: Sequence[int], padding_mode: str, cval: float
) -> tuple[np.ndarray, tuple[slice, ...]]:
    padding = [(0, 0), (0, 0)]
    crop = []
    for size, roi in zip(inputs.shape[2:], roi_size, strict=True):
        extra = max(roi - size, 0)
        before = extra // 2
        padding.append((before, extra - before))
        crop.append(slice(before, before + size))
    if not any(before or after for before, after in padding):
        return inputs, tuple(crop)
    modes = {"constant": "constant", "reflect": "reflect", "replicate": "edge", "circular": "wrap"}
    if padding_mode not in modes:
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")
    if padding_mode != "constant" and cval != 0:
        raise ValueError(f"Padding mode {padding_mode} does not accept a nonzero cval")
    for size, (before, after) in zip(inputs.shape[2:], padding[2:], strict=True):
        if padding_mode == "reflect" and max(before, after) >= size:
            raise ValueError("Reflect padding must be smaller than the input dimension")
        if padding_mode == "circular" and max(before, after) > size:
            raise ValueError("Circular padding cannot wrap more than once")
    options = {"constant_values": cval} if padding_mode == "constant" else {}
    return np.pad(inputs, padding, mode=modes[padding_mode], **options), tuple(crop)


def sliding_window_inference(
    inputs: np.ndarray | mx.array,
    roi_size: Sequence[int | None] | int,
    sw_batch_size: int,
    predictor: Callable[..., mx.array],
    overlap: Sequence[float] | float = 0.25,
    mode: str = "constant",
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: str = "constant",
    cval: float = 0.0,
    sw_device: mx.Device | str | None = None,
    device: str | None = None,
    progress: bool = False,
    roi_weight_map: np.ndarray | mx.array | None = None,
    process_fn: Callable | None = None,
    buffer_steps: int | None = None,
    buffer_dim: int = -1,
    with_coord: bool = False,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """Predict and blend 1D/2D/3D windows, returning host NumPy logits.

    The predictor receives MLX ``(N, C, *roi_size)`` arrays and must return one
    MLX array at the same spatial resolution (output channels may differ).
    Output dtype follows the input, including when predictor dtype differs.
    ``with_coord`` passes MONAI's list of batch/channel/spatial slice lists.

    ``sw_device`` selects the MLX predictor device; ``device`` is None or
    'cpu' because stitching is on the host. Active buffering, process_fn,
    progress bars, multiple outputs, and scaled outputs are unsupported.
    None/nonpositive ROI components use the corresponding input size.
    """
    if device not in (None, "cpu"):
        raise ValueError("device must be None or 'cpu' for NumPy stitching")
    if buffer_steps is not None and buffer_steps > 0:
        raise NotImplementedError("Positive buffer_steps is not supported")
    if process_fn is not None or progress:
        raise NotImplementedError("process_fn and progress bars are not supported")
    if isinstance(sw_device, str):
        if sw_device not in ("cpu", "gpu"):
            raise ValueError("sw_device must be an MLX device, 'cpu', or 'gpu'")
        sw_device = mx.cpu if sw_device == "cpu" else mx.gpu
    host = np.asarray(inputs)
    spatial = host.shape[2:]
    if not 1 <= len(spatial) <= 3 or any(size < 1 for size in host.shape):
        raise ValueError("inputs must be a nonempty (N, C, *spatial) 1D/2D/3D array")
    if host.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError("inputs must have float16 or float32 dtype")
    if not isinstance(sw_batch_size, int) or sw_batch_size < 1:
        raise ValueError("sw_batch_size must be a positive integer")
    roi = tuple(
        int(value) if value is not None and value > 0 else size
        for value, size in zip(_tuple_rep(roi_size, len(spatial)), spatial, strict=True)
    )
    amounts = _tuple_rep(overlap, len(spatial))
    if any(not 0 <= amount < 1 for amount in amounts):
        raise ValueError(f"overlap must be >= 0 and < 1, got {amounts}")
    padded, crop = _pad_input(host, roi, padding_mode, cval)
    image_size = padded.shape[2:]
    intervals = _get_scan_interval(image_size, roi, len(spatial), amounts)
    windows = dense_patch_slices(image_size, roi, intervals)
    importance = (
        compute_importance_map(roi, mode, sigma_scale, host.dtype)
        if roi_weight_map is None
        else np.asarray(roi_weight_map, dtype=host.dtype)
    )
    if importance.shape not in (roi, (1, 1, *roi)):
        raise ValueError("roi_weight_map must have shape roi_size or (1, 1, *roi_size)")
    importance = importance.reshape(1, 1, *roi)
    output = None
    counts = np.zeros((1, 1, *image_size), dtype=host.dtype)
    # MONAI builds this once per spatial window, independently of image batch.
    for window in windows:
        counts[(slice(None), slice(None), *window)] += importance
    num_windows = len(windows)
    for start in range(0, num_windows * host.shape[0], sw_batch_size):
        coordinates = [
            [
                slice(index // num_windows, index // num_windows + 1),
                slice(None),
                *windows[index % num_windows],
            ]
            for index in range(start, min(start + sw_batch_size, num_windows * host.shape[0]))
        ]
        patches = np.concatenate([padded[tuple(coord)] for coord in coordinates], axis=0)
        with mx.stream(sw_device if sw_device is not None else mx.default_device()):
            batch = mx.array(patches)
            prediction = (
                predictor(batch, coordinates, *args, **kwargs)
                if with_coord
                else predictor(batch, *args, **kwargs)
            )
            if not isinstance(prediction, mx.array):
                raise TypeError("predictor must return one MLX array")
            mx.eval(prediction)
            predicted = np.array(prediction, copy=True)
        if (
            predicted.ndim != host.ndim
            or predicted.shape[0] != len(coordinates)
            or predicted.shape[2:] != roi
            or predicted.shape[1] < 1
        ):
            raise ValueError("predictor output must match the window batch and ROI spatial shape")
        if output is None:
            output = np.zeros((host.shape[0], predicted.shape[1], *image_size), dtype=host.dtype)
        elif predicted.shape[1] != output.shape[1]:
            raise ValueError("predictor output channels must stay constant across windows")
        # MONAI's in-place multiplication rounds to predictor dtype BEFORE the
        # addition into the input-dtype accumulator. Keep these separate.
        predicted *= importance
        for coord, patch in zip(coordinates, predicted, strict=True):
            output[tuple(coord)] += patch
    output /= counts
    return output[(slice(None), slice(None), *crop)]
