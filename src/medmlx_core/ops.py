"""Shared MAISI MLX primitives. Importing this module does not import MLX."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from medmlx_core.layout import (
    conv3d_weight_to_mlx,
    conv_transpose3d_weight_to_mlx,
    require_ncdhw,
    to_ncdhw,
    to_ndhwc,
)


def as_fp32(array: Any, mx: Any) -> Any:
    """Copy a host or device array onto the MLX GPU as float32."""

    return mx.array(array, dtype=mx.float32)


def silu(array: Any, mx: Any) -> Any:
    """SiLU, matching ``torch.nn.SiLU`` / MONAI ``Swish(alpha=1)``."""

    return array * mx.sigmoid(array)


def linear(array: Any, weight: Any, bias: Any | None, mx: Any) -> Any:
    """``x @ W.T + b`` with Torch/MLX layout ``[out, in]``."""

    output = array @ mx.swapaxes(weight, 0, 1)
    if bias is not None:
        output = output + bias
    return output


def group_norm_ncdhw(
    array: Any,
    weight: Any,
    bias: Any,
    *,
    num_groups: int,
    eps: float,
    mx: Any,
) -> Any:
    """NCDHW GroupNorm matching ``mlx.nn.GroupNorm(..., pytorch_compatible=True)``."""

    require_ncdhw(array, name="group_norm input")
    spatial = to_ndhwc(array, mx)
    batch = int(spatial.shape[0])
    channels = int(spatial.shape[-1])
    if channels % num_groups != 0:
        raise ValueError(
            f"group_norm channels {channels} must be divisible by num_groups {num_groups}"
        )
    group_size = channels // num_groups
    grouped = mx.reshape(spatial, (batch, -1, num_groups, group_size))
    grouped = mx.transpose(grouped, (0, 2, 1, 3))
    grouped = mx.reshape(grouped, (batch, num_groups, -1))
    normalized = mx.fast.layer_norm(grouped, eps=eps, weight=None, bias=None)
    restored = mx.reshape(normalized, (batch, num_groups, -1, group_size))
    restored = mx.transpose(restored, (0, 2, 1, 3))
    restored = mx.reshape(restored, spatial.shape)
    return to_ncdhw(restored * weight + bias, mx)


def conv3d_ncdhw(
    array: Any,
    weight: Any,
    bias: Any | None,
    *,
    padding: int,
    mx: Any,
    stride: int = 1,
) -> Any:
    """NCDHW conv3d: NDHWC call, Torch weight remap, NCDHW return."""

    require_ncdhw(array, name="conv3d input")
    output = mx.conv3d(
        to_ndhwc(array, mx),
        conv3d_weight_to_mlx(weight, mx),
        stride=stride,
        padding=padding,
    )
    if bias is not None:
        output = output + bias
    return to_ncdhw(output, mx)


def conv_transpose3d_ncdhw(
    array: Any,
    weight: Any,
    bias: Any | None,
    *,
    padding: int,
    output_padding: int,
    mx: Any,
    stride: int = 1,
) -> Any:
    """NCDHW conv_transpose3d using Torch weight layout ``[I, O, K]``."""

    require_ncdhw(array, name="conv_transpose3d input")
    output = mx.conv_transpose3d(
        to_ndhwc(array, mx),
        conv_transpose3d_weight_to_mlx(weight, mx),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    if bias is not None:
        output = output + bias
    return to_ncdhw(output, mx)


def split_conv3d_ncdhw(
    array: Any,
    weight: Any,
    bias: Any | None,
    *,
    padding: int,
    mx: Any,
    stride: int = 1,
    num_splits: int = 1,
    dim_split: int = 1,
    convolution: Callable[..., Any] = conv3d_ncdhw,
) -> Any:
    """MaisiConvolution: one conv, or overlap-split along NCDHW axis ``dim_split+2``."""

    return _split_convolution_ncdhw(
        array,
        apply=lambda chunk: convolution(
            chunk, weight, bias, padding=padding, stride=stride, mx=mx
        ),
        stride=stride,
        num_splits=num_splits,
        dim_split=dim_split,
        name="split_conv3d",
        mx=mx,
    )


def split_conv_transpose3d_ncdhw(
    array: Any,
    weight: Any,
    bias: Any | None,
    *,
    padding: int,
    output_padding: int,
    mx: Any,
    stride: int = 1,
    num_splits: int = 1,
    dim_split: int = 1,
) -> Any:
    """Split/stitch MAISI ConvTranspose3d along one spatial axis."""

    return _split_convolution_ncdhw(
        array,
        apply=lambda chunk: conv_transpose3d_ncdhw(
            chunk,
            weight,
            bias,
            padding=padding,
            output_padding=output_padding,
            stride=stride,
            mx=mx,
        ),
        stride=stride,
        num_splits=num_splits,
        dim_split=dim_split,
        name="split_conv_transpose3d",
        mx=mx,
    )


def _split_convolution_ncdhw(
    array: Any,
    *,
    apply: Callable[[Any], Any],
    stride: int,
    num_splits: int,
    dim_split: int,
    name: str,
    mx: Any,
) -> Any:
    require_ncdhw(array, name=f"{name} input")
    if num_splits <= 1:
        return apply(array)
    if dim_split not in (0, 1, 2):
        raise ValueError(f"dim_split must be 0, 1, or 2; got {dim_split}")
    axis = dim_split + 2
    length = int(array.shape[axis])
    split_size = length // num_splits
    if split_size < 1:
        raise ValueError(
            f"{name} length {length} is smaller than num_splits {num_splits}"
        )
    overlap = 3
    if overlap % stride > 0:
        overlap = (overlap // stride + 1) * stride
    remainder = length % split_size
    chunks: list[Any] = []
    for index in range(num_splits):
        start = 0 if index == 0 else index * split_size - overlap
        extra = remainder if index == num_splits - 1 else overlap
        end = (index + 1) * split_size + extra
        chunks.append(_slice_axis(array, axis, start, end))
    outputs = [apply(chunk) for chunk in chunks]
    other = (dim_split + 1 if dim_split < 2 else 0) + 2
    in_other = int(chunks[0].shape[other])
    out_other = int(outputs[0].shape[other])
    out_split = split_size
    out_overlap = overlap
    # MaisiConvolution checks integer // == 2, not exact 2x growth. A
    # trailing-pad stride-2 downsample maps S+1 to S//2, for example 16 -> 17 -> 8.
    if in_other > 0 and out_other // in_other == 2:
        out_split *= 2
        out_overlap *= 2
    elif out_other > 0 and in_other // out_other == 2:
        out_split //= 2
        out_overlap //= 2
    cropped: list[Any] = []
    for index, output in enumerate(outputs):
        if index == 0:
            cropped.append(_slice_axis(output, axis, 0, out_split))
        else:
            cropped.append(
                _slice_axis(output, axis, out_overlap, out_overlap + out_split)
            )
    return mx.concatenate(cropped, axis=axis)


def _slice_axis(array: Any, axis: int, start: int, end: int) -> Any:
    slices: list[slice] = [slice(None)] * 5
    slices[axis] = slice(start, end)
    return array[tuple(slices)]


def avg_pool3d_ncdhw(array: Any, *, mx: Any, kernel_size: int = 2, stride: int = 2) -> Any:
    """NCDHW AvgPool3d matching ``mlx.nn.AvgPool3d`` / Torch ``AvgPool3d(2)``."""

    require_ncdhw(array, name="avg pool input")
    spatial = to_ndhwc(array, mx)
    mlx_nn = importlib.import_module("mlx.nn")

    pooled = mlx_nn.AvgPool3d(kernel_size=kernel_size, stride=stride)(spatial)
    return to_ncdhw(pooled, mx)


def upsample_nearest_ncdhw(array: Any, *, mx: Any, scale: int = 2) -> Any:
    """NCDHW nearest x scale. Same as ``mlx.nn.Upsample(..., mode='nearest')``."""

    require_ncdhw(array, name="nearest upsample input")
    if scale < 1:
        raise ValueError(f"upsample scale must be >= 1; got {scale}")
    output = array
    for axis in (2, 3, 4):
        output = mx.repeat(output, scale, axis=axis)
    return output


def upsample_trilinear_ncdhw(array: Any, *, mx: Any, scale: int = 2) -> Any:
    """NCDHW trilinear x scale, ``align_corners=False`` (Torch interpolate default)."""

    require_ncdhw(array, name="trilinear upsample input")
    mlx_nn = importlib.import_module("mlx.nn")

    upsampled = mlx_nn.Upsample(
        scale_factor=scale, mode="linear", align_corners=False
    )(to_ndhwc(array, mx))
    return to_ncdhw(upsampled, mx)


def pad_spatial_trailing_ncdhw(array: Any, mx: Any) -> Any:
    """``F.pad(x, (0, 1) * 3)``: add one voxel on the high side of D, H, W."""

    require_ncdhw(array, name="spatial pad input")
    return mx.pad(array, [(0, 0), (0, 0), (0, 1), (0, 1), (0, 1)])


def concat_channels_ncdhw(left: Any, right: Any, mx: Any) -> Any:
    """Skip-concat on NCDHW channel axis 1."""

    require_ncdhw(left, name="skip concat left")
    require_ncdhw(right, name="skip concat right")
    return mx.concatenate([left, right], axis=1)
