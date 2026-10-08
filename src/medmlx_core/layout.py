"""NCDHW module layout and NDHWC remaps for MLX 3D conv/norm."""

from __future__ import annotations

from typing import Any

LAYOUT_NCDHW = "maisi.layout.ncdhw"
LAYOUT_TO_NDHWC = "maisi.layout.to_ndhwc"
LAYOUT_TO_NCDHW = "maisi.layout.to_ncdhw"
LAYOUT_CONV3D_W = "maisi.layout.conv3d_w"
LAYOUT_CONV_TRANSPOSE3D_W = "maisi.layout.conv_transpose3d_w"
LAYOUT_LINEAR_W = "maisi.layout.linear_w"
LAYOUT_EMBED_W = "maisi.layout.embed_w"
LAYOUT_GN_AFFINE = "maisi.layout.gn_affine"
LAYOUT_VECTOR = "maisi.layout.vector"
IDENTITY_WEIGHT_LAYOUTS = frozenset(
    {LAYOUT_LINEAR_W, LAYOUT_EMBED_W, LAYOUT_GN_AFFINE, LAYOUT_VECTOR}
)


def require_ncdhw(array: Any, *, name: str) -> None:
    """Raise if *array* is not a rank-5 NCDHW tensor."""

    if int(array.ndim) != 5:
        raise ValueError(f"{name} must be NCDHW rank-5; got ndim={array.ndim}")


def to_ndhwc(array: Any, mx: Any) -> Any:
    """NCDHW → NDHWC at a conv/norm call."""

    return mx.transpose(array, (0, 2, 3, 4, 1))


def to_ncdhw(array: Any, mx: Any) -> Any:
    """NDHWC → NCDHW after a conv/norm call."""

    return mx.transpose(array, (0, 4, 1, 2, 3))


def conv3d_weight_to_mlx(weight: Any, mx: Any) -> Any:
    """Torch conv3d weight [O, I, KD, KH, KW] → MLX [O, KD, KH, KW, I]."""

    if int(weight.ndim) != 5:
        raise ValueError(f"conv3d weight must be rank-5 [O,I,K]; got ndim={weight.ndim}")
    return mx.transpose(weight, (0, 2, 3, 4, 1))


def conv_transpose3d_weight_to_mlx(weight: Any, mx: Any) -> Any:
    """Torch conv_transpose3d weight [I, O, KD, KH, KW] → MLX [O, KD, KH, KW, I]."""

    if int(weight.ndim) != 5:
        raise ValueError(f"conv_transpose3d weight must be rank-5 [I,O,K]; got ndim={weight.ndim}")
    return mx.transpose(weight, (1, 2, 3, 4, 0))


def tokens_from_ncdhw(array: Any, mx: Any) -> Any:
    """NCDHW [B,C,D,H,W] → tokens [B, D*H*W, C]."""

    require_ncdhw(array, name="attention input")
    batch, channels, depth, height, width = (int(dim) for dim in array.shape)
    flat = mx.reshape(array, (batch, channels, depth * height * width))
    return mx.transpose(flat, (0, 2, 1))


def tokens_to_ncdhw(tokens: Any, spatial: tuple[int, int, int, int, int], mx: Any) -> Any:
    """Tokens [B, N, C] → NCDHW *spatial*."""

    batch, channels, depth, height, width = spatial
    ncdhw = mx.transpose(tokens, (0, 2, 1))
    return mx.reshape(ncdhw, (batch, channels, depth, height, width))
