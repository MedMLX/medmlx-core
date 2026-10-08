"""Channel-last MLX arithmetic shared by the three pinned inference graphs."""

from collections.abc import Mapping
from typing import Any

from medmlx_core.upsample import deconv2x_ncdhw


class ChannelLastGraph:
    def __init__(self, weights: Mapping[str, Any], *, mx: Any) -> None:
        self.w = weights
        self.mx = mx

    def conv(self, x: Any, name: str, *, padding: int = 0, stride: int = 1, groups: int = 1) -> Any:
        w = self.w[f"{name}.weight"]
        axes = (0, *range(2, w.ndim), 1)
        operation = self.mx.conv3d if w.ndim == 5 else self.mx.conv2d
        y = operation(x, w.transpose(axes), stride=stride, padding=padding, groups=groups)
        bias = self.w.get(f"{name}.bias")
        return y if bias is None else y + bias

    def linear(self, x: Any, name: str) -> Any:
        y = x @ self.w[f"{name}.weight"].T
        bias = self.w.get(f"{name}.bias")
        return y if bias is None else y + bias

    def layer_norm(self, x: Any, name: str | None = None) -> Any:
        weight = None if name is None else self.w[f"{name}.weight"]
        bias = None if name is None else self.w[f"{name}.bias"]
        return self.mx.fast.layer_norm(x, weight, bias, 1e-5)

    def batch_norm(self, x: Any, name: str, *, eps: float = 1e-5) -> Any:
        scale = self.w[f"{name}.weight"] * self.mx.rsqrt(self.w[f"{name}.running_var"] + eps)
        return (x - self.w[f"{name}.running_mean"]) * scale + self.w[f"{name}.bias"]

    def relu(self, x: Any) -> Any:
        return self.mx.maximum(x, 0)

    def swish(self, x: Any) -> Any:
        return x * self.mx.sigmoid(x)

    def instance_norm(self, x: Any) -> Any:
        axes = tuple(range(1, x.ndim - 1))
        mean = self.mx.mean(x, axis=axes, keepdims=True)
        return (x - mean) * self.mx.rsqrt(self.mx.var(x, axis=axes, keepdims=True) + 1e-5)

    def residual3d(self, x: Any, name: str) -> Any:
        def activate(y: Any) -> Any:
            return self.mx.where(y >= 0, y, y * 0.01)

        y = activate(self.instance_norm(self.conv(x, f"{name}.conv1.conv", padding=1)))
        y = self.instance_norm(self.conv(y, f"{name}.conv2.conv", padding=1))
        if f"{name}.conv3.conv.weight" in self.w:
            x = self.instance_norm(self.conv(x, f"{name}.conv3.conv"))
        return activate(y + x)

    def deconv3d(self, x: Any, name: str) -> Any:
        result = deconv2x_ncdhw(
            x.transpose(0, 4, 1, 2, 3), self.w[f"{name}.weight"], None, mx=self.mx
        )
        return result.transpose(0, 2, 3, 4, 1)

    def bilinear2x(self, x: Any) -> Any:
        """Source HoVer-Net uses align_corners=True at every 2x decoder stage."""
        mx = self.mx
        for axis in (1, 2):
            size = x.shape[axis]
            coordinate = mx.arange(size * 2, dtype=mx.float32) * ((size - 1) / (size * 2 - 1))
            low = mx.floor(coordinate).astype(mx.int32)
            high = mx.minimum(low + 1, size - 1)
            fraction = coordinate - low.astype(mx.float32)
            shape = [1] * x.ndim
            shape[axis] = size * 2
            fraction = fraction.reshape(shape)
            x = mx.take(x, low, axis=axis) * (1 - fraction) + mx.take(x, high, axis=axis) * fraction
        return x

    def require_input(self, x: Any, shape: tuple[int, ...]) -> None:
        if x.ndim != len(shape) + 1 or tuple(x.shape[1:]) != shape or x.shape[0] < 1:
            raise ValueError(f"Expected Bx{'x'.join(map(str, shape))} input; got {x.shape}")
        if x.dtype != self.mx.float32:
            raise ValueError("Pinned MONAI MLX graphs require float32 inputs")
