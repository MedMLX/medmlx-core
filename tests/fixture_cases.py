"""Fixed inputs and call recipes; expected values come only from RadNN fixtures."""

from __future__ import annotations

import ast
import os
import platform
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SNAPSHOT = Path(
    os.environ.get(
        "MEDMLX_RADNN_SNAPSHOT", "/home/alif/Documents/GitHub/.medmlx-extract/radnn-cbaa1ac"
    )
)
# Bitwise outputs differ across backends by float32 rounding, so each backend has its
# own snapshot recording. The original Linux CPU recording sits at the fixture root.
REFERENCE_BACKEND = "linux-x86_64-cpu"


def backend_key(mx: Any) -> str:
    return f"{platform.system().lower()}-{platform.machine()}-{mx.default_device().type.name}"


SOURCES = {
    "runtime": "radnn/runtime/mlx.py",
    "precision": "radnn/runtime/mlx_precision.py",
    "checkpoints": "radnn/runtime/checkpoints.py",
    "layout": "radnn/integrations/nv_generate/mlx/layout.py",
    "ops": "radnn/integrations/nv_generate/mlx/ops.py",
    "upsample": "radnn/engines/mlx_segresnet/upsample.py",
    "errors": "radnn/errors.py",
}


def definitions(source: str) -> dict[str, str]:
    """Compare executable definitions, ignoring docstrings and import relocation."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
        ):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body.pop(0)
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }


def array_cases() -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    rng = np.random.default_rng(7081)

    def random(shape: tuple[int, ...]) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    arrays = {
        "x": random((1, 2, 8, 6, 4)),
        "small": random((1, 2, 1, 2, 3)),
        "w": random((3, 2, 3, 3, 3)),
        "wt": random((2, 3, 3, 3, 3)),
        "wd": random((2, 3, 2, 2, 2)),
        "bias": random((3,)),
        "norm_weight": random((2,)),
        "norm_bias": random((2,)),
        "linear_x": random((2, 4)),
        "linear_w": random((3, 4)),
        "bad_rank": random((2, 3)),
        "skip": random((1, 2, 2, 4, 6)),
        "half": random((1, 2, 1, 2, 3)).astype(np.float16),
    }
    cases: dict[str, list[dict[str, Any]]] = {}

    def add(module: str, function: str, args: list[str | None], **kwargs: Any) -> None:
        items = cases.setdefault(module, [])
        items.append(
            {"id": f"{function}_{len(items)}", "function": function, "args": args, "kwargs": kwargs}
        )

    add("layout", "require_ncdhw", ["x"], name="input")
    add("layout", "require_ncdhw", ["bad_rank"], name="input")
    add("layout", "to_ndhwc", ["x"])
    add("layout", "to_ncdhw", ["x"])
    add("layout", "conv3d_weight_to_mlx", ["w"])
    add("layout", "conv_transpose3d_weight_to_mlx", ["wt"])
    add("layout", "conv3d_weight_to_mlx", ["bad_rank"])
    add("layout", "conv_transpose3d_weight_to_mlx", ["bad_rank"])
    add("layout", "tokens_from_ncdhw", ["x"])
    # Use a fixed independently stored tokens input for the inverse mapping.
    arrays["tokens"] = random((1, 192, 2))
    add("layout", "tokens_to_ncdhw", ["tokens"], spatial=[1, 2, 8, 6, 4])
    add("ops", "as_fp32", ["half"])
    add("ops", "silu", ["x"])
    for bias in ("bias", None):
        add("ops", "linear", ["linear_x", "linear_w", bias])
        add("ops", "conv3d_ncdhw", ["x", "w", bias], padding=1)
        add(
            "ops",
            "conv_transpose3d_ncdhw",
            ["small", "wt", bias],
            padding=1,
            stride=2,
            output_padding=1,
        )
        add("upsample", "deconv2x_ncdhw", ["small", "wd", bias])
    for groups in (1, 2, 3):
        add(
            "ops",
            "group_norm_ncdhw",
            ["x", "norm_weight", "norm_bias"],
            num_groups=groups,
            eps=1e-5,
        )
    for dim in (0, 1, 2):
        add("ops", "split_conv3d_ncdhw", ["x", "w", "bias"], padding=1, num_splits=2, dim_split=dim)
        add(
            "ops",
            "split_conv_transpose3d_ncdhw",
            ["x", "wt", "bias"],
            padding=1,
            stride=2,
            output_padding=1,
            num_splits=2,
            dim_split=dim,
        )
    add("ops", "split_conv3d_ncdhw", ["x", "w", None], padding=1)
    add("ops", "split_conv_transpose3d_ncdhw", ["x", "wt", None], padding=1, output_padding=0)
    add(
        "ops",
        "split_conv3d_ncdhw",
        ["x", "w", None],
        padding=1,
        stride=2,
        num_splits=2,
        dim_split=0,
    )
    add("ops", "split_conv3d_ncdhw", ["x", "w", None], padding=1, num_splits=2, dim_split=3)
    add("ops", "split_conv3d_ncdhw", ["x", "w", None], padding=1, num_splits=20, dim_split=0)
    add("ops", "avg_pool3d_ncdhw", ["x"])
    add("ops", "avg_pool3d_ncdhw", ["x"], kernel_size=3, stride=1)
    for scale in (1, 2, 0):
        add("ops", "upsample_nearest_ncdhw", ["small"], scale=scale)
    add("ops", "upsample_trilinear_ncdhw", ["small"])
    add("ops", "pad_spatial_trailing_ncdhw", ["small"])
    add("ops", "concat_channels_ncdhw", ["x", "x"])
    add("upsample", "deconv2x_ncdhw", ["half", "wd", None])
    add("upsample", "deconv2x_ncdhw", ["small", "bad_rank", None])
    add("upsample", "deconv2x_ncdhw", ["small", "w", None])
    add("upsample", "deconv2x_ncdhw", ["small", "wd", "norm_bias"])
    add("upsample", "upsample_add_ncdhw", ["small", "skip"])
    add("upsample", "upsample_add_ncdhw", ["half", "skip"])
    add("upsample", "upsample_add_ncdhw", ["small", "x"])
    add("precision", "eps", [])
    add("precision", "conv", ["small", "w", "bias"], padding=1, stride=1)
    add("precision", "conv", ["small", "w", None], padding=[1, 1, 1], stride=[1, 1, 1])
    add("precision", "_moments", ["x"])
    add("precision", "norm", ["x", "norm_weight", "norm_bias"])
    return arrays, cases


def run_array_case(module: Any, case: dict[str, Any], arrays: Any, mx: Any) -> Any:
    args = [None if key is None else mx.array(arrays[key]) for key in case["args"]]
    kwargs = dict(case["kwargs"])
    if "spatial" in kwargs:
        kwargs["spatial"] = tuple(kwargs["spatial"])
    if hasattr(module, "Float32Operators"):
        instance = module.Float32Operators(mx, groups=1, eps=1e-5)
        if case["function"] == "eps":
            return instance.eps
        return getattr(instance, case["function"])(*args, **kwargs)
    if case["function"] != "require_ncdhw":
        kwargs["mx"] = mx
    return getattr(module, case["function"])(*args, **kwargs)


def outcome(call: Callable[[], Any], mx: Any | None = None) -> dict[str, Any]:
    try:
        value = call()
        if value is None:
            return {"kind": "none"}
        if mx is not None:
            mx.eval(value)
        if isinstance(value, tuple):
            return {"kind": "tuple", "arrays": [np.array(item) for item in value]}
        return {"kind": "array", "arrays": [np.array(value)]}
    except (ValueError, TypeError, RuntimeError, ImportError) as exc:
        return {"kind": "error", "type": type(exc).__name__, "message": str(exc)}


def peak_cases(module: Any) -> list[dict[str, Any]]:
    """Exercise both memory API generations and invalid backend counters."""
    results = []
    for route in ("native", "metal", "absent"):
        for value in (42, 42.75, 0, -1, True, "42"):
            events = []
            api = SimpleNamespace(
                reset_peak_memory=lambda e=events: e.append("reset"),
                get_peak_memory=lambda v=value: v,
            )
            mx = api if route == "native" else SimpleNamespace()
            if route == "metal":
                mx.metal = api
            module.reset_mlx_peak_memory(mx)
            result = outcome(lambda backend=mx: module.mlx_peak_memory_bytes(backend))
            result.pop("arrays", None)
            if result["kind"] == "array":
                result["value"] = module.mlx_peak_memory_bytes(mx)
            results.append({"route": route, "counter": value, "events": events, **result})
    return results


def error_contracts(module: Any) -> dict[str, Any]:
    errors = {
        "RadnnError": module.RadnnError("message", "detail"),
        "InvalidInputError": module.InvalidInputError("bad input"),
        "MissingDependencyError": module.MissingDependencyError(
            "dependency missing", extra="conversion", hint="install conversion"
        ),
        "ModelExecutionError": module.ModelExecutionError("execution failed"),
        "AssetNotReadyError": module.AssetNotReadyError(
            "asset absent", reason="missing weights", hint="stage weights"
        ),
    }
    return {
        name: {
            "str": str(exc),
            "args": list(exc.args),
            "attributes": vars(exc),
            "mro": [base.__name__ for base in type(exc).__mro__],
        }
        for name, exc in errors.items()
    }


def darwin_contracts(module: Any) -> dict[str, Any]:
    """Only platform/sysctl/backend boundaries are simulated, never readiness gates."""
    from unittest.mock import patch

    calls = []
    backend = SimpleNamespace(
        __version__="0.32.3",
        metal=SimpleNamespace(is_available=lambda: True),
        gpu="gpu",
        default_device=lambda: "Device(gpu, 0)",
        set_default_device=lambda device: calls.append(device),
    )

    def sysctl(command: list[str], **_kwargs: Any) -> Any:
        return SimpleNamespace(
            stdout="Apple M3\n" if "brand_string" in command[-1] else "17179869184\n"
        )

    with (
        patch.object(module.platform, "system", return_value="Darwin"),
        patch.object(module.platform, "machine", return_value="arm64"),
        patch.object(module.platform, "python_version", return_value="3.12.13"),
        patch.object(module.platform, "mac_ver", return_value=("15.0", ("", "", ""), "")),
        patch("subprocess.run", side_effect=sysctl),
        patch.object(module, "_load_mlx_core", return_value=backend),
    ):
        report = module.probe_mlx_runtime().to_payload()
        required = module.require_mlx_runtime().to_payload()
        device = module.require_mlx_device("  MLX ").to_payload()
        assert module.import_mlx() is backend
        invalid = outcome(lambda: module.require_mlx_device("cpu"))
        backend.metal.is_available = lambda: False
        missing_metal = module.probe_mlx_runtime().to_payload()
        missing_metal.pop("hint")  # installation hint is part of the platform policy change
        rejected = outcome(module.require_mlx_runtime)
        return {
            "probe": report,
            "required": required,
            "device": device,
            "selected": calls,
            "device_name": module.mlx_default_device_name(backend),
            "invalid_device": invalid,
            "missing_metal": missing_metal,
            "rejected": {key: value for key, value in rejected.items() if key != "arrays"},
        }
