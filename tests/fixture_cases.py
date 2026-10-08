"""Canonical seeded cases and versioned upstream-reference fixture boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

RECORDING_PLATFORM = "darwin-arm64"
FIXTURES = Path(__file__).parent / "fixtures" / RECORDING_PLATFORM


@dataclass(frozen=True, slots=True)
class ArrayCase:
    id: str
    function: str
    args: tuple[str | None, ...]
    kwargs: tuple[tuple[str, Any], ...]
    error: type[Exception] | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    schema_version: int
    monai_version: str
    monai_revision: str
    torch_version: str
    torch_revision: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(f"Unsupported reference schema: {self.schema_version}")
        if any(
            not isinstance(value, str) or not value
            for value in (self.monai_version, self.torch_version)
        ):
            raise ValueError("Reference versions must be explicit strings")
        for revision in (self.monai_revision, self.torch_revision):
            if (
                not isinstance(revision, str)
                or len(revision) != 40
                or any(character not in "0123456789abcdef" for character in revision)
            ):
                raise ValueError(f"Invalid upstream revision: {revision}")

    @classmethod
    def from_json(cls, value: str) -> ReferenceSpec:
        payload = json.loads(value)
        if not isinstance(payload, dict) or payload.keys() != {field.name for field in fields(cls)}:
            raise ValueError("Reference spec must contain exactly the declared fields")
        return cls(**payload)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass(frozen=True, slots=True)
class RecordedCase:
    id: str
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FixtureMetadata:
    upstream: ReferenceSpec
    platform: str
    seed: int
    cases: tuple[RecordedCase, ...]
    reference_device: str
    numpy_version: str
    torch_build: str
    mode: str | None

    @classmethod
    def from_json(cls, value: str) -> FixtureMetadata:
        payload = json.loads(value)
        if (
            not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 2
        ):
            raise ValueError("Invalid fixture schema")
        recording_platform, seed = payload.get("platform"), payload.get("seed")
        if recording_platform != RECORDING_PLATFORM or type(seed) is not int:
            raise ValueError("Fixture must be recorded on darwin-arm64 with an explicit seed")
        upstream = ReferenceSpec.from_json(json.dumps(payload.get("upstream")))
        recorded = payload.get("cases", [])
        if not isinstance(recorded, list):
            raise ValueError("Fixture cases must be a list")
        provenance = tuple(
            payload.get(key) for key in ("reference_device", "numpy_version", "torch_build")
        )
        if any(not isinstance(value, str) or not value for value in provenance):
            raise ValueError("Fixture must declare its reference device and framework builds")
        mode = payload.get("mode")
        if mode not in (None, "constant", "gaussian"):
            raise ValueError("Invalid fixture blend mode")
        cases = []
        for case in recorded:
            if not isinstance(case, dict) or not isinstance(case.get("id"), str):
                raise ValueError("Fixture case must have an id")
            outputs = case.get("outputs")
            if not isinstance(outputs, list) or any(not isinstance(key, str) for key in outputs):
                raise ValueError("Fixture outputs must be array keys")
            cases.append(RecordedCase(case["id"], tuple(outputs)))
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("Fixture case ids must be unique")
        return cls(upstream, recording_platform, seed, tuple(cases), *provenance, mode)

    def to_json(self) -> str:
        payload = {"schema_version": 2, **asdict(self)}
        if self.mode is None:
            payload.pop("mode")
        return json.dumps(payload)


REFERENCE = ReferenceSpec.from_json((Path(__file__).parent / "reference_spec.json").read_text())


def load_fixture(name: str) -> tuple[FixtureMetadata, dict[str, np.ndarray]]:
    path = FIXTURES / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing upstream reference: {path}; run the README fixture commands"
        )
    with np.load(path, allow_pickle=False) as archive:
        metadata = FixtureMetadata.from_json(str(archive["metadata"]))
        arrays = {key: archive[key] for key in archive.files if key != "metadata"}
    if metadata.upstream != REFERENCE:
        raise ValueError(f"Fixture does not match the pinned upstream source: {path}")
    return metadata, arrays


def array_cases() -> tuple[dict[str, np.ndarray], dict[str, list[ArrayCase]]]:
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
    cases: dict[str, list[ArrayCase]] = {}

    def add(
        module: str,
        function: str,
        args: list[str | None],
        *,
        error: type[Exception] | None = None,
        error_message: str | None = None,
        **kwargs: Any,
    ) -> None:
        items = cases.setdefault(module, [])
        items.append(
            ArrayCase(
                f"{function}_{len(items)}",
                function,
                tuple(args),
                tuple(kwargs.items()),
                error,
                error_message,
            )
        )

    _add_layout_cases(add, arrays, random)
    _add_basic_cases(add)
    _add_split_cases(add)
    _add_resize_cases(add)
    _add_decoder_cases(add)
    _add_precision_cases(add, arrays, random)
    return arrays, cases


def _add_layout_cases(
    add: Callable[..., None],
    arrays: dict[str, np.ndarray],
    random: Callable[[tuple[int, ...]], np.ndarray],
) -> None:
    add("layout", "require_ncdhw", ["x"], name="input")
    add(
        "layout",
        "require_ncdhw",
        ["bad_rank"],
        name="input",
        error=ValueError,
        error_message="input must be NCDHW rank-5; got ndim=2",
    )
    add("layout", "to_ndhwc", ["x"])
    add("layout", "to_ncdhw", ["x"])
    add("layout", "conv3d_weight_to_mlx", ["w"])
    add("layout", "conv_transpose3d_weight_to_mlx", ["wt"])
    add(
        "layout",
        "conv3d_weight_to_mlx",
        ["bad_rank"],
        error=ValueError,
        error_message="conv3d weight must be rank-5 [O,I,K]; got ndim=2",
    )
    add(
        "layout",
        "conv_transpose3d_weight_to_mlx",
        ["bad_rank"],
        error=ValueError,
        error_message="conv_transpose3d weight must be rank-5 [I,O,K]; got ndim=2",
    )
    add("layout", "tokens_from_ncdhw", ["x"])
    # Use a fixed independently stored tokens input for the inverse mapping.
    arrays["tokens"] = random((1, 192, 2))
    add("layout", "tokens_to_ncdhw", ["tokens"], spatial=[1, 2, 8, 6, 4])


def _add_basic_cases(add: Callable[..., None]) -> None:
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
            error=ValueError if groups == 3 else None,
            error_message="group_norm channels 2 must be divisible by num_groups 3"
            if groups == 3
            else None,
        )


def _add_split_cases(add: Callable[..., None]) -> None:
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
    add(
        "ops",
        "split_conv3d_ncdhw",
        ["x", "w", None],
        padding=1,
        num_splits=2,
        dim_split=3,
        error=ValueError,
        error_message="dim_split must be 0, 1, or 2; got 3",
    )
    add(
        "ops",
        "split_conv3d_ncdhw",
        ["x", "w", None],
        padding=1,
        num_splits=20,
        dim_split=0,
        error=ValueError,
        error_message="split_conv3d length 8 is smaller than num_splits 20",
    )


def _add_resize_cases(add: Callable[..., None]) -> None:
    add("ops", "avg_pool3d_ncdhw", ["x"])
    add("ops", "avg_pool3d_ncdhw", ["x"], kernel_size=3, stride=1)
    for scale in (1, 2, 0):
        add(
            "ops",
            "upsample_nearest_ncdhw",
            ["small"],
            scale=scale,
            error=ValueError if scale == 0 else None,
            error_message="upsample scale must be >= 1; got 0" if scale == 0 else None,
        )
    add("ops", "upsample_trilinear_ncdhw", ["small"])
    add("ops", "pad_spatial_trailing_ncdhw", ["small"])
    add("ops", "concat_channels_ncdhw", ["x", "x"])


def _add_decoder_cases(add: Callable[..., None]) -> None:
    add(
        "upsample",
        "deconv2x_ncdhw",
        ["half", "wd", None],
        error=TypeError,
        error_message="SegResNet decoder deconvolution requires float32 inputs",
    )
    add(
        "upsample",
        "deconv2x_ncdhw",
        ["small", "bad_rank", None],
        error=ValueError,
        error_message="SegResNet decoder deconvolution weight must be rank 5",
    )
    add(
        "upsample",
        "deconv2x_ncdhw",
        ["small", "w", None],
        error=ValueError,
        error_message=(
            "SegResNet decoder deconvolution weight has shape (3, 2, 3, 3, 3); "
            "expected (2, 2, 2, 2, 2)"
        ),
    )
    add(
        "upsample",
        "deconv2x_ncdhw",
        ["small", "wd", "norm_bias"],
        error=ValueError,
        error_message="SegResNet decoder deconvolution bias must have shape (3,) and dtype float32",
    )
    add("upsample", "upsample_add_ncdhw", ["small", "skip"])
    add(
        "upsample",
        "upsample_add_ncdhw",
        ["half", "skip"],
        error=TypeError,
        error_message="SegResNet decoder fusion requires float32 inputs",
    )
    add(
        "upsample",
        "upsample_add_ncdhw",
        ["small", "x"],
        error=ValueError,
        error_message="Decoder skip shape (1, 2, 8, 6, 4) must equal (1, 2, 2, 4, 6)",
    )


def _add_precision_cases(
    add: Callable[..., None],
    arrays: dict[str, np.ndarray],
    random: Callable[[tuple[int, ...]], np.ndarray],
) -> None:
    add("precision", "eps", [])
    add("precision", "conv", ["small", "w", "bias"], padding=1, stride=1)
    add("precision", "conv", ["small", "w", None], padding=[1, 1, 1], stride=[1, 1, 1])
    add("precision", "_moments", ["x"])
    add("precision", "norm", ["x", "norm_weight", "norm_bias"])
    for length in (3, 4, 64, 68, 132, 260):
        key = f"moments_{length}"
        arrays[key] = random((1, 2, 1, 1, length))
        add("precision", "_moments", [key])


def run_array_case(module: Any, case: ArrayCase, arrays: Any, mx: Any) -> Any:
    args = [None if key is None else mx.array(arrays[key]) for key in case.args]
    kwargs = dict(case.kwargs)
    if "spatial" in kwargs:
        kwargs["spatial"] = tuple(kwargs["spatial"])
    if hasattr(module, "Float32Operators"):
        instance = module.Float32Operators(mx, groups=1, eps=1e-5)
        if case.function == "eps":
            return instance.eps
        return getattr(instance, case.function)(*args, **kwargs)
    if case.function != "require_ncdhw":
        kwargs["mx"] = mx
    return getattr(module, case.function)(*args, **kwargs)


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
        missing_metal.pop("hint")
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
