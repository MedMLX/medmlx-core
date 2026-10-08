"""Explicit MLX host readiness. Never a silent fallback."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

from medmlx_core.errors import MissingDependencyError

MLX_EXTRA = "mlx"
_UNSUPPORTED_HINT = (
    "Install medmlx-core on macOS Apple Silicon, or medmlx-core[cpu], "
    "medmlx-core[cuda12], or medmlx-core[cuda13] on Linux x86_64, "
    "and request --device mlx."
)


@dataclass(frozen=True, slots=True)
class MlxHostReport:
    """Facts required before an explicit native MLX run may start."""

    available: bool
    mlx_version: str | None
    macos_version: str | None
    apple_chip: str | None
    memory_bytes: int | None
    python_version: str
    platform_system: str
    platform_machine: str
    reason: str | None
    hint: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "available": self.available,
            "mlx_version": self.mlx_version,
            "macos_version": self.macos_version,
            "apple_chip": self.apple_chip,
            "memory_bytes": self.memory_bytes,
            "python_version": self.python_version,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
            "reason": self.reason,
            "hint": self.hint,
        }


def probe_mlx_runtime() -> MlxHostReport:
    """Report MLX availability without selecting a fallback runtime."""

    system = platform.system()
    machine = platform.machine()
    python_version = platform.python_version()
    macos_version = _macos_version(system)
    apple_chip = _apple_chip(system, machine)
    memory_bytes = _memory_bytes(system)
    mlx_version, import_reason = _mlx_version()
    if (system, machine) not in {("Darwin", "arm64"), ("Linux", "x86_64")}:
        return MlxHostReport(
            available=False,
            mlx_version=mlx_version,
            macos_version=macos_version,
            apple_chip=apple_chip,
            memory_bytes=memory_bytes,
            python_version=python_version,
            platform_system=system,
            platform_machine=machine,
            reason=(
                "MLX inference requires macOS on Apple Silicon or Linux x86_64; "
                f"this host is {system} {machine}"
            ),
            hint=_UNSUPPORTED_HINT,
        )
    if mlx_version is None:
        return MlxHostReport(
            available=False,
            mlx_version=None,
            macos_version=macos_version,
            apple_chip=apple_chip,
            memory_bytes=memory_bytes,
            python_version=python_version,
            platform_system=system,
            platform_machine=machine,
            reason=import_reason or "mlx is not installed",
            hint=_UNSUPPORTED_HINT,
        )
    metal_ok, metal_reason = (
        _metal_available() if system == "Darwin" else _linux_backend_available()
    )
    if not metal_ok:
        return MlxHostReport(
            available=False,
            mlx_version=mlx_version,
            macos_version=macos_version,
            apple_chip=apple_chip,
            memory_bytes=memory_bytes,
            python_version=python_version,
            platform_system=system,
            platform_machine=machine,
            reason=metal_reason or "MLX Metal GPU is not available",
            hint=_UNSUPPORTED_HINT,
        )
    return MlxHostReport(
        available=True,
        mlx_version=mlx_version,
        macos_version=macos_version,
        apple_chip=apple_chip,
        memory_bytes=memory_bytes,
        python_version=python_version,
        platform_system=system,
        platform_machine=machine,
        reason=None,
        hint=None,
    )


def require_mlx_runtime() -> MlxHostReport:
    """Return a ready MLX host report or fail with an actionable error."""

    report = probe_mlx_runtime()
    if report.available:
        return report
    raise MissingDependencyError(
        report.reason or "MLX is not available on this host",
        extra=MLX_EXTRA,
        hint=report.hint or _UNSUPPORTED_HINT,
    )


def require_mlx_device(device: str) -> MlxHostReport:
    """Accept only an explicit mlx device and fail if the host cannot run it."""

    requested = device.strip().casefold()
    if requested != "mlx":
        raise MissingDependencyError(
            f"An MLX execution route requires an explicit mlx device; got {device!r}",
            extra=MLX_EXTRA,
            hint="Pass --device mlx. There is no Torch, MPS, NumPy, or CPU fallback.",
        )
    return require_mlx_runtime()


def _load_mlx_core() -> Any:
    from importlib import import_module

    return import_module("mlx.core")


def _mlx_version() -> tuple[str | None, str | None]:
    try:
        mx = _load_mlx_core()
    except ImportError:
        return None, "mlx is not installed"
    version = getattr(mx, "__version__", None)
    if isinstance(version, str) and version.strip():
        return version, None
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    try:
        return package_version("mlx"), None
    except PackageNotFoundError:
        return None, "mlx is installed but reports no version"


def _macos_version(system: str) -> str | None:
    if system != "Darwin":
        return None
    version = platform.mac_ver()[0]
    return version or None


def _apple_chip(system: str, machine: str) -> str | None:
    if system != "Darwin" or machine != "arm64":
        return None
    try:
        import subprocess

        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return machine
    brand = completed.stdout.strip()
    return brand or machine


def _memory_bytes(system: str) -> int | None:
    if system != "Darwin":
        return None
    try:
        import subprocess

        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    raw = completed.stdout.strip()
    if not raw.isdigit():
        return None
    return int(raw)


def import_mlx() -> Any:
    """Select Metal on Darwin; retain the configured CPU/CUDA backend on Linux."""

    report = require_mlx_runtime()
    mx = _load_mlx_core()
    if report.platform_system == "Linux":
        return mx
    if not hasattr(mx, "metal") or not mx.metal.is_available():
        raise MissingDependencyError(
            "MLX Metal GPU is not available; refusing a CPU or host fallback",
            extra=MLX_EXTRA,
            hint=_UNSUPPORTED_HINT,
        )
    mx.set_default_device(mx.gpu)
    current = str(mx.default_device()).casefold()
    if "gpu" not in current:
        raise MissingDependencyError(
            f"MLX default device is {mx.default_device()!s}, not GPU",
            extra=MLX_EXTRA,
            hint=_UNSUPPORTED_HINT,
        )
    return mx


def mlx_default_device_name(mx: Any) -> str:
    """Return the live MLX default-device string."""

    return str(mx.default_device())


def reset_mlx_peak_memory(mx: Any) -> None:
    """Reset the runtime-native MLX peak-memory counter when available."""

    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()
        return
    metal = getattr(mx, "metal", None)
    metal_reset = None if metal is None else getattr(metal, "reset_peak_memory", None)
    if callable(metal_reset):
        metal_reset()


def mlx_peak_memory_bytes(mx: Any) -> int | None:
    """Return the runtime-native peak-memory count in bytes when available."""

    getter = getattr(mx, "get_peak_memory", None)
    if not callable(getter):
        metal = getattr(mx, "metal", None)
        getter = None if metal is None else getattr(metal, "get_peak_memory", None)
    if not callable(getter):
        return None
    value = getter()
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise RuntimeError("MLX peak-memory counter did not return a non-negative number")
    return int(value)


def _metal_available() -> tuple[bool, str | None]:
    try:
        mx = _load_mlx_core()
    except ImportError:
        return False, "mlx is not installed"
    if not hasattr(mx, "metal") or not mx.metal.is_available():
        return False, "MLX Metal GPU is not available; refusing a CPU or host fallback"
    return True, None


def _linux_backend_available() -> tuple[bool, str | None]:
    try:
        mx = _load_mlx_core()
    except ImportError:
        return False, "mlx is not installed"
    if not mx.is_available(mx.default_device()):
        return False, "MLX configured CPU or CUDA backend is not available"
    return True, None


__all__ = [  # noqa: RUF022 -- retain the snapshot export order
    "MLX_EXTRA",
    "MlxHostReport",
    "import_mlx",
    "mlx_peak_memory_bytes",
    "mlx_default_device_name",
    "probe_mlx_runtime",
    "reset_mlx_peak_memory",
    "require_mlx_device",
    "require_mlx_runtime",
]
