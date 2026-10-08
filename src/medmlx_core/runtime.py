"""Explicit MLX host readiness. Never a silent fallback."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

from medmlx_core.errors import MissingDependencyError

MLX_EXTRA = "mlx"
_UNSUPPORTED_HINT = (
    "Install medmlx-core on macOS Apple Silicon with Metal available, then call import_mlx()."
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
    supported = (system, machine) == ("Darwin", "arm64")
    mlx_version, import_reason = _mlx_version() if supported else (None, None)
    if not supported:
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
                "MLX inference requires macOS on Apple Silicon with Metal; "
                f"this host is {system} {machine}"
            ),
            hint=_UNSUPPORTED_HINT,
        )
    if mlx_version is None or import_reason is not None:
        return MlxHostReport(
            available=False,
            mlx_version=mlx_version,
            macos_version=macos_version,
            apple_chip=apple_chip,
            memory_bytes=memory_bytes,
            python_version=python_version,
            platform_system=system,
            platform_machine=machine,
            reason=import_reason or "mlx is not installed",
            hint=_UNSUPPORTED_HINT,
        )
    metal_ok, metal_reason = _metal_available()
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
            hint="Request the MLX runtime; no backend fallback is selected.",
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
        return _checked_mlx_version(version)
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    try:
        return _checked_mlx_version(package_version("mlx"))
    except PackageNotFoundError:
        return None, "mlx is installed but reports no version"


def _checked_mlx_version(version: str) -> tuple[str, str | None]:
    parts = version.split(".")
    if (
        len(parts) == 3
        and all(part.isdigit() for part in parts)
        and (0, 32, 3) <= tuple(int(part) for part in parts) < (0, 33, 0)
    ):
        return version, None
    return version, (
        f"MLX >=0.32.3,<0.33 is required; found {version}. Upgrade MLX before inference."
    )


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
    """Require macOS Apple Silicon and select the Metal GPU."""

    require_mlx_runtime()
    mx = _load_mlx_core()
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


__all__ = [
    "MLX_EXTRA",
    "MlxHostReport",
    "import_mlx",
    "mlx_default_device_name",
    "mlx_peak_memory_bytes",
    "probe_mlx_runtime",
    "require_mlx_device",
    "require_mlx_runtime",
    "reset_mlx_peak_memory",
]
