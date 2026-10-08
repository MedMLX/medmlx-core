"""Shared exceptions with RadNN-compatible names and builtin base classes."""

from __future__ import annotations


class RadnnError(Exception):
    """Base class for every error raised by the public radnn API."""

    def __str__(self) -> str:
        if not self.args:
            return ""
        if len(self.args) == 1:
            return str(self.args[0])
        return str(self.args)


class MissingDependencyError(RadnnError, ImportError):
    """Raised when an optional model dependency/extra is not installed.

    Args:
        message: Human-readable description of the missing dependency.
        extra: The pip extra that provides the dependency (e.g. ``"medsam2"``),
            or ``None`` when no single extra covers it.
        hint: An actionable install/setup hint, e.g.
            ``"Install the 'medsam2' extra (...)"``.

    Attributes:
        extra (str | None): The pip extra carrying the dependency.
        hint (str | None): Suggested install/setup command.
    """

    def __init__(
        self,
        message: str,
        *,
        extra: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.extra = extra
        self.hint = hint


class ModelExecutionError(RadnnError, RuntimeError):
    """Raised when a model runner fails during execution.

    This signals a model/runtime failure, not a caller input error.
    """


class InvalidInputError(RadnnError, ValueError):
    """Raised when caller-supplied inputs are missing, malformed, or an unsupported combination."""


class AssetNotReadyError(RadnnError, FileNotFoundError):
    """Raised when a model's local assets/checkpoints are missing or not staged.

    Args:
        message: Human-readable description of the missing/unstaged asset.
        reason: Short machine-friendly reason, e.g.
            ``"missing checkpoints/nv_segment_ct/.../model.pt"``.
        hint: An actionable staging hint, e.g. the ``radnn ... stage`` command.

    Attributes:
        reason (str | None): Why the asset is not ready.
        hint (str | None): Suggested staging/setup command.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.hint = hint


MedmlxError = RadnnError

__all__ = ["AssetNotReadyError", "InvalidInputError", "MedmlxError",
           "MissingDependencyError", "ModelExecutionError", "RadnnError"]
