"""Optional Torch checkpoint deserialization; install the conversion extra."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from medmlx_core.errors import InvalidInputError, MissingDependencyError


def load_torch_checkpoint(path: Path, *, weights_only: bool = True) -> dict[str, Any]:
    """Deserialize source weights on the host; requires Torch, without model execution."""

    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(
            "Checkpoint conversion requires torch to unpickle the source files",
            extra="conversion",
            hint="Install medmlx-core[conversion] and rerun the converter.",
        ) from exc
    payload = torch.load(path, map_location="cpu", weights_only=weights_only)
    if not isinstance(payload, dict):
        raise InvalidInputError(f"{path.name} is not a mapping checkpoint")
    return cast(dict[str, Any], payload)


__all__ = ["load_torch_checkpoint"]
