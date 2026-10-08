"""Checkpoint admission rejects executable pickle objects before conversion."""

import pickle
from pathlib import Path
from typing import Any

import pytest

from medmlx_core.conversion import load_torch_checkpoint


class _ExecutableCheckpoint:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> Any:
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",)


def test_checkpoint_loader_rejects_pickle_execution(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "weights.pt"
    marker = tmp_path / "executed"
    torch.save({"payload": _ExecutableCheckpoint(marker)}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_torch_checkpoint(path)
    assert not marker.exists()
    torch.save({"model": {"weight": torch.tensor([2.0, 3.0])}}, path)
    assert load_torch_checkpoint(path)["model"]["weight"].tolist() == [2.0, 3.0]
