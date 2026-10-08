"""Record the frozen RadNN rolling stitcher with small seeded patch predictors."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data.utils import compute_importance_map
from monai.inferers.utils import sliding_window_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radnn-snapshot", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures"
    )
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.radnn_snapshot))
    streaming = importlib.import_module("radnn.integrations.nv_segment_ct.streaming")
    if not Path(streaming.__file__).resolve().is_relative_to(args.radnn_snapshot.resolve()):
        raise RuntimeError("Reference implementation must come from the frozen RadNN snapshot")
    rng = np.random.default_rng(20261007)
    inputs = rng.integers(-64, 65, (1, 1, 11, 9, 7)).astype(np.float32) / 64
    weights = rng.integers(-4, 5, (1, 3, 1, 1, 1)).astype(np.float32) / 8
    bias = rng.integers(-4, 5, (1, 3, 1, 1, 1)).astype(np.float32) / 16
    roi, class_ids = (8, 6, 4), [9, 3, 117]
    padding = tuple(
        max(patch - size, 0) // 2 for size, patch in zip(inputs.shape[2:], roi, strict=True)
    )
    padded_shape = tuple(
        max(size, patch) for size, patch in zip(inputs.shape[2:], roi, strict=True)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("constant", "gaussian"):
        scores = np.empty((1, 3, *padded_shape), dtype=np.float32)
        coordinates = []

        def predict(patch, coords, coordinates=coordinates):
            coordinates.append([s.start for s in coords[0][2:]])
            return torch.from_numpy(patch.numpy() * weights + bias)

        def finalized(plane, values, count, scores=scores):
            scores[0, :, plane] = values.numpy()

        importance = compute_importance_map(roi, mode=mode)
        labels = streaming.rolling_logits(
            torch.from_numpy(inputs),
            predict,
            class_ids,
            roi,
            weight=importance,
            finalized=finalized,
        )
        crop = tuple(
            slice(before, before + size)
            for before, size in zip(padding, inputs.shape[2:], strict=True)
        )
        scores = scores[(slice(None), slice(None), *crop)]
        expected = sliding_window_inference(
            torch.from_numpy(inputs),
            roi,
            1,
            lambda x: torch.from_numpy(x.numpy() * weights + bias),
            mode=mode,
            padding_mode="replicate",
        )
        if not np.array_equal(scores, expected.numpy()):
            raise RuntimeError("RadNN rolling reference differs from real MONAI")
        np.savez_compressed(
            args.output_dir / f"radnn_rolling_{mode}.npz",
            radnn_commit="cbaa1ac",
            monai_version="1.6.0",
            seed=20261007,
            inputs=inputs,
            weights=weights,
            bias=bias,
            roi_size=roi,
            class_ids=np.array(class_ids, dtype=np.uint8),
            scores=scores,
            labels=labels.numpy(),
            coordinates=np.array(coordinates),
            mode=mode,
        )


if __name__ == "__main__":
    main()
