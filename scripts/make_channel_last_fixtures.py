"""Capture independent channel-last graph outputs from a qualified RadNN source tree."""

import argparse
import hashlib
import importlib.util
import json
import sys
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radnn-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.radnn_root))
    source = args.radnn_root / "radnn/engines/mlx_monai/ops.py"
    spec = importlib.util.spec_from_file_location("original_monai_graph", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(20261007)
    arrays = {
        "input_2d": rng.normal(size=(1, 5, 7, 4)).astype(np.float32),
        "conv.weight": rng.normal(size=(6, 2, 3, 3)).astype(np.float32),
        "conv.bias": rng.normal(size=(6,)).astype(np.float32),
        "bn.weight": rng.normal(size=(6,)).astype(np.float32),
        "bn.bias": rng.normal(size=(6,)).astype(np.float32),
        "bn.running_mean": rng.normal(size=(6,)).astype(np.float32),
        "bn.running_var": rng.uniform(0.1, 1, size=(6,)).astype(np.float32),
        "input_3d": rng.normal(size=(1, 3, 4, 5, 2)).astype(np.float32),
        "deconv.weight": rng.normal(size=(2, 3, 2, 2, 2)).astype(np.float32),
    }
    graph = module.Graph(
        {key: mx.array(value) for key, value in arrays.items() if not key.startswith("input_")},
        mx=mx,
    )
    y = graph.conv(mx.array(arrays["input_2d"]), "conv", padding=1, groups=2)
    y = graph.swish(graph.batch_norm(y, "bn", eps=1e-3))
    y = graph.bilinear2x(graph.layer_norm(y))
    z = graph.instance_norm(graph.deconv3d(mx.array(arrays["input_3d"]), "deconv"))
    mx.eval(y, z)
    arrays["expected_2d"] = np.asarray(y)
    arrays["expected_3d"] = np.asarray(z)
    arrays["metadata"] = np.array(
        json.dumps(
            {
                "seed": 20261007,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "reference": "Qualified RadNN working source Graph before core extraction",
                "mlx_version": version("mlx"),
                "backend": str(mx.default_device()),
            }
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)


if __name__ == "__main__":
    main()
