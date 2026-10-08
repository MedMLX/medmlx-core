#!/usr/bin/env python3
"""Generate small golden fixtures from the frozen cbaa1ac snapshot, never core."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from fixture_cases import (  # noqa: E402
    REFERENCE_BACKEND,
    SNAPSHOT,
    SOURCES,
    array_cases,
    backend_key,
    darwin_contracts,
    definitions,
    error_contracts,
    outcome,
    peak_cases,
    run_array_case,
)

FIXTURES = REPO / "tests" / "fixtures"


def save(name: str, metadata: dict, arrays: dict | None = None) -> None:
    payload = {} if arrays is None else dict(arrays)
    payload["metadata"] = np.asarray(json.dumps({"radnn_commit": "cbaa1ac", **metadata}))
    np.savez_compressed(FIXTURES / f"{name}.npz", **payload)


def reference(name: str):
    source = SNAPSHOT / SOURCES[name]
    module_name = SOURCES[name].removesuffix(".py").replace("/", ".")
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != source:
        raise RuntimeError(f"Reference must come from frozen snapshot: {module.__file__}")
    return module


def main() -> None:
    global FIXTURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if str(SNAPSHOT) not in sys.path:
        sys.path.insert(0, str(SNAPSHOT))
    if (key := backend_key(mx)) != REFERENCE_BACKEND:
        FIXTURES = FIXTURES / key
    FIXTURES.mkdir(parents=True, exist_ok=True)
    modules = {name: reference(name) for name in SOURCES}
    arrays, cases = array_cases()
    for name in ("ops", "layout", "upsample", "precision"):
        module = modules[name]
        payload = dict(arrays)
        recorded = []
        for case in cases[name]:
            result = outcome(lambda c=case, m=module: run_array_case(m, c, arrays, mx), mx)
            values = result.pop("arrays", [])
            output_keys = []
            for index, value in enumerate(values):
                key = f"{case['id']}__output_{index}"
                payload[key] = value
                output_keys.append(key)
            recorded.append({**case, "expected": result, "outputs": output_keys})
        constants = {
            key: value
            for key, value in vars(module).items()
            if key in ("_SOURCE", "_SOURCES") or key.startswith("LAYOUT_")
        }
        if name == "layout":
            constants["IDENTITY_WEIGHT_LAYOUTS"] = sorted(module.IDENTITY_WEIGHT_LAYOUTS)
        save(
            name,
            {
                "seed": 7081,
                "mlx_version": "0.32.3",
                "reference_device": str(mx.default_device()),
                "cases": recorded,
                "definitions": definitions((SNAPSHOT / SOURCES[name]).read_text()),
                "constants": constants,
            },
            payload,
        )

    runtime = modules["runtime"]
    save(
        "runtime",
        {
            "darwin": darwin_contracts(runtime),
            "peak": peak_cases(runtime),
            "definitions": definitions((SNAPSHOT / SOURCES["runtime"]).read_text()),
        },
    )
    save(
        "errors",
        {
            "contracts": error_contracts(modules["errors"]),
            "definitions": definitions((SNAPSHOT / SOURCES["errors"]).read_text()),
        },
    )
    checkpoint = modules["checkpoints"]
    host = arrays["linear_w"].astype(np.float64)
    tensor = torch.from_numpy(host.copy())
    state = {"module.weight": tensor, "array": host, "epoch": 4, "description": "ignored"}
    mapped = checkpoint.tensor_mapping_from_payload(state, what="synthetic")
    pairs = checkpoint.mapping_from_pairs(list(mapped.items()), what="synthetic")
    with TemporaryDirectory(dir=FIXTURES) as directory:
        path = Path(directory) / "synthetic.pt"
        torch.save(state, path)
        loaded = checkpoint.load_torch_checkpoint(path)
        restored = checkpoint.tensor_mapping_from_payload(loaded, what="synthetic")
        torch.save({"module.weight": tensor}, path)
        safe = checkpoint.load_torch_checkpoint(path, weights_only=True)
        safe_array = checkpoint._maybe_array(safe["module.weight"])
    source_definitions = definitions((SNAPSHOT / SOURCES["checkpoints"]).read_text())
    save(
        "checkpoints",
        {"keys": list(mapped), "pair_keys": list(pairs), "definitions": source_definitions},
        {
            "host": host,
            **{f"mapped__{key}": value for key, value in mapped.items()},
            **{f"loaded__{key}": value for key, value in restored.items()},
            "safe": safe_array,
        },
    )

    # Numerical helpers take mx explicitly, so RadNN's host gate needs no bypass.
    for path in sorted(FIXTURES.glob("*.npz")):
        if path.stat().st_size >= 1_000_000:
            raise RuntimeError(f"Fixture exceeds 1 MB: {path}")
        print(f"{path.relative_to(REPO)}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
