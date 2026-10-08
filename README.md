# medmlx-core

Shared runtime and MLX primitives for standalone MedMLX medical imaging models.
The package runs without RadNN, Torch, or MONAI. Torch is loaded only when the
optional checkpoint converter is called.

**Status:** Shared helpers extracted from the frozen RadNN snapshot `cbaa1ac`.
Portable numerical fixtures match bitwise on MLX 0.32.3 CPU. Metal precision
kernels and fused decoder interpolation are preserved, but numerical equivalence
on Apple Silicon is pending; Linux fixtures record their original backend errors.
No model architectures or pretrained weights are included.

## Install and platform policy

```sh
uv pip install "medmlx-core @ git+https://github.com/MedMLX/medmlx-core.git"
uv pip install "medmlx-core[cpu] @ git+https://github.com/MedMLX/medmlx-core.git"
```

macOS arm64 uses Metal. Linux x86_64 uses the configured MLX CPU or CUDA backend;
install `cpu`, `cuda12`, or `cuda13` as appropriate. Linux probing retains all report
fields, with `macos_version`, `apple_chip`, and `memory_bytes` set to `None`.
`import_mlx()` selects the Metal GPU on macOS and retains MLX's selected device on
Linux. It rejects an unavailable selected backend. `require_mlx_device()` still
requires the explicit name `mlx`. There is no automatic backend fallback.

The `conversion` extra supplies Torch for `load_torch_checkpoint()`. Its original
`weights_only=False` default and error metadata (`extra="models"`) are retained.
MLX stays pinned to `>=0.32.2,<0.33`; Python is `>=3.12,<3.14`.

## Public API and source mapping

Main entry points are also re-exported from `medmlx_core`. Source paths below are
relative to `/home/alif/Documents/GitHub/.medmlx-extract/radnn-cbaa1ac/`;
destinations are relative to this repository. Counts include all lines.

| Frozen source (lines) | Destination (lines) | Public API |
| --- | --- | --- |
| `radnn/runtime/mlx.py` (285) | `src/medmlx_core/runtime.py` (300) | `MlxHostReport`, `MLX_EXTRA`, `probe_mlx_runtime`, `require_mlx_runtime`, `require_mlx_device`, `import_mlx`, `mlx_default_device_name`, `reset_mlx_peak_memory`, `mlx_peak_memory_bytes` |
| `radnn/runtime/mlx_precision.py` (317) | `src/medmlx_core/precision.py` (317) | `Float32Operators` (Metal only) |
| `radnn/runtime/checkpoints.py` (79) | `src/medmlx_core/checkpoints.py` (63), `conversion.py` (28) | `mapping_from_pairs`, `tensor_mapping_from_payload`, `_maybe_array`, `load_torch_checkpoint` |
| `radnn/integrations/nv_generate/mlx/ops.py` (287) | `src/medmlx_core/ops.py` (287) | `as_fp32`, `silu`, `linear`, NCDHW convolution, split convolution, group normalization, pooling, interpolation, padding and concatenation |
| `radnn/integrations/nv_generate/mlx/layout.py` (73) | `src/medmlx_core/layout.py` (73) | Layout constants, rank validation, NCDHW/NDHWC, convolution-weight and token remaps |
| `radnn/engines/mlx_segresnet/upsample.py` (146) | `src/medmlx_core/upsample.py` (146) | `deconv2x_ncdhw` (portable), `upsample_add_ncdhw` (Metal only) |
| `radnn/errors.py` (123; selected classes) | `src/medmlx_core/errors.py` (84) | `RadnnError` (`MedmlxError` alias), `InvalidInputError`, `MissingDependencyError`, `ModelExecutionError`, `AssetNotReadyError` |

The 73-file import scan covered every requested integration, engine, and MAISI
contract glob. Core supplies all imported MLX runtime/precision/checkpoint names
and the four shared exception subclasses. `radnn._shared.json` remains in RadNN:
its `JsonObject` and `load_json_object` consumers handle bundle/asset records, and
none of the extracted modules needs it.

Model-specific state-dict wrappers, inventories, key renaming, scale factors,
architectures, and converters belong to individual model packages. Core includes
the generic array extraction, duplicate-key handling, Torch deserialization and
convolution weight remaps they reuse. Asset staging, readiness, registry, run
records, package inputs, artifact loading, QC and DICOM/NIfTI I/O remain in RadNN.

## Equivalence fixtures

The committed fixtures contain fixed seeded inputs/weights (`seed=7081`), RadNN
outputs/errors, executable definitions, kernel sources and `radnn_commit=cbaa1ac`.
Each archive is under 80 KB. Tests use `np.array_equal`, check dtypes and shapes,
and run without access to RadNN. There are no tolerance-based comparisons.
Simulated Darwin arm64 reports are compared with RadNN; the real Linux host and
simulated CPU/CUDA backend selection are tested separately. CUDA execution is
pending hardware validation.

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/alif/Documents/GitHub/.medmlx-extract/radnn-cbaa1ac \
/home/alif/Documents/GitHub/.medmlx-extract/env/bin/python scripts/make_reference_fixtures.py

ruff check .
CORE_SRC="$PWD/src"
PYTHONPATH=src:$CORE_SRC /home/alif/Documents/GitHub/.medmlx-extract/env/bin/python -m pytest -q
```

The reference generator imports only the frozen snapshot. Numerical helpers take
`mx` explicitly, so no runtime gate or numerical implementation is monkeypatched.
To qualify Metal outputs, regenerate and run fixtures on Apple Silicon with the
same MLX version. Existing Linux fixtures establish kernel/source and failure
parity, not Metal numerical output parity. Fixtures are synthetic, not clinical
or pretrained-model validation.

## RadNN migration (not performed)

Add `medmlx-core` as a dependency and preserve backend extras in RadNN's packaging.
Replace import prefixes as follows, keeping symbol names:

| RadNN import prefix | Replacement |
| --- | --- |
| `radnn.runtime.mlx` | `medmlx_core.runtime` |
| `radnn.runtime.mlx_precision` | `medmlx_core.precision` |
| `radnn.runtime.checkpoints` | `medmlx_core.checkpoints` |
| `radnn.integrations.nv_generate.mlx.ops` | `medmlx_core.ops` |
| `radnn.integrations.nv_generate.mlx.layout` | `medmlx_core.layout` |
| `radnn.engines.mlx_segresnet.upsample` | `medmlx_core.upsample` |

In `radnn.errors`, import/re-export core's `RadnnError` and four extracted
subclasses, and retain the remaining RadNN subclasses deriving from that imported
base. This preserves identity for existing exception handlers. Remove the copied
implementations or replace them with compatibility re-exports. Keep model-specific
and orchestration imports, including `radnn._shared.json`. Linux support belongs
to core's host policy; Metal-only functions retain their original limitation.

## License

Proprietary until release review.
