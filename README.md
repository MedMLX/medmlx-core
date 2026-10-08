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

The `conversion` extra supplies Torch for `load_torch_checkpoint()`. Safe
`weights_only=True` loading is the default. Explicit `weights_only=False` remains
available for trusted legacy payloads; error metadata retains `extra="models"`.
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

## Sliding-window inference

MONAI-compatible sliding-window inference with no torch or MONAI at runtime. Algorithms derive from MONAI 1.6.0; see `THIRD_PARTY_NOTICES.md`.

Runtime
dependencies are only NumPy and MLX. Neither RadNN, Torch, nor MONAI is imported
by the runtime module. No networks or weight converters are moved by this task;
their seeded-model and synthetic conversion tests belong to the model extractions.
No `conversion` extra is needed here.

```python
from medmlx_core.sliding_window import sliding_window_inference

logits = sliding_window_inference(
    image,  ### NumPy or MLX (N, C, *spatial), float32/float16
    roi_size=(128, 128, 128),
    sw_batch_size=1,
    predictor=network,  ### MLX array -> MLX array, same spatial resolution
    overlap=0.25,
    mode="constant",
    padding_mode="replicate",
)
```

The result is a host NumPy array with the input dtype and original spatial size;
output channel count follows the predictor. Public helpers are
`dense_patch_slices` and `compute_importance_map`. Scalar/per-axis overlap and
sigma, fallback ROI components, all four Torch padding modes, supplied ROI maps,
window batching across images, predictor arguments, and `with_coord` are supported.
`sw_device` accepts an MLX device or `"cpu"`/`"gpu"`; None uses MLX's default.
`device` is None or `"cpu"` because accumulation stays on the host.

Window stride is `max(int(roi * (1-overlap)), 1)` (ROI-sized when image equals
ROI), with the last window shifted to the edge and the last spatial axis varying
fastest. Small images receive symmetric padding, with the extra voxel on the
right. Gaussian maps use FP32 separable factors centered at `(size-1)/2`, sigma
`size*sigma_scale`, no peak normalization, and a minimum weight of 0.001.
Patches are weighted in predictor dtype before ordered input-dtype accumulation;
the count map is accumulated once per spatial window and used for normalization.

Multiple/scaled outputs, positive `buffer_steps`, `process_fn`, progress bars,
and MetaTensor metadata are outside this extraction and fail explicitly where
applicable. `buffer_steps=None`/nonpositive values use unbuffered MONAI semantics.
NV's bounded rolling label accumulator and interactive correction inferer are
not replaced by this full-volume logit implementation.

#### Route settings

The three staged `configs/inference.json` settings below were supplied by the
user; the staged files are absent from the frozen snapshot. Unspecified options
are MONAI defaults. NV settings were read from frozen RadNN cbaa1ac
`integrations/nv_segment_ct/mlx_worker.py:78-79,103-105` and `streaming.py:75-117`.

| Route | ROI | Configured batch | Overlap | Blend | Padding |
| --- | --- | --- | --- | --- | --- |
| `brats_mri_segmentation` (SegResNet) | 240 × 240 × 160 | 1 | 0.5 | constant | constant |
| `renalStructures_CECT_segmentation` (SegResNet) | 96³ | 4 | 0.25 | constant | constant |
| `renalStructures_UNEST_segmentation` (UNesT) | 96³ | 4 | 0.5 | constant | constant |
| NV automatic / class prompt | 128³ | 1 | 0.25 | constant | replicate |

All use cval 0, sigma .125 (unused for constant blending), and no MONAI
`buffer_steps`. Frozen `engines/mlx_segresnet/bundle.py:53-55` and
`integrations/renal_unest/mlx_bundle.py:44-46` override the configured inferer to
batch size 1 and `sw_device=device=cpu`, with AMP disabled. Both configured
batch sizes and these overrides are covered by parity tests. The MLX network
uses its default device independently of these Torch host-device settings.

NV interactive correction calls a separate `point_based_window_inferer`, with
ROI read from its bundle, `center_only=True`, `transpose=True`, previous logits
±1, and connected-component combination (`correction_worker.py:92-127`,
`mlx_point_window.py`). That path does not call ordinary sliding-window inference.

The nnunet-mlx implementation was read for MLX evaluation/host-export patterns;
its redistributed window steps and peak-normalized Gaussian were not adopted.

#### Sources and what stays in RadNN

No source file is moved wholesale. `src/medmlx_core/sliding_window.py` ports
MONAI 1.6.0 `monai/inferers/utils.py` window inference and scan intervals, plus
`monai/data/utils.py` dense windows and importance maps. Independent parity tests
live in `tests/test_sliding_window.py`.

| Reference source / symbols | Destination | Lines |
| --- | --- | --- |
| MONAI `inferers/utils.py`: `sliding_window_inference`, `_get_scan_interval`; `data/utils.py`: `dense_patch_slices`, `compute_importance_map` | `src/medmlx_core/sliding_window.py` | 244 |
| Real MONAI parity, route settings, padding, maps, coordinates, dtype contracts | `tests/test_sliding_window.py` | 318 |
| Frozen RadNN `integrations/nv_segment_ct/streaming.py`: `rolling_logits` (reference only) | `scripts/make_reference_fixtures.py` | 95 |

`scripts/make_reference_fixtures.py`
imports frozen `radnn/integrations/nv_segment_ct/streaming.py:75-184` to record
small constant/Gaussian score and label fixtures with seeded linear predictors,
weights, inputs, coordinates, and `radnn_commit=cbaa1ac` metadata. No Apple runtime
gate monkeypatch is needed: this reference module runs entirely on Torch CPU.

Asset staging, checkpoint loading, readiness, registry, telemetry/run records,
package inputs, QC, DICOM/NIfTI I/O, transforms, label reduction, and interactive
connected-component correction stay in RadNN because they are orchestration or
model-specific operations, not the shared logit stitcher.

#### RadNN integration changes needed (not performed)

1. Import `sliding_window_inference` from `medmlx_core.sliding_window` in a RadNN
   inferer adapter. Convert its CPU Torch input once with
   `inputs.detach().cpu().numpy()`, pass the model's MLX callback, then wrap the
   resulting logits with `torch.from_numpy` for the existing evaluator/transforms.
2. In SegResNet `bundle.py` and UNesT `mlx_bundle.py`, replace the MONAI inferer
   target with that RadNN adapter, passing the staged ROI/blending/padding options.
   Split `BundlePredictor._forward_window` so the MLX callback retains validation,
   telemetry, evaluation, synchronization, and cache handling but accepts/returns
   MLX arrays; `UNesTPredictor` inherits that callback. Calling the present Torch
   `forward` directly as the new predictor would violate the MLX API.
3. In NV `mlx_worker.py`, replace the `rolling_logits` import/call with this
   function using the confirmed options above and an MLX-returning measured
   predictor. Preserve first-index argmax in `class_ids` order and the `max <= 0`
   background rule before existing postprocessing. This allocates full logits;
   retain the existing rolling path when its bounded memory behavior is required.
4. Keep `mlx_point_window.py` and `correction_worker.py` on their point-window
   path until that separate MONAI connected-component algorithm is extracted.
   Ordinary sliding-window inference cannot replace that call by changing imports.

#### Validation

Use the supplied environment; no installation or network is needed:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/alif/Documents/GitHub/.medmlx-extract/env/bin/python scripts/make_reference_fixtures.py
PATH=/home/alif/Documents/GitHub/.medmlx-extract/env/bin:$PATH ruff check .
CORE_SRC=$PWD/src
PYTHONPATH=src:$CORE_SRC PYTHONDONTWRITEBYTECODE=1 /home/alif/Documents/GitHub/.medmlx-extract/env/bin/python -m pytest -q
```

Tests use real MONAI 1.6.0 and Torch CPU, not mocks. Constant reference cases,
padding, coordinates, and mixed-dtype cases assert bitwise equality. Gaussian
FP32 outputs assert maximum absolute difference <= 1e-6; direct maps <= 1e-7.
NumPy/Torch exponential kernels can round differently; supplying an identical
Gaussian map gives bitwise equality in the coordinate-dependent test. The
formerly failing test inadvertently inferred FP64 Torch offsets from MONAI's
NumPy-integer coordinates, while its MLX predictor used FP32. The resulting
FP64 weighting before FP32 accumulation caused the 5.96e-8 residual. Specifying
`dtype=patch.dtype` fixes the reference predictor; accumulation and normalization
are bitwise identical without changing runtime math. The fixtures' discrete
label maps match bitwise. Each route has odd overlapping and padded cases with
ROI scaled by 1/16, plus a cheap predictor run at its actual full ROI dimensions.
Apple GPU performance, production-volume memory use, and full model integration
are pending; only the supplied Linux MLX CPU backend is exercised here.

## Channel-last released-network layers

`medmlx_core.channel_last.ChannelLastGraph` supplies shared FP32 channel-last
convolution, projection, normalization, activation, bilinear interpolation and
3D transpose-convolution primitives for `swin-unetr-mlx`, `hovernet-mlx` and
`flexible-unet-mlx`. Model topology and checkpoint specifications stay in those
packages. Import this module directly; importing core still does not load MLX.

The arithmetic is extracted from the qualified October 7, 2026 uncommitted
RadNN `radnn/engines/mlx_monai/ops.py` working source. Only the import of the
existing transpose-convolution owner and the class name change. Its source
SHA-256 is recorded in `tests/fixtures/channel_last.npz`. Two independent
pre-extraction fixtures cover grouped 2D convolution, BatchNorm epsilon,
LayerNorm, Swish, source-aligned bilinear interpolation, 3D transpose convolution
and InstanceNorm. Both match bitwise on M1 Max Metal with MLX 0.32.2. The three
standalone released-checkpoint packages separately exercise this helper against
original MONAI 1.4.0 CPU outputs.

```bash
PYTHONPATH=src pytest -q tests/test_channel_last.py
python scripts/make_channel_last_fixtures.py --radnn-root /path/to/qualified/radnn \
  --output /path/to/fresh-fixture.npz
```

The existing full suite's older MLX 0.32.3 Linux CPU fixtures have 23 bitwise
failures on this MLX 0.32.2 Metal host. The same 23 failures reproduce on unchanged
core main `17ff628`; no tolerance or existing fixture was changed. Full core
cross-backend qualification remains open. These failures are separate from the
new helper's two passing Metal fixtures and the standalone network-window gates.

## License

Proprietary until release review.
The built wheel includes the complete upstream MONAI license and third-party
notices alongside the runtime modules.
