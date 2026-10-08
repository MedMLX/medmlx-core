# Numerical parity triage

The reviewer reported 35 failures, 159 passes and two skips on Metal. All 35
failures below use `assert_array_equal` for floating-point arithmetic performed
by different frameworks. They are bad assertions: source inspection confirms
the same upstream operation, shape, layout, parameters and crop semantics, but
FP32 reduction order, FMA use and transcendental implementations differ. This
classification does not establish that every resulting numerical error is small;
the replacement numerical gates must still pass on Metal.

The supplied log summary includes absolute errors of 1.1920929e-7,
4.7683716e-7 and 3.8146973e-6. Its relative maximum of 0.00015369 cannot set a
relative-only tolerance near zero. Per-case maxima and later graph stages were
not supplied. Acceptance budgets are declared in `numerical_contract.json`;
they preserve exact shape/dtype checks and reject all non-finite results. These
are budgets for the seeded helper tests, not a general model accuracy claim.

A host check recalculated 66 recorded outputs using the same pinned upstream
operations in float64 and rounded them to FP32. All satisfied their declared
budgets. Maximum absolute differences were 4.76837158e-6 for dot products,
4.76837158e-7 for normalization, 2.38418579e-7 for pointwise operations,
1.19209290e-7 for interpolation and 3.81469727e-6 across channel-last graphs.
This measures upstream FP32 rounding; it does not substitute for Metal testing.

| Failing test (relative to `tests/`) | Decision and real property |
| --- | --- |
| `test_channel_last.py::test_channel_last_seeded_graph_matches_upstream[2d]` | Rewrite: compare the final graph within its cumulative budget; isolate each layer on its upstream stage input. |
| `test_channel_last.py::test_channel_last_seeded_graph_matches_upstream[3d]` | Rewrite: deconvolution/InstanceNorm reductions need numerical, not bitwise, parity. |
| `test_channel_last.py::test_individual_layers_match_upstream[linear]` | Rewrite: FP32 dot products have different reduction/FMA order. |
| `test_channel_last.py::test_individual_layers_match_upstream[affine_ln]` | Rewrite: LayerNorm statistics and affine rounding differ. |
| `test_channel_last.py::test_individual_layers_match_upstream[volume]` | Rewrite: compare the upstream convolution within the dot-product budget. |
| `test_channel_last.py::test_residual_blocks_match_monai[residual_same]` | Rewrite: check the complete MONAI block within the graph budget. |
| `test_channel_last.py::test_residual_blocks_match_monai[residual_project]` | Rewrite: check the complete projected MONAI block within the graph budget. |
| `test_equivalence.py::test_upstream_outputs[ops.silu_1]` | Rewrite: SiLU has framework-specific exponential/sigmoid rounding. |
| `test_equivalence.py::test_upstream_outputs[ops.conv3d_ncdhw_3]` | Rewrite: biased FP32 convolution uses different accumulation order. |
| `test_equivalence.py::test_upstream_outputs[ops.conv_transpose3d_ncdhw_4]` | Rewrite: transpose-convolution products/FMA differ. |
| `test_equivalence.py::test_upstream_outputs[ops.conv3d_ncdhw_6]` | Rewrite: unbiased convolution needs bounded numerical parity. |
| `test_equivalence.py::test_upstream_outputs[ops.conv_transpose3d_ncdhw_7]` | Rewrite: unbiased transpose convolution needs bounded numerical parity. |
| `test_equivalence.py::test_upstream_outputs[ops.group_norm_ncdhw_8]` | Rewrite: one-group statistics reduce in different orders. |
| `test_equivalence.py::test_upstream_outputs[ops.group_norm_ncdhw_9]` | Rewrite: two-group statistics reduce in different orders. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv3d_ncdhw_11]` | Rewrite: MONAI split/crop semantics match; convolution rounding differs. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv_transpose3d_ncdhw_12]` | Rewrite: same axis-0 split/crop semantics, different arithmetic. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv3d_ncdhw_13]` | Rewrite: same axis-1 split/crop semantics, different arithmetic. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv_transpose3d_ncdhw_14]` | Rewrite: same axis-1 transpose split/crop semantics, different arithmetic. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv3d_ncdhw_15]` | Rewrite: same axis-2 split/crop semantics, different arithmetic. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv_transpose3d_ncdhw_16]` | Rewrite: same axis-2 transpose split/crop semantics, different arithmetic. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv3d_ncdhw_17]` | Rewrite: unsplit convolution has the same FP32 rounding issue. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv_transpose3d_ncdhw_18]` | Rewrite: unsplit transpose convolution has the same rounding issue. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv3d_ncdhw_19]` | Rewrite: stride-2 split/crop matches upstream; compare numerically. |
| `test_equivalence.py::test_upstream_outputs[ops.avg_pool3d_ncdhw_23]` | Rewrite: eight-value FP32 average reductions need a rounding budget. |
| `test_equivalence.py::test_upstream_outputs[ops.upsample_trilinear_ncdhw_27]` | Rewrite: separable interpolation changes weighted-sum order. |
| `test_equivalence.py::test_upstream_outputs[upsample.upsample_add_ncdhw_6]` | Rewrite: interpolated sums and the skip addition need a rounding budget. |
| `test_equivalence.py::test_upstream_outputs[precision.conv_1]` | Rewrite: Metal FP32 matrix accumulation differs from Torch CPU convolution. |
| `test_equivalence.py::test_upstream_outputs[precision.conv_2]` | Rewrite: the same issue applies without bias. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_3]` | Rewrite: FP32 Welford is compared to a NumPy float64 population reference. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_5]` | Rewrite: short Welford reductions need bounded mean/variance error. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_6]` | Rewrite: vector-tail moments need bounded mean/variance error. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_7]` | Rewrite: tiled moments need bounded mean/variance error. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_8]` | Rewrite: partial-tile moments need bounded mean/variance error. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_9]` | Rewrite: merged-tile moments need bounded mean/variance error. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_10]` | Rewrite: multiple merged tiles need bounded mean/variance error. |

No arithmetic code change is justified by these failures. The separate runtime
defect is acceptance of Linux/CPU/CUDA despite the Apple-Silicon-only contract.
Runtime admission and packaging must require macOS arm64/Metal. Fixture loading
must use the one `darwin-arm64` recording directory without backend selection.

Additional bad coverage identified before editing:

- `test_real_linux_runtime_and_execution`: delete; unsupported environment.
- `test_linux_retains_explicit_cpu_or_cuda_backend[cpu,gpu]`: delete; unsupported backends.
- `test_unknown_backend_fails_with_regeneration_hint`: replace with a missing-file contract; backend selection is outside the package contract.
- `test_reference_covers_every_numerical_case`: delete; duplicates the host fixture coverage test.
- Error cases in `test_upstream_outputs`, except GroupNorm divisibility: remove from this parametrization; host admission tests already exercise the exact same checks.
- `test_renal_deconv2x_matches_torch_repeated`: rewrite using seeded small-channel weights at the same 48-cubed spatial size; an external model checkpoint is an unnecessary environment assumption. Exact signs near zero are not a valid FP32 helper contract.
- `test_decoder_fusion_matches_torch_at_borders_and_singleton_axes`: retain its border/stride coverage, replace the loose 1e-4 budget with the interpolation budget.
- `test_scaled_window_settings_match_monai`: retain scaled ROI, batching and padding coverage; execute the MLX predictor on Metal.
- `test_full_roi_window_settings_match_monai`: retain full-size ROI coverage; execute the MLX predictor on Metal.
- `test_single_window_batch_matches_monai`: retain batch-size-one coverage; execute the MLX predictor on Metal without a bundle-specific override.
- `test_cached_gaussian_map_and_coordinate_predictor_are_bitwise_equal`: retain cached-map/coordinate coverage; execute the MLX predictor on Metal.
- `test_invalid_or_unimplemented_options_fail_before_prediction`: add CPU-device rejection; the host accumulator remains distinct from the predictor backend.
- `test_unsupported_hosts_are_rejected`: strengthen the rejection property; an unsupported host must never load MLX.
- `test_every_archive_has_current_provenance_and_is_small`: require every archive to be a small upstream reference in the single Mac directory.
- `test_fixture_metadata_round_trip`: round-trip schema 2 and the recording platform without backend selection.
- `test_fixture_parser_accepts_typed_objects_or_raises_value_error`: require the constructed metadata to describe the one supported recording platform.
- `test_import_does_not_load_mlx`: delete; its package-import check duplicates the blocked-framework test, while forced loading on attribute lookup is an incidental implementation detail.

Earlier extraction tests were also replaced by independent contracts:

- The original channel-last source-recording test now checks seeded PyTorch/MONAI layer outputs, full graphs and residual blocks.
- The original snapshot-output test is now `test_upstream_outputs`; it uses independently recorded upstream arrays with explicit numerical budgets.
- The original executable-definition test was deleted; textual identity to a recorded implementation says nothing about correct behavior.
- Synthetic checkpoint mapping and optional Torch checkpoint tests now compare the original seeded host arrays and actual Torch save/load round trips.
- Simulated Darwin readiness and peak-memory tests now check the documented host/API contract directly.
- The original error-recording test was replaced by `test_shared_error_base`, `test_medmlx_error_is_the_real_public_base` and `test_error_metadata`.
- `test_version` was replaced by `test_package_import_does_not_load_optional_frameworks`; a hardcoded version assertion adds no behavioral coverage.
- `test_runtime_works_with_torch_and_monai_blocked` retains actual helper execution with conversion-only dependency admission.
- `test_report_serializes_without_backend_objects` now isolates the report's JSON contract from the availability of a live Metal device.
- `test_recorded_monai_scores_coordinates_and_labels` replaces frozen bundle outputs with actual MONAI stitching, coordinates and channel-index argmax; bundle class remaps and thresholds are outside this core helper.
- `test_patch_dependent_blending_matches_monai`, `test_padding_values_and_crop_match_torch`, `test_predictor_weighting_precedes_accumulator_dtype_conversion` and `test_scalar_and_fallback_roi_match_monai` retain their properties; explicit tensor-type admission makes the upstream return contract clear.

Channel-last stage coverage now feeds each layer its upstream stage input so
its error is isolated from earlier layers. Full 2D/3D graphs and residual blocks
remain separate composition checks. The shared numerical gate has host tests
that reject wrong values/layout/dtype/shape, NaN/Inf and malformed budgets.

Pure layout, casts, copies, nearest-neighbor replication, padding, concatenation,
ReLU, coordinates and discrete labels retain exact comparison. Constant-window
fixtures also retain exact comparison because their seeded dyadic predictor
arithmetic is exactly representable and ordered host accumulation is the contract.
