"""Shared MLX runtime for MedMLX model packages."""

from medmlx_core.checkpoints import (
    load_torch_checkpoint,
    mapping_from_pairs,
    tensor_mapping_from_payload,
)
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MedmlxError,
    MissingDependencyError,
    ModelExecutionError,
    RadnnError,
)
from medmlx_core.layout import (
    conv3d_weight_to_mlx,
    conv_transpose3d_weight_to_mlx,
    require_ncdhw,
    to_ncdhw,
    to_ndhwc,
    tokens_from_ncdhw,
    tokens_to_ncdhw,
)
from medmlx_core.ops import (
    as_fp32,
    avg_pool3d_ncdhw,
    concat_channels_ncdhw,
    conv3d_ncdhw,
    conv_transpose3d_ncdhw,
    group_norm_ncdhw,
    linear,
    pad_spatial_trailing_ncdhw,
    silu,
    split_conv3d_ncdhw,
    split_conv_transpose3d_ncdhw,
    upsample_nearest_ncdhw,
    upsample_trilinear_ncdhw,
)
from medmlx_core.precision import Float32Operators
from medmlx_core.runtime import (
    MLX_EXTRA,
    MlxHostReport,
    import_mlx,
    mlx_default_device_name,
    mlx_peak_memory_bytes,
    probe_mlx_runtime,
    require_mlx_device,
    require_mlx_runtime,
    reset_mlx_peak_memory,
)
from medmlx_core.sliding_window import (
    compute_importance_map,
    dense_patch_slices,
    sliding_window_inference,
)
from medmlx_core.upsample import deconv2x_ncdhw, upsample_add_ncdhw

__version__ = "0.0.0"

__all__ = [
    "MLX_EXTRA",
    "AssetNotReadyError",
    "Float32Operators",
    "InvalidInputError",
    "MedmlxError",
    "MissingDependencyError",
    "MlxHostReport",
    "ModelExecutionError",
    "RadnnError",
    "as_fp32",
    "avg_pool3d_ncdhw",
    "compute_importance_map",
    "concat_channels_ncdhw",
    "conv3d_ncdhw",
    "conv3d_weight_to_mlx",
    "conv_transpose3d_ncdhw",
    "conv_transpose3d_weight_to_mlx",
    "deconv2x_ncdhw",
    "dense_patch_slices",
    "group_norm_ncdhw",
    "import_mlx",
    "linear",
    "load_torch_checkpoint",
    "mapping_from_pairs",
    "mlx_default_device_name",
    "mlx_peak_memory_bytes",
    "pad_spatial_trailing_ncdhw",
    "probe_mlx_runtime",
    "require_mlx_device",
    "require_mlx_runtime",
    "require_ncdhw",
    "reset_mlx_peak_memory",
    "silu",
    "sliding_window_inference",
    "split_conv3d_ncdhw",
    "split_conv_transpose3d_ncdhw",
    "tensor_mapping_from_payload",
    "to_ncdhw",
    "to_ndhwc",
    "tokens_from_ncdhw",
    "tokens_to_ncdhw",
    "upsample_add_ncdhw",
    "upsample_nearest_ncdhw",
    "upsample_trilinear_ncdhw",
]
