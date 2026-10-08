# medmlx-core

Shared MLX runtime for the [MedMLX](https://github.com/MedMLX) model packages: device selection, precision control, and checkpoint loading.

**Status:** Empty skeleton. The runtime is moving here from RadNN (`radnn/runtime/mlx.py` and `radnn/runtime/mlx_precision.py`).

Code belongs here only when two or more model packages use it. Model-specific code stays in the model's own repository.

## Install

    uv pip install "medmlx-core @ git+https://github.com/MedMLX/medmlx-core.git"

On Apple Silicon nothing else is needed. On Linux, MLX also needs a backend, so add one of the `cpu`, `cuda12`, or `cuda13` extras:

    uv pip install "medmlx-core[cpu] @ git+https://github.com/MedMLX/medmlx-core.git"

## License

Proprietary until release review.
