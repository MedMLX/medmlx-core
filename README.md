# medmlx-core

Shared MLX runtime for the [MedMLX](https://github.com/medmlx) model packages: device selection, precision control, and checkpoint loading.

**Status:** Empty skeleton. The runtime is moving here from RadNN (`radnn/runtime/mlx.py` and `radnn/runtime/mlx_precision.py`).

Code belongs here only when two or more model packages use it. Model-specific code stays in the model's own repository.

## Install

    uv pip install "medmlx-core @ git+https://github.com/medmlx/medmlx-core.git"

## License

Proprietary until release review.
