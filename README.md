# medmlx-core

Shared MLX building blocks for [MedMLX](https://github.com/MedMLX) medical imaging
models on Apple Silicon.

- Metal runtime checks and memory reporting
- 3D convolutions, resampling and sliding-window inference
- Tensor layouts, precision helpers and checkpoint conversion

[PyPI](https://pypi.org/project/medmlx-core/) ·
[Releases](https://github.com/MedMLX/medmlx-core/releases) ·
[Technical reference](https://github.com/MedMLX/medmlx-core/blob/main/docs/technical.md)

## Requirements

Apple Silicon Mac, macOS, Python 3.12 or 3.13, and Metal GPU access.
NumPy and MLX install automatically. Inference raises an error when Metal is
unavailable.

## Install

```bash
pip install medmlx-core
```

For optional PyTorch checkpoint conversion:

```bash
pip install "medmlx-core[conversion]"
```

## Python

Run a small synthetic volume through overlapping patches:

```python
import numpy as np
from medmlx_core import sliding_window_inference

volume = np.ones((1, 1, 32, 32, 32), dtype=np.float32)
output = sliding_window_inference(
    volume,
    roi_size=(16, 16, 16),
    sw_batch_size=1,
    predictor=lambda patch: patch * 2,
)
print(output.shape)  # (1, 1, 32, 32, 32)
```

Arrays use batch, channel, depth, height and width axes. Replace the example
predictor with your model; the result is a NumPy array.

MedMLX model packages provide the architectures, weights and image workflows.
This library supplies their shared runtime operations.

## Technical details

API contracts, numerical references and development commands are in the
[technical reference](https://github.com/MedMLX/medmlx-core/blob/main/docs/technical.md).
See the [decoder note](https://github.com/MedMLX/medmlx-core/blob/main/docs/transpose-convolution.md)
for the Metal transpose-convolution fix.

## License

[Apache-2.0](https://github.com/MedMLX/medmlx-core/blob/main/LICENSE).
Includes [third-party notices](https://github.com/MedMLX/medmlx-core/blob/main/THIRD_PARTY_NOTICES.md)
and the upstream MONAI license.
