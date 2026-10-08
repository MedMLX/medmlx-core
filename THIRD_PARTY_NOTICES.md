# Third-party notices

`src/medmlx_core/sliding_window.py` implements algorithms derived from MONAI
1.6.0 `monai/inferers/utils.py` and `monai/data/utils.py`.

Copyright (c) MONAI Consortium. Licensed under the Apache License, Version 2.0.
The original license is preserved in `licenses/MONAI_LICENSE`.
The NumPy/MLX implementation, restricted output/device API, and validation are
modifications made in this extraction and licensed under Apache-2.0;
the upstream MONAI portions retain their Apache license.

`src/medmlx_core/channel_last.py` retains layer arithmetic translated from
MONAI 1.4.0 for the source-pinned released network graphs. The new shared class
was extracted from RadNN's qualified working source; its model consumers are
SwinUNETR, FlexibleUNet and HoVer-Net. MONAI Consortium copyright and Apache-2.0
obligations apply; the full license remains in `licenses/MONAI_LICENSE`.
