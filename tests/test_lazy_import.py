import subprocess
import sys


def test_import_does_not_load_mlx() -> None:
    code = (
        "import sys, medmlx_core; "
        "assert 'mlx.core' not in sys.modules, 'import medmlx_core loaded mlx.core'; "
        "medmlx_core.sliding_window_inference; "
        "assert 'mlx.core' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
