from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest):
    """Keep test-generated checkpoints inside the authorized workspace."""
    from tempfile import TemporaryDirectory

    root = Path(__file__).parent / ".scratch"
    root.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=root, prefix=request.node.name[:40]) as directory:
        yield Path(directory)
    if not any(root.iterdir()):
        root.rmdir()
