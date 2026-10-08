"""Host input admission rejects malformed arrays before touching an MLX backend."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from fixture_cases import array_cases

ARRAYS, CASES = array_cases()
HOST_CASES = [
    (name, case)
    for name, cases in CASES.items()
    for case in cases
    if case.error is not None and case.function != "group_norm_ncdhw"
]


@pytest.mark.parametrize(
    "name,case", HOST_CASES, ids=[f"{name}.{case.id}" for name, case in HOST_CASES]
)
def test_invalid_input_fails_before_backend_access(name, case):
    module = importlib.import_module(f"medmlx_core.{name}")
    args = [None if key is None else ARRAYS[key] for key in case.args]
    kwargs = dict(case.kwargs)
    if case.function != "require_ncdhw":
        # Only the dtype tag is supplied. Any numerical backend access would fail.
        kwargs["mx"] = SimpleNamespace(float32=np.float32)
    with pytest.raises(case.error) as captured:
        getattr(module, case.function)(*args, **kwargs)
    assert str(captured.value) == case.error_message


def test_group_norm_rank_admission_precedes_backend_access():
    from medmlx_core import group_norm_ncdhw

    with pytest.raises(ValueError, match="NCDHW rank-5"):
        group_norm_ncdhw(
            ARRAYS["bad_rank"],
            ARRAYS["norm_weight"],
            ARRAYS["norm_bias"],
            num_groups=1,
            eps=1e-5,
            mx=SimpleNamespace(),
        )
