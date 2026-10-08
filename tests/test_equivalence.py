"""RadNN golden outputs and preserved executable definitions from cbaa1ac."""

import importlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from fixture_cases import definitions, outcome, run_array_case

FIXTURES = Path(__file__).parent / 'fixtures'


def load_fixture(name):
    with np.load(FIXTURES / f'{name}.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != 'metadata'}
        metadata = json.loads(str(archive['metadata']))
    assert metadata['radnn_commit'] == 'cbaa1ac'
    return metadata, arrays


ARRAY_CASES = [
    (module, case)
    for module in ('layout', 'ops', 'upsample', 'precision')
    for case in load_fixture(module)[0]['cases']
]


@pytest.mark.parametrize(('name', 'case'), ARRAY_CASES,
                         ids=[f'{name}.{case["id"]}' for name, case in ARRAY_CASES])
def test_snapshot_outputs(name, case):
    _, arrays = load_fixture(name)
    module = importlib.import_module(f'medmlx_core.{name}')
    actual = outcome(lambda: run_array_case(module, case, arrays, mx), mx)
    values = actual.pop('arrays', [])
    expected = case['expected']
    # These kernels require Metal. CPU fixtures record RadNN's real backend failure,
    # not emulated output. Their exact kernel sources are compared separately below.
    if expected.get('message') == '[metal_kernel] No Metal back-end.' and mx.metal.is_available():
        pytest.skip('Regenerate numerical Metal fixtures on Apple Silicon')
    assert actual == expected, f'{name}.{case["id"]}: return/error contract changed'
    assert len(values) == len(case['outputs'])
    for actual_array, key in zip(values, case['outputs'], strict=True):
        expected_array = arrays[key]
        assert actual_array.dtype == expected_array.dtype, key
        assert actual_array.shape == expected_array.shape, key
        assert np.array_equal(actual_array, expected_array), f'{name}.{key}: numerical drift'


@pytest.mark.parametrize('name', ['layout', 'ops', 'upsample', 'precision',
                                 'checkpoints', 'errors', 'runtime'])
def test_snapshot_executable_definitions(name):
    metadata, _ = load_fixture(name)
    module = importlib.import_module(f'medmlx_core.{name}')
    actual = definitions(Path(module.__file__).read_text())
    if name == 'checkpoints':
        conversion = importlib.import_module('medmlx_core.conversion')
        actual.update(definitions(Path(conversion.__file__).read_text()))
    if name == 'runtime':
        # The only intentional behavior change accepts Linux and its configured backend.
        for changed in ('probe_mlx_runtime', 'import_mlx', '_linux_backend_available'):
            actual.pop(changed, None)
    for key, definition in actual.items():
        assert definition == metadata['definitions'][key], f'{name}.{key}: implementation drift'
    for key, expected in metadata.get('constants', {}).items():
        value = getattr(module, key)
        if isinstance(value, frozenset):
            value = sorted(value)
        assert json.loads(json.dumps(value)) == expected, f'{name}.{key}: kernel/layout drift'
