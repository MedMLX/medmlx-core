import numpy as np
import pytest
from test_equivalence import load_fixture

from medmlx_core.checkpoints import (
    _maybe_array,
    load_torch_checkpoint,
    mapping_from_pairs,
    tensor_mapping_from_payload,
)
from medmlx_core.errors import InvalidInputError


def test_synthetic_weight_mapping_matches_radnn():
    metadata, arrays = load_fixture('checkpoints')
    torch = pytest.importorskip('torch')
    state = {'module.weight': torch.from_numpy(arrays['host'].copy()),
             'array': arrays['host'], 'epoch': 4, 'description': 'ignored'}
    actual = tensor_mapping_from_payload(state, what='synthetic')
    assert list(actual) == metadata['keys']
    for key, value in actual.items():
        assert value.dtype == arrays[f'mapped__{key}'].dtype
        assert np.array_equal(value, arrays[f'mapped__{key}'])
    pairs = mapping_from_pairs(list(actual.items()), what='synthetic')
    assert list(pairs) == metadata['pair_keys']
    for key in pairs:
        assert np.array_equal(pairs[key], arrays[f'mapped__{key}'])
    assert _maybe_array(arrays['host']) is arrays['host']
    for value in (1, 1.0, True, 'text', np.float32(1), None, object()):
        assert _maybe_array(value) is None


@pytest.mark.parametrize('weights_only', [False, True])
def test_optional_torch_checkpoint_matches_radnn(tmp_path, weights_only):
    _, arrays = load_fixture('checkpoints')
    torch = pytest.importorskip('torch')
    state = {'module.weight': torch.from_numpy(arrays['host'].copy())}
    if not weights_only:
        state.update(array=arrays['host'], epoch=4, description='ignored')
    path = tmp_path / 'synthetic.pt'
    torch.save(state, path)
    actual = tensor_mapping_from_payload(
        load_torch_checkpoint(path, weights_only=weights_only), what='synthetic')
    if weights_only:
        assert list(actual) == ['module.weight']
        assert np.array_equal(actual['module.weight'], arrays['safe'])
    else:
        assert list(actual) == ['module.weight', 'array']
        for key, value in actual.items():
            assert np.array_equal(value, arrays[f'loaded__{key}'])


def test_rejects_duplicate_mapping_and_malformed_tensor():
    array = np.ones((2,), dtype=np.float32)
    with pytest.raises(InvalidInputError, match='duplicate synthetic parameter: same'):
        mapping_from_pairs([('same', array), ('same', array)], what='synthetic')
    with pytest.raises(InvalidInputError, match='duplicate synthetic parameter: 1'):
        tensor_mapping_from_payload({1: array, '1': array}, what='synthetic')
    with pytest.raises(InvalidInputError, match='checkpoint payload must be a mapping'):
        tensor_mapping_from_payload([], what='synthetic')

    class MalformedTensor:
        def numpy(self):
            return [1, 2]

    with pytest.raises(InvalidInputError, match=r'tensor numpy\(\) must return an ndarray'):
        _maybe_array(MalformedTensor())


def test_torch_loader_rejects_non_mapping_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    path = tmp_path / 'bad.pt'
    torch.save([1, 2], path)
    with pytest.raises(InvalidInputError, match=r'bad\.pt is not a mapping checkpoint'):
        load_torch_checkpoint(path)
