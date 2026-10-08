import json
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from fixture_cases import darwin_contracts, error_contracts, peak_cases
from test_equivalence import load_fixture

from medmlx_core import errors, runtime


def test_simulated_darwin_arm64_matches_radnn():
    metadata, _ = load_fixture('runtime')
    assert darwin_contracts(runtime) == metadata['darwin']


def test_peak_memory_api_generations_match_radnn():
    metadata, _ = load_fixture('runtime')
    assert peak_cases(runtime) == metadata['peak']


def test_errors_match_radnn():
    metadata, _ = load_fixture('errors')
    assert error_contracts(errors) == metadata['contracts']
    assert str(errors.RadnnError()) == ''
    assert errors.MedmlxError is errors.RadnnError


def test_real_linux_runtime_and_execution():
    if (runtime.platform.system(), runtime.platform.machine()) != ('Linux', 'x86_64'):
        pytest.skip('Real Linux x86_64 host required')
    report = runtime.probe_mlx_runtime()
    assert report.available
    assert report.reason is report.hint is None
    assert report.macos_version is report.apple_chip is report.memory_bytes is None
    assert report.platform_system == 'Linux'
    assert report.platform_machine == 'x86_64'
    assert set(report.to_payload()) == {
        'available', 'mlx_version', 'macos_version', 'apple_chip', 'memory_bytes',
        'python_version', 'platform_system', 'platform_machine', 'reason', 'hint',
    }
    assert report.mlx_version == '0.32.3'
    assert runtime.require_mlx_runtime() == report
    assert runtime.require_mlx_device('MLX') == report
    before = mx.default_device()
    selected = runtime.import_mlx()
    assert selected is mx
    assert mx.default_device() == before
    assert mx.is_available(before)
    assert runtime.mlx_default_device_name(mx) == str(before)
    assert np.array_equal(np.array(selected.array([1, 2, 3]) + 1), [2, 3, 4])


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_linux_retains_explicit_cpu_or_cuda_backend(device):
    calls = []
    backend = SimpleNamespace(
        __version__='0.32.3', default_device=lambda: device,
        is_available=lambda selected: selected == device,
        set_default_device=lambda selected: calls.append(selected),
    )
    with (
        patch.object(runtime.platform, 'system', return_value='Linux'),
        patch.object(runtime.platform, 'machine', return_value='x86_64'),
        patch.object(runtime, '_load_mlx_core', return_value=backend),
    ):
        assert runtime.probe_mlx_runtime().available
        assert runtime.import_mlx() is backend
        assert calls == []
        backend.is_available = lambda _: False
        assert not runtime.probe_mlx_runtime().available
        with pytest.raises(errors.MissingDependencyError, match='configured CPU or CUDA backend'):
            runtime.import_mlx()


@pytest.mark.parametrize(('system', 'machine'), [('Darwin', 'x86_64'), ('Linux', 'aarch64'),
                                                ('Windows', 'AMD64')])
def test_unsupported_hosts_are_rejected(system, machine):
    with (patch.object(runtime.platform, 'system', return_value=system),
          patch.object(runtime.platform, 'machine', return_value=machine)):
        report = runtime.probe_mlx_runtime()
        assert not report.available
        assert f'{system} {machine}' in report.reason
        with pytest.raises(errors.MissingDependencyError):
            runtime.require_mlx_runtime()


def test_missing_mlx_reports_unavailable():
    with patch.object(runtime, '_load_mlx_core', side_effect=ImportError('not installed')):
        assert runtime._mlx_version() == (None, 'mlx is not installed')
        assert runtime._metal_available() == (False, 'mlx is not installed')
        assert not runtime.probe_mlx_runtime().available


def test_report_serializes_without_backend_objects():
    report = runtime.probe_mlx_runtime()
    assert json.loads(json.dumps(report.to_payload())) == report.to_payload()
