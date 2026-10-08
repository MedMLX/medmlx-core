"""Independent pre-extraction outputs for shared network-layer arithmetic."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from medmlx_core.channel_last import ChannelLastGraph


@pytest.mark.parametrize("case", ["2d", "3d"])
def test_channel_last_matches_qualified_source(case):
    if not mx.metal.is_available():
        pytest.skip("M1 Max Metal fixtures; other backends need separate qualification")
    with np.load(Path(__file__).parent / "fixtures/channel_last.npz", allow_pickle=False) as source:
        weights = {
            key: mx.array(source[key])
            for key in source.files
            if key.endswith((".weight", ".bias", ".running_mean", ".running_var"))
        }
        graph = ChannelLastGraph(weights, mx=mx)
        x = mx.array(source[f"input_{case}"])
        if case == "2d":
            result = graph.conv(x, "conv", padding=1, groups=2)
            result = graph.swish(graph.batch_norm(result, "bn", eps=1e-3))
            result = graph.bilinear2x(graph.layer_norm(result))
        else:
            result = graph.instance_norm(graph.deconv3d(x, "deconv"))
        mx.eval(result)
        actual = np.asarray(result)
        expected = source[f"expected_{case}"]
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)
