"""Float32 convolution and normalization arithmetic shared by MLX engines."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from itertools import count
from typing import TYPE_CHECKING, cast

# MLX's standard convolutions/reductions do not expose these accumulation orders.

if TYPE_CHECKING:
    from mlx.core import array as Array

from medmlx_core.typing import MetalKernel, MlxRuntime

_KERNEL_NAMESPACES = count()

_SOURCES = {
    "conv": (
        ["X", "W", "BIAS"],
        ["Y"],
        """
uint tid = thread_position_in_threadgroup.x;
uint sg = tid / 32;
uint row0 = threadgroup_position_in_grid.y * 128;
uint col0 = threadgroup_position_in_grid.z * 32;
threadgroup float aa[4096];
threadgroup float bb[1024];
metal::simdgroup_float8x8 c00, c01, c10, c11;
c00.thread_elements()[0] = 0; c00.thread_elements()[1] = 0;
c01.thread_elements()[0] = 0; c01.thread_elements()[1] = 0;
c10.thread_elements()[0] = 0; c10.thread_elements()[1] = 0;
c11.thread_elements()[0] = 0; c11.thread_elements()[1] = 0;
// Seed bias before the channel-major products, as in the Torch oracle.
for (int base = -32; base < CI * KZ * KY * KX; base += 32) {
    for (uint i = tid; i < 4096; i += 512) {
        uint row = row0 + i / 32;
        int k = base + int(i % 32);
        float av = 0;
        if (row < BATCH * D * H * S) {
            if (k == -32) av = 1;
            else if (k >= 0 && k < CI * KZ * KY * KX) {
                uint ch = k / (KZ * KY * KX);
                uint rem = k % (KZ * KY * KX);
                int z = int((row / (H * S)) % D) * SZ + int(rem / (KY * KX));
                int y = int((row / S) % H) * SY + int((rem / KX) % KY);
                int x = int(row % S) * SX + int(rem % KX);
                uint b = row / (D * H * S);
                if (z >= 0 && z < DP && y >= 0 && y < HP && x >= 0 && x < XP)
                    av = X[(((b * CI + ch) * DP + z) * HP + y) * XP + x];
            }
        }
        aa[i] = av;
        if (i < 1024) {
            uint oc = col0 + i % 32;
            int wk = base + int(i / 32);
            float bv = 0;
            if (oc < CO) {
                if (wk == -32) bv = BIAS[oc];
                else if (wk >= 0 && wk < CI * KZ * KY * KX) bv = W[wk * CO + oc];
            }
            bb[i] = bv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32; kk += 8) {
        metal::simdgroup_float8x8 a0, a1, b0, b1;
        simdgroup_load(a0, aa + (sg / 2) * 16 * 32 + kk, 32);
        simdgroup_load(a1, aa + ((sg / 2) * 16 + 8) * 32 + kk, 32);
        simdgroup_load(b0, bb + kk * 32 + (sg % 2) * 16, 32);
        simdgroup_load(b1, bb + kk * 32 + (sg % 2) * 16 + 8, 32);
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// All groups have finished reading aa; reuse it for the output tile.
simdgroup_store(c00, aa + (sg / 2) * 16 * 32 + (sg % 2) * 16, 32);
simdgroup_store(c01, aa + (sg / 2) * 16 * 32 + (sg % 2) * 16 + 8, 32);
simdgroup_store(c10, aa + ((sg / 2) * 16 + 8) * 32 + (sg % 2) * 16, 32);
simdgroup_store(c11, aa + ((sg / 2) * 16 + 8) * 32 + (sg % 2) * 16 + 8, 32);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < 4096; i += 512) {
    uint row = row0 + i / 32, oc = col0 + i % 32;
    if (row < BATCH * D * H * S && oc < CO) {
        uint offset = ((row / (D * H * S)) * CO + oc) * D * H * S;
        Y[offset + row % (D * H * S)] = aa[i];
    }
}
        """,
    ),
    "tile": (
        ["X"],
        ["M", "V"],
        """
        uint i=thread_position_in_grid.x;
        if (i>=GROUPS*TILES*4) return;
        uint g=i/(TILES*4), t=i/4%TILES, lane=i%4;
        float m=0, v=0;
        uint count=min(16u, uint(N/4)-min(uint(N/4), t*16));
        for (uint j=0; j<count; ++j) {
          float x=X[g*N+t*64+j*4+lane], delta=x-m;
          m=fma(1.0f/float(j+1), delta, m);
          v=fma(delta, x-m, v);
        }
        M[i]=m; V[i]=v;
        """,
    ),
    "merge": (
        ["M", "V"],
        ["MO", "VO"],
        """
        uint i=thread_position_in_grid.x;
        if (i>=GROUPS*PAIRS*4) return;
        uint g=i/(PAIRS*4), t=i/4%PAIRS, lane=i%4;
        uint a=(g*TILES+2*t)*4+lane, b=a+4;
        uint left=min(uint(COUNT), uint(NV)-2*t*COUNT);
        uint right=min(uint(COUNT), uint(NV)-(2*t+1)*COUNT);
        float delta=M[b]-M[a], cd=float(right)/float(left+right)*delta;
        MO[i]=M[a]+cd;
        VO[i]=fma(delta*float(left), cd, V[a]+V[b]);
        """,
    ),
    "join": (
        ["M", "V", "MA", "VA"],
        ["MO", "VO"],
        """
        uint i=thread_position_in_grid.x;
        if (i>=GROUPS*4) return;
        float c=float(ADD)/float(COUNT+ADD), delta=MA[i]-M[i];
        float cd=c*delta;
        MO[i]=M[i]+cd;
        VO[i]=fma(delta*float(COUNT), cd, V[i]+VA[i]);
        """,
    ),
    "finish": (
        ["X", "M", "V"],
        ["MEAN", "VAR"],
        """
        uint g=thread_position_in_grid.x;
        if (g>=GROUPS) return;
        float m=0, v=0;
        uint count=0;
        for (uint i=NV*4; i<N; ++i) {
          float x=X[g*N+i], delta=x-m;
          ++count; m+=delta/float(count); v=fma(delta, x-m, v);
        }
        for (uint lane=0; lane<4; ++lane) {
          float c=float(NV)/float(count+NV), delta=M[g*4+lane]-m;
          m=fma(c, delta, m);
          float dc=delta*delta*c;
          v+=fma(dc, float(count), V[g*4+lane]);
          count+=NV;
        }
        MEAN[g]=m; VAR[g]=v/float(N);
        """,
    ),
    "coeff": (
        ["M", "V", "W", "B", "EPS"],
        ["S", "BIAS"],
        """
        uint i=thread_position_in_grid.x;
        if (i>=BATCH*CHANNELS) return;
        uint c=i%CHANNELS, g=i/(CHANNELS/GROUPS);
        // Retain the low part of the configured epsilon sum.
        float v=V[g], hi=v+EPS[0], t=hi-v;
        float lo=(v-(hi-t))+(EPS[0]-t)+EPS[1];
        // Correct the reciprocal root using its float32 FMA residual.
        float r=rsqrt(hi), q=hi*r, qe=fma(hi,r,-q);
        float error=fma(-q,r,1.0f)-qe*r-lo*r*r;
        r=fma(0.5f*r,error,r);
        float s=r*W[c]; S[i]=s; BIAS[i]=fma(-s,M[g],B[c]);
        """,
    ),
    "affine": (
        ["X", "S", "B"],
        ["Y"],
        """
        uint i=thread_position_in_grid.x;
        if (i>=SIZE) return;
        uint c=i/SPATIAL;
        Y[i]=fma(S[c],X[i],B[c]);
        """,
    ),
}


class Float32Operators:
    """Tiled ordered convolution FMA and bounded four-lane Welford moments."""

    def __init__(self, mx: MlxRuntime, groups: int, eps: float) -> None:
        self.mx = mx
        self.groups = groups
        namespace = next(_KERNEL_NAMESPACES)
        # Preserve the configured epsilon while keeping all tensor math float32.
        high = struct.unpack("f", struct.pack("f", eps))[0]
        self.eps = mx.array([high, eps - high], dtype=mx.float32)
        self.kernels = {
            name: cast(
                MetalKernel,
                mx.fast.metal_kernel(
                    name=f"medmlx_float32_packed_{namespace}_{name}",
                    input_names=inputs,
                    output_names=outputs,
                    source=source,
                    header="#include <metal_simdgroup_matrix>\n#pragma clang fp contract(off)\n",
                ),
            )
            for name, (inputs, outputs, source) in _SOURCES.items()
        }

    def _run(
        self, name: str, inputs: list[Array], shape: tuple[int, ...], **constants: int
    ) -> list[Array]:
        return self.kernels[name](
            inputs=inputs,
            template=list(constants.items()),
            grid=(
                (512, (shape[0] * math.prod(shape[2:]) + 127) // 128, (shape[1] + 31) // 32)
                if name == "conv"
                else (math.prod(shape), 1, 1)
            ),
            threadgroup=(512 if name == "conv" else 256, 1, 1),
            output_shapes=[shape] * len(_SOURCES[name][1]),
            output_dtypes=[self.mx.float32] * len(_SOURCES[name][1]),
        )

    def conv(
        self,
        values: Array,
        weight: Array,
        bias: Array | None,
        *,
        padding: int | Sequence[int],
        stride: int | Sequence[int],
    ) -> Array:
        mx = self.mx
        pads = (padding,) * 3 if isinstance(padding, int) else padding
        steps = (stride,) * 3 if isinstance(stride, int) else stride
        padded = mx.pad(values, [(0, 0), (0, 0), *((p, p) for p in pads)])
        kernel = tuple(int(n) for n in weight.shape[2:])
        spatial = tuple(
            (int(n) + 2 * p - k) // s + 1
            for n, p, k, s in zip(values.shape[2:], pads, kernel, steps, strict=True)
        )
        batch, channels = map(int, values.shape[:2])
        out_channels = int(weight.shape[0])
        initial = mx.zeros((out_channels,), dtype=mx.float32) if bias is None else bias
        return self._run(
            "conv",
            [padded, mx.transpose(weight, (1, 2, 3, 4, 0)), initial],
            (batch, out_channels, *spatial),
            BATCH=batch,
            CI=channels,
            CO=out_channels,
            D=spatial[0],
            H=spatial[1],
            S=spatial[2],
            KZ=kernel[0],
            KY=kernel[1],
            KX=kernel[2],
            SZ=steps[0],
            SY=steps[1],
            SX=steps[2],
            DP=int(padded.shape[2]),
            HP=int(padded.shape[3]),
            XP=int(padded.shape[4]),
        )[0]

    def _moments(self, values: Array) -> tuple[Array, Array]:
        groups = int(values.shape[0]) * self.groups
        size = int(values.size) // groups
        vectors = size // 4
        tiles = max(1, (vectors + 15) // 16)
        mean, variance = self._run(
            "tile", [values], (groups, tiles, 4), GROUPS=groups, TILES=tiles, N=size
        )
        count = 16
        remainders: list[tuple[Array, Array, int]] = []
        # Keep the rightmost partial blocks in ascending order, as in MONAI's CPU reference.
        while tiles > 1:
            if tiles % 2:
                remainders.append(
                    (mean[:, -1, :], variance[:, -1, :], min(count, vectors - (tiles - 1) * count))
                )
            pairs = tiles // 2
            mean, variance = self._run(
                "merge",
                [mean, variance],
                (groups, pairs, 4),
                GROUPS=groups,
                TILES=tiles,
                PAIRS=pairs,
                COUNT=count,
                NV=vectors,
            )
            count *= 2
            tiles = pairs
        remainders.append((mean[:, 0, :], variance[:, 0, :], min(count, vectors)))
        mean, variance, count = remainders[0]
        for added_mean, added_variance, added_count in remainders[1:]:
            mean, variance = self._run(
                "join",
                [mean, variance, added_mean, added_variance],
                (groups, 4),
                GROUPS=groups,
                COUNT=count,
                ADD=added_count,
            )
            count += added_count
        mean, variance = self._run(
            "finish", [values, mean, variance], (groups,), GROUPS=groups, N=size, NV=vectors
        )
        return mean, variance

    def norm(self, values: Array, weight: Array, bias: Array) -> Array:
        mean, variance = self._moments(values)
        batch, channels = map(int, values.shape[:2])
        scale, offset = self._run(
            "coeff",
            [mean, variance, weight, bias, self.eps],
            (batch * channels,),
            BATCH=batch,
            CHANNELS=channels,
            GROUPS=self.groups,
        )
        return self._run(
            "affine",
            [values, scale, offset],
            tuple(values.shape),
            SIZE=int(values.size),
            SPATIAL=math.prod(values.shape[2:]),
        )[0]
