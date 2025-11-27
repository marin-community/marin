# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import equinox as eqx
import pytest
from jax._src.core import ShardingTypeError
from jax.sharding import AxisType, Mesh

import haliax as hax
from haliax import Axis, NamedArray
from haliax.nn.normalization import softmax
from haliax.partitioning import axis_mapping, pspec_for_axis, set_mesh
from test_utils import skip_if_not_enough_devices

SeqQ = Axis("seq_q", 4)
SeqK = Axis("seq_k", 4)
Embed = Axis("embed", 16)
Head = Axis("head", 2)
DHead = Axis("dhead", 8)
Mlp = Axis("mlp", 32)
Vocab = Axis("vocab", 64)


class TinyTransformerParams(eqx.Module):
    wq: NamedArray
    wk: NamedArray
    wv: NamedArray
    wo: NamedArray
    w1: NamedArray
    w2: NamedArray
    w_vocab: NamedArray
    b_mlp: NamedArray
    b_rms: NamedArray


def _rms_norm(x: NamedArray, bias: NamedArray | None = None) -> NamedArray:
    var = hax.mean(hax.square(x), axis=Embed)
    inv = hax.rsqrt(var + 1e-5)
    out = x * inv
    if bias is not None:
        with pytest.raises(ShardingTypeError):
            # bias needs to be resharded explicitly
            out = out + bias

        # capture the compute axis mapping for bias
        out = out + hax.auto_sharded(bias)

    return out


def _tiny_transformer(params: TinyTransformerParams, x: NamedArray) -> NamedArray:
    x = _rms_norm(x, params.b_rms)
    q = hax.einsum("... e, e h d -> ... h d", x, params.wq)
    k = hax.rename(hax.einsum("... s e, e h d -> ... s h d", x, params.wk), {SeqQ: SeqK})
    v = hax.rename(hax.einsum("... s e, e h d -> ... s h d", x, params.wv), {SeqQ: SeqK})

    scores = hax.einsum("... s h d, ... t h d -> ... s t h d", q, k)
    attn = softmax(scores, axis=SeqK)
    context = hax.einsum("... s t h d, ... t h d -> ... s h d", attn, v)
    attn_out = hax.einsum("... s h d, h d e -> ... s e", context, params.wo)

    hidden = hax.einsum("... s e, e m -> ... s m", attn_out, params.w1)
    hidden = hidden + params.b_mlp
    hidden = hax.tanh(hidden)
    hidden = hax.einsum("... s m, m e -> ... s e", hidden, params.w2)

    logits = hax.einsum("... s e, e v -> ... s v", hidden, params.w_vocab)
    return logits


@skip_if_not_enough_devices(4)
def test_transformer_block_explicit_sharding():
    devices = jax.devices()
    if len(devices) % 2 != 0:
        import pytest

        pytest.skip("Need even number of devices to build (data, model) mesh")

    mesh = Mesh(
        np.array(devices).reshape(-1, 2),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    Batch = Axis("batch", mesh.devices.shape[0])

    common_map = {
        "head": "model",
        "dhead": None,
        "mlp": "model",
        "vocab": "model",
        "seq_q": None,
        "seq_k": None,
    }

    param_map = {"embed": "data", **common_map}
    compute_map = {"batch": "data", **common_map}

    params = TinyTransformerParams(
        wq=hax.ones((Embed, Head, DHead)),
        wk=hax.ones((Embed, Head, DHead)),
        wv=hax.ones((Embed, Head, DHead)),
        wo=hax.ones((Head, DHead, Embed)),
        w1=hax.ones((Embed, Mlp)),
        w2=hax.ones((Mlp, Embed)),
        w_vocab=hax.ones((Embed, Vocab)),
        b_mlp=hax.ones((Mlp,)),
        b_rms=hax.ones((Embed,)),
    )

    x = hax.ones((Batch, SeqQ, Embed))

    with set_mesh(mesh):
        params = hax.shard(params, param_map)
        x = hax.shard(x, compute_map)

        fn = jax.jit(_tiny_transformer)
        with axis_mapping(compute_map):
            logits = fn(params, x)

    expected_spec = pspec_for_axis((Batch, SeqQ, Vocab), compute_map)
    assert logits.array.sharding.spec == expected_spec
    assert logits.axes == (Batch, SeqQ, Vocab)
