# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from jax.random import PRNGKey

import haliax as hax
from haliax.nn.attention import (
    alibi_attention_bias,
    dot_product_attention,
    dot_product_attention_weights,
    forgetful_causal_mask,
)
from test_utils import skip_if_no_torch


def test_dot_product_attention_requires_axis_to_be_present():
    Pos = hax.Axis("Pos", 20)
    KeyPos = hax.Axis("Pos_key", 20)
    NumHeads = hax.Axis("NumHeads", 1)
    Hid = hax.Axis("Hid", 8)

    query = hax.ones((NumHeads, KeyPos, Hid))  # NB: KeyPos not Pos
    key = hax.ones((KeyPos, NumHeads, Hid))
    value = hax.ones((KeyPos, NumHeads, Hid))

    try:
        dot_product_attention(Pos, Hid, query, key, value)
    except ValueError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("Should have raised an error")


def test_attention_doesnt_allow_overlapping_axes():
    KeyPos = hax.Axis("Pos_key", 20)
    NumHeads = hax.Axis("NumHeads", 1)
    Hid = hax.Axis("Hid", 8)

    query = hax.ones((NumHeads, KeyPos, Hid))  # NB: KeyPos not Pos
    key = hax.ones((KeyPos, NumHeads, Hid))
    value = hax.ones((KeyPos, NumHeads, Hid))

    try:
        dot_product_attention(KeyPos, Hid, query, key, value)
    except ValueError as e:
        assert "must be distinct" in str(e)
    else:
        raise AssertionError("Should have raised an error")


def test_alibi_attention_bias():
    KeyPos = hax.Axis("KeyPos", 20)
    NumHeads = hax.Axis("NumHeads", 1)
    Hid = hax.Axis("Hid", 8)

    bias = alibi_attention_bias(NumHeads, KeyPos)

    query = hax.ones((NumHeads, Hid))
    key = hax.ones((KeyPos, NumHeads, Hid))

    weights_bias = dot_product_attention_weights(Hid, KeyPos, query, key, bias=bias)
    weights_no_bias = dot_product_attention_weights(Hid, KeyPos, query, key)

    assert weights_bias[KeyPos, -1] > weights_bias[KeyPos, -2]
    assert weights_bias[KeyPos, -1] > weights_no_bias[KeyPos, -1]

    assert weights_no_bias[KeyPos, -1] == weights_no_bias[KeyPos, -2]


@skip_if_no_torch
def test_alibi_attention_compared_to_hf():
    import torch
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor

    L, H = hax.make_axes(L=1, H=16)

    # Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
    torch_tensor = (
        build_alibi_tensor(torch.ones(1, L.size), H.size, dtype=torch.float32).numpy().reshape(H.size, L.size)
    )

    hax_tensor = np.array(alibi_attention_bias(H, L).array)

    assert np.allclose(torch_tensor, hax_tensor)


def test_fcm_attention_mask():
    KeyPos, QueryPos, Head = hax.make_axes(KeyPos=20, QueryPos=10, Head=8)

    mask = forgetful_causal_mask(KeyPos, mask_prob=0.6, sample_prob=False, key=PRNGKey(0))

    assert mask.axes == (KeyPos,)
    assert mask.array[0].item() == 1

    assert mask.astype(float).sum().item() <= KeyPos.size

    query = hax.arange(QueryPos).broadcast_axis(Head)
    key = hax.arange(KeyPos).broadcast_axis(Head)

    weights = dot_product_attention_weights(Head, KeyPos, query, key, mask=mask)

    # check that all masked out values are zero
    weights = weights.rearrange((KeyPos, QueryPos))

    assert (weights * (mask == 0)).sum() == 0
    assert (weights * (mask == 1)).sum() > 0
