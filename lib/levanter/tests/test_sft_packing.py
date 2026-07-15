# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""SFT packing/masking regressions driven end-to-end from a real ChatDataset.

These guard the wiring between the chat-data pipeline and training: that packed
documents cannot attend across segment boundaries, that the per-token loss weight
charges exactly the shifted `{% generation %}` spans, and that packing a batch does
not change the per-document next-token loss.
"""
from __future__ import annotations

import asyncio

import equinox
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chex import assert_trees_all_close
from haliax import Axis
from haliax.partitioning import ResourceAxis
from jax.sharding import NamedSharding, PartitionSpec

from levanter.data.text.datasets import ChatDataset
from levanter.data.text.examples import named_lm_example_from_grug
from levanter.data.text.formats import ChatProcessor
from levanter.grug.attention import reference_attention
from levanter.layers.attention import AttentionBackend, AttentionMask, dot_product_attention
from levanter.models.llama import LlamaConfig
from levanter.store.cache import SerialCacheWriter
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from levanter.utils.tree_utils import inference_mode

from test_text_chat import MULTI_TOOL_TEMPLATE
from test_utils import use_test_mesh

MODEL_NAME = "marin-community/marin-tokenizer"

# Packed sequence length. Larger than the two documents combined (so both pack into
# one example) and equal to the flash block size, so jax_flash/splash run in one block.
POS = 128

# Two documents that both carry masked assistant spans under MULTI_TOOL_TEMPLATE: a
# tool-calling turn plus a final answer, and a plain chat turn.
_CONV_TOOL = [
    {"role": "user", "content": "Call the adder."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "add", "arguments": {"a": 2, "b": 3}}}],
    },
    {"role": "tool", "content": {"result": 5}},
    {"role": "assistant", "content": "The sum is 5."},
]
_CONV_CHAT = [
    {"role": "user", "content": "Say hi."},
    {"role": "assistant", "content": "hi there"},
]
_FIXTURE_CONVS = [_CONV_TOOL, _CONV_CHAT]


@pytest.fixture(scope="module")
def tokenizer() -> MarinTokenizer:
    try:
        return load_tokenizer(MODEL_NAME)
    except Exception as e:  # noqa: BLE001 - network/optional-dep failure should skip, not error
        pytest.skip(f"Could not load tokenizer {MODEL_NAME}: {e}")


def _chat_example(tokenizer: MarinTokenizer, cache_dir, conversations, *, max_segments: int):
    """Build the first ChatDataset example over `conversations` plus the processed rows."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    processor = ChatProcessor(tokenizer, chat_template=MULTI_TOOL_TEMPLATE, mask_user_turns=True)
    processed = processor([{"messages": conv} for conv in conversations])
    exemplar = {"input_ids": np.zeros((0,), np.int32), "assistant_masks": np.zeros((0,), np.int32)}
    with SerialCacheWriter(str(cache_dir), exemplar) as writer:
        writer.write_batch(
            [
                {
                    "input_ids": row["input_ids"].astype(np.int32),
                    "assistant_masks": row["assistant_masks"].astype(np.int32),
                }
                for row in processed
            ]
        )
    dataset = ChatDataset(
        writer.result(),
        Axis("position", POS),
        max_segments_per_example=max_segments,
        slice_strategy="raise",
        mask_user_turns=True,
        block_cross_document_attention=True,
    )
    example = asyncio.run(dataset.getitem_async(0))
    return example, processed


def _hot_key_qkv(length: int, head_size: int):
    """Query/key/value for the segment-leak probe of test_attention.py::test_segment_ids_are_respected.

    Every query is ones; key 0 alone carries a large score component and value 0 alone
    carries the detector value 300 in channel 1. A query attends key 0 only when causal
    masking and segment masking both allow it, so channel 1 of the output is 300 exactly
    for positions in key 0's segment and 0 elsewhere.
    """
    query = np.ones((length, head_size), np.float32)
    key = np.zeros((length, head_size), np.float32)
    key[0, 0] = 100.0
    value = np.zeros((length, head_size), np.float32)
    value[0, 1] = 300.0
    return query, key, value


@pytest.mark.parametrize("impl", ["vanilla", "jax_flash", "default", "splash"])
def test_packed_segments_block_cross_segment_attention(tokenizer, tmp_path, impl):
    if impl == "splash" and jax.default_backend() != "tpu":
        pytest.skip("splash kernel requires TPU")

    example, _ = _chat_example(tokenizer, tmp_path / "packed", _FIXTURE_CONVS, max_segments=len(_FIXTURE_CONVS))
    query_segments, _ = example.attn_mask.segment_ids
    segments = np.asarray(query_segments)
    length = segments.shape[0]
    # splash/default require 128-wide heads; vanilla/jax_flash use a 2-wide head.
    head_size = 128 if impl in ("default", "splash") else 2

    Pos = Axis("Pos", length)
    KPos = Pos.alias("KPos")
    Head = Axis("Head", head_size)
    query_np, key_np, value_np = _hot_key_qkv(length, head_size)

    with use_test_mesh() as mesh:
        query = hax.named(query_np, (Pos, Head))
        key = hax.named(key_np, (KPos, Head))
        value = hax.named(value_np, (KPos, Head))
        query, key, value = jax.device_put(
            [query, key, value], NamedSharding(mesh, PartitionSpec(ResourceAxis.DATA, None))
        )
        seg = jax.device_put(segments, NamedSharding(mesh, PartitionSpec(ResourceAxis.DATA)))
        seg = hax.named(seg, (Pos,))
        mask = AttentionMask(causal_offset=0, segment_ids=seg)

        jit_dpa = equinox.filter_jit(dot_product_attention)
        result = jit_dpa(
            Pos,
            KPos,
            Head,
            query,
            key,
            value,
            attn_backend=AttentionBackend(impl),
            mask=mask,
            flash_block_size=length,
        )
        detector = np.asarray(result.array)[:, 1]

    in_key_segment = segments == segments[0]
    assert in_key_segment.sum() > 0 and (~in_key_segment).sum() > 0, "fixture must span >1 segment"
    # Positions sharing key 0's document see the value; every other position (other
    # documents and padding) must see exactly zero — no cross-segment leak.
    assert_trees_all_close(detector[in_key_segment], 300.0, atol=1e-2, rtol=1e-2)
    assert_trees_all_close(detector[~in_key_segment], 0.0, atol=1e-2, rtol=1e-2)


def test_packed_segments_block_cross_segment_attention_grug_reference(tokenizer, tmp_path):
    example, _ = _chat_example(tokenizer, tmp_path / "packed", _FIXTURE_CONVS, max_segments=len(_FIXTURE_CONVS))
    query_segments, _ = example.attn_mask.segment_ids
    segments = np.asarray(query_segments)
    length = segments.shape[0]

    query_np, key_np, value_np = _hot_key_qkv(length, head_size=2)
    # reference_attention wants (batch, seq, heads, head_size).
    query = jnp.asarray(query_np)[None, :, None, :]
    key = jnp.asarray(key_np)[None, :, None, :]
    value = jnp.asarray(value_np)[None, :, None, :]

    out = reference_attention(query, key, value, example.attn_mask, logits_dtype=jnp.float32)
    detector = np.asarray(out)[0, :, 0, 1]

    in_key_segment = segments == segments[0]
    assert_trees_all_close(detector[in_key_segment], 300.0, atol=1e-2, rtol=1e-2)
    assert_trees_all_close(detector[~in_key_segment], 0.0, atol=1e-2, rtol=1e-2)


def test_packed_loss_weight_charges_only_generation_spans(tokenizer, tmp_path):
    example, processed = _chat_example(
        tokenizer, tmp_path / "packed", _FIXTURE_CONVS, max_segments=len(_FIXTURE_CONVS)
    )
    tokens = np.asarray(example.tokens)
    loss_weight = np.asarray(example.loss_weight)
    query_segments, _ = example.attn_mask.segment_ids
    segments = np.asarray(query_segments)

    # The loss weight is a binary next-token training mask.
    assert set(np.unique(loss_weight).tolist()) <= {0.0, 1.0}

    charged = np.nonzero(loss_weight > 0)[0]
    # Every charged position trains its NEXT token and never crosses a document
    # boundary or lands on padding.
    assert np.all(segments[charged] == segments[charged + 1])
    assert np.all(segments[charged + 1] != -1)

    # The next tokens charged are exactly the assistant/generation tokens, in order:
    # nothing from user, tool-response, or role-header spans is charged.
    charged_next_tokens = tokens[charged + 1]
    expected_assistant_tokens = np.concatenate([row["input_ids"][row["assistant_masks"] == 1] for row in processed])
    np.testing.assert_array_equal(charged_next_tokens, expected_assistant_tokens)

    # Padding carries no loss.
    assert float(loss_weight[segments == -1].sum()) == 0.0


def test_packed_leading_document_loss_matches_unpacked(tokenizer, tmp_path):
    """Packing must not change the leading document's per-token loss.

    Cross-document attention is blocked by segment ids, but position ids are absolute
    and do not reset per document, so only the position-aligned leading document is
    invariant under packing today; later documents shift and will match once a
    pad-per-document packing path preserves per-document positions. This pins the
    leading-document invariant end-to-end through a real attention stack, which is the
    guard against a cross-document leak surfacing as a changed loss.
    """
    packed, processed = _chat_example(tokenizer, tmp_path / "packed", _FIXTURE_CONVS, max_segments=len(_FIXTURE_CONVS))
    unpacked, _ = _chat_example(tokenizer, tmp_path / "unpacked", _FIXTURE_CONVS[:1], max_segments=1)
    leading_len = len(processed[0]["input_ids"])

    # A rotary-position model whose attention honors segment ids; vanilla runs the same
    # on CPU and TPU so the leading-document loss matches regardless of backend.
    config = LlamaConfig(
        max_seq_len=POS,
        hidden_dim=32,
        intermediate_dim=64,
        num_heads=4,
        num_kv_heads=4,
        num_layers=2,
        attn_backend=AttentionBackend.VANILLA,
    )

    # Build the model and run the forward under an active mesh; the attention path uses
    # shard_map, which requires a concrete mesh, so both must be inside use_test_mesh and
    # the forward must be jitted (matching tests/test_qwen3_moe.py).
    with use_test_mesh():
        model = inference_mode(config.build(Axis("vocab", tokenizer.vocab_size), key=jax.random.PRNGKey(0)), True)

        @hax.named_jit
        def per_position_loss(model, example):
            return model.compute_next_token_loss(example, reduction=None, reduction_axis=()).array

        def host_example(grug_example):
            # ChatDataset pins its arrays to CPU; drop that device commitment (via host
            # numpy) so the jit under the mesh places the inputs on its own devices.
            named = named_lm_example_from_grug(grug_example, Pos=model.Pos)
            return jax.tree.map(np.asarray, named)

        packed_loss = np.asarray(per_position_loss(model, host_example(packed)))
        unpacked_loss = np.asarray(per_position_loss(model, host_example(unpacked)))

    # The leading document occupies positions 0..leading_len-1 in both cases; its loss
    # must be identical. (Uncharged positions are zero in both, so this also confirms
    # the loss weight agrees.)
    assert float(np.abs(packed_loss[:leading_len]).sum()) > 0, "leading document must charge some loss"
    assert_trees_all_close(packed_loss[:leading_len], unpacked_loss[:leading_len], atol=1e-4, rtol=1e-4)
