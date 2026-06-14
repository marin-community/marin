# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import haliax as hax
import jax.numpy as jnp
import pytest

from levanter.layers.attention import AttentionMask


_BENCH_PATH = Path(__file__).parents[2] / "scripts" / "bench" / "bench_splash_attention_masks.py"
_BENCH_SPEC = importlib.util.spec_from_file_location("bench_splash_attention_masks", _BENCH_PATH)
assert _BENCH_SPEC is not None and _BENCH_SPEC.loader is not None
bench_splash_attention_masks = importlib.util.module_from_spec(_BENCH_SPEC)
_BENCH_SPEC.loader.exec_module(bench_splash_attention_masks)

BenchShape = bench_splash_attention_masks.BenchShape
_doc_length_batches = bench_splash_attention_masks._doc_length_batches
_doc_lengths = bench_splash_attention_masks._doc_lengths
_packed_prefix_lengths_per_segment = bench_splash_attention_masks._packed_prefix_lengths_per_segment
_packed_prefix_mask = bench_splash_attention_masks._packed_prefix_mask
_packed_segment_ids = bench_splash_attention_masks._packed_segment_ids
_parse_doc_length_batches = bench_splash_attention_masks._parse_doc_length_batches


@pytest.mark.parametrize(
    ("profile", "expected_lengths"),
    [
        ("equal", (32, 32, 32, 32)),
        ("staggered", (13, 26, 38, 51)),
        ("long-tail", (67, 34, 18, 9)),
    ],
)
def test_packed_doc_profiles_feed_segment_run_metadata(profile, expected_lengths):
    shape = BenchShape(
        batch=2,
        seq_len=128,
        heads=2,
        head_dim=16,
        block_size=128,
        docs_per_sequence=4,
        prefix_tokens_per_doc=16,
        doc_length_profile=profile,
        doc_lengths=None,
        dtype=jnp.bfloat16,
    )
    Batch = hax.Axis("batch", shape.batch)
    Pos = hax.Axis("position", shape.seq_len)
    KPos = Pos.alias("key_position")

    assert _doc_lengths(shape) == expected_lengths

    segment_ids = _packed_segment_ids(shape, Batch, Pos, KPos)
    prefix_mask = _packed_prefix_mask(shape, Batch, Pos)
    segment_id_mask = AttentionMask.causal(segment_ids=(segment_ids.q, segment_ids.kv))
    segment_run_mask = AttentionMask.causal().with_segment_runs(
        segment_ids.q,
        kv_segment_ids=segment_ids.kv,
        max_segments=shape.docs_per_sequence,
    )

    metadata = segment_run_mask.segment_run_metadata
    assert metadata is not None
    assert metadata.segment_lengths.array.shape == (shape.batch, shape.docs_per_sequence)
    assert (metadata.segment_lengths.array[0] == jnp.asarray(expected_lengths)).all()
    assert (prefix_mask.array[:, : min(shape.prefix_tokens_per_doc, expected_lengths[0])]).all()
    assert (segment_run_mask.materialize(Pos, KPos).array == segment_id_mask.materialize(Pos, KPos).array).all()

    prefix_lengths_per_segment = _packed_prefix_lengths_per_segment(shape, Batch, metadata.segment_lengths.axes[-1])
    packed_prefix_mask = AttentionMask.prefix_lm(prefix_mask=prefix_mask, segment_ids=(segment_ids.q, segment_ids.kv))
    segment_run_prefix_mask = segment_run_mask.with_prefix_lengths_per_segment(prefix_lengths_per_segment)
    assert (
        segment_run_prefix_mask.materialize(Pos, KPos).array == packed_prefix_mask.materialize(Pos, KPos).array
    ).all()


def test_explicit_doc_lengths_override_profile_and_docs_per_sequence():
    explicit_lengths = (5, 17, 29, 77)
    shape = BenchShape(
        batch=2,
        seq_len=128,
        heads=2,
        head_dim=16,
        block_size=128,
        docs_per_sequence=999,
        prefix_tokens_per_doc=16,
        doc_length_profile="equal",
        doc_lengths=(explicit_lengths,),
        dtype=jnp.bfloat16,
    )
    Batch = hax.Axis("batch", shape.batch)
    Pos = hax.Axis("position", shape.seq_len)
    KPos = Pos.alias("key_position")

    segment_ids = _packed_segment_ids(shape, Batch, Pos, KPos)
    segment_run_mask = AttentionMask.causal().with_segment_runs(
        segment_ids.q,
        kv_segment_ids=segment_ids.kv,
        max_segments=len(explicit_lengths),
    )

    metadata = segment_run_mask.segment_run_metadata
    assert metadata is not None
    assert _doc_lengths(shape) == explicit_lengths
    assert _doc_length_batches(shape) == (explicit_lengths, explicit_lengths)
    assert (metadata.segment_lengths.array[0] == jnp.asarray(explicit_lengths)).all()


def test_explicit_doc_lengths_can_vary_by_batch():
    doc_lengths_by_batch = ((5, 17, 106), (8, 9, 10, 101))
    shape = BenchShape(
        batch=2,
        seq_len=128,
        heads=2,
        head_dim=16,
        block_size=128,
        docs_per_sequence=4,
        prefix_tokens_per_doc=16,
        doc_length_profile="equal",
        doc_lengths=doc_lengths_by_batch,
        dtype=jnp.bfloat16,
    )
    Batch = hax.Axis("batch", shape.batch)
    Pos = hax.Axis("position", shape.seq_len)
    KPos = Pos.alias("key_position")

    segment_ids = _packed_segment_ids(shape, Batch, Pos, KPos)
    prefix_mask = _packed_prefix_mask(shape, Batch, Pos)
    segment_run_mask = AttentionMask.causal().with_segment_runs(
        segment_ids.q,
        kv_segment_ids=segment_ids.kv,
        max_segments=shape.docs_per_sequence,
    )

    metadata = segment_run_mask.segment_run_metadata
    assert metadata is not None
    assert _doc_length_batches(shape) == doc_lengths_by_batch
    assert (metadata.segment_lengths.array[0] == jnp.asarray([5, 17, 106, 0])).all()
    assert (metadata.segment_lengths.array[1] == jnp.asarray([8, 9, 10, 101])).all()
    assert int(metadata.num_segments.array[0]) == 3
    assert int(metadata.num_segments.array[1]) == 4
    assert (prefix_mask.array[0, :5]).all()
    assert (prefix_mask.array[1, :8]).all()

    prefix_lengths_per_segment = _packed_prefix_lengths_per_segment(shape, Batch, metadata.segment_lengths.axes[-1])
    packed_prefix_mask = AttentionMask.prefix_lm(prefix_mask=prefix_mask, segment_ids=(segment_ids.q, segment_ids.kv))
    segment_run_prefix_mask = segment_run_mask.with_prefix_lengths_per_segment(prefix_lengths_per_segment)
    assert (
        segment_run_prefix_mask.materialize(Pos, KPos).array == packed_prefix_mask.materialize(Pos, KPos).array
    ).all()


def test_parse_doc_lengths_supports_per_batch_layouts():
    assert _parse_doc_length_batches("5,17,106;8,9,10,101") == ((5, 17, 106), (8, 9, 10, 101))
