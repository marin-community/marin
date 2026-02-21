# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax

from levanter.data.dataset import ListAsyncDataset
from levanter.data.text import (
    BatchTokenizer,
    BlockShuffleConfig,
    ChatLmDatasetFormat,
    ChatDataset,
    DatasetComponent,
    DirectDatasetComponent,
    GrugLmExample,
    LmDataConfig,
    PrebuiltLmDatasetFormat,
    UrlDatasetSourceConfig,
    build_lm_dataset_cache,
    dataset_for_component,
    grug_lm_example_from_named,
    named_lm_example_from_grug,
    preprocessor_for_format,
)
from levanter.data.text.datasets import TokenSeqDataset
from levanter.models.lm_model import LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.schedule import BatchSchedule
from tests.test_utils import skip_if_hf_model_not_accessible


def test_dont_blow_up_without_validation_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        component = DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=["kaa"], validation_urls=[]),
            cache_dir=tmpdir,
        )
        config = LmDataConfig(
            components={"tiny": component},
            tokenizer="passthrough",
            vocab_size=64,
        )

        Pos = hax.Axis("position", 10)
        # mostly just making sure this doesn't blow up
        assert config.validation_sets(Pos) == {}


def test_lm_data_config_accepts_block_shuffle_as_shuffle_policy():
    cfg = LmDataConfig(shuffle=BlockShuffleConfig(io_block_size=128, window_blocks=8))
    assert isinstance(cfg.shuffle, BlockShuffleConfig)


def test_train_sets_applies_block_shuffle_policy_for_direct_component():
    Pos = hax.Axis("position", 1)
    examples = [GrugLmExample.causal(jnp.array([i], dtype=jnp.int32)) for i in range(10)]
    direct_component = DirectDatasetComponent(datasets={"train": ListAsyncDataset(examples)})
    cfg = LmDataConfig(
        components={"direct": direct_component},
        shuffle=BlockShuffleConfig(io_block_size=4, window_blocks=2, perm_type="feistel"),
    )

    train_sets = cfg.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))
    shuffled_examples = train_sets["direct"].as_sync_dataset().get_batch(list(range(10)))
    shuffled_ids = [int(ex.tokens[0]) for ex in shuffled_examples]

    assert sorted(shuffled_ids) == list(range(10))
    assert set(shuffled_ids[-2:]) == {8, 9}


class _FakeReadHandle:
    def __init__(self, payload: np.ndarray, read_counter: dict[str, int]):
        self._payload = payload
        self._read_counter = read_counter

    async def _read(self):
        self._read_counter["reads"] += 1
        return self._payload

    def read(self):
        return self._read()


class _FakeTensorData:
    def __init__(self, flat_tokens: np.ndarray, read_counter: dict[str, int]):
        self._flat_tokens = flat_tokens
        self._read_counter = read_counter

    def __getitem__(self, item):
        return _FakeReadHandle(self._flat_tokens[item], self._read_counter)


@pytest.mark.asyncio
async def test_token_seq_dataset_coalesces_reads_and_preserves_order():
    read_counter = {"reads": 0}
    flat_tokens = np.arange(30, dtype=np.int32)
    token_array = SimpleNamespace(
        data_size=len(flat_tokens),
        data=_FakeTensorData(flat_tokens, read_counter),
    )
    doc_cache = SimpleNamespace(store=SimpleNamespace(tree={"input_ids": token_array}))

    ds = TokenSeqDataset(doc_cache, seq_len=3)
    indices = [3, 1, 2, 2, 4, 9, 8]
    batch = await ds.get_batch(indices)

    expected = [flat_tokens[i * 3 : (i + 1) * 3] for i in indices]
    for actual, exp in zip(batch, expected, strict=True):
        np.testing.assert_array_equal(actual, exp)

    # Unique sorted indices [1,2,3,4,8,9] -> runs [1..5) and [8..10)
    assert read_counter["reads"] == 2
    stats = ds.read_stats
    assert stats.examples_requested == len(indices)
    assert stats.unique_examples_requested == 6
    assert stats.coalesced_ranges == 2
    assert stats.tensorstore_reads_issued == 2


@pytest.mark.asyncio
async def test_token_seq_dataset_rejects_negative_indices():
    read_counter = {"reads": 0}
    flat_tokens = np.arange(12, dtype=np.int32)
    token_array = SimpleNamespace(
        data_size=len(flat_tokens),
        data=_FakeTensorData(flat_tokens, read_counter),
    )
    doc_cache = SimpleNamespace(store=SimpleNamespace(tree={"input_ids": token_array}))
    ds = TokenSeqDataset(doc_cache, seq_len=3)

    with pytest.raises(ValueError, match="Negative indices"):
        await ds.get_batch([0, -1])


def test_lm_example_handles_ignore_id():
    Pos = hax.Axis("position", 10)
    Vocab = hax.Axis("vocab", Pos.size + 1)
    Embed = hax.Axis("embed", 10)
    tokens = hax.arange(Pos, dtype=jnp.int32)

    ignore_id = 6
    eos_id = 10

    ex_ignore = LmExample.causal(tokens, ignore_id=ignore_id, eos_id=eos_id)
    ex_no_ignore = LmExample.causal(tokens, eos_id=eos_id)
    assert ex_ignore.loss_weight[Pos, ignore_id - 1] == 0

    logits = hax.ones((Pos, Embed))
    lm_head = hax.zeros((Embed, Vocab))
    lm_head = lm_head.at[Vocab, ignore_id].set(-100)

    ignored_loss = maybe_fused_next_token_loss(
        Pos, Embed, Vocab, logits, lm_head, tokens, loss_weight=ex_ignore.loss_weight
    )
    no_ignore_loss = maybe_fused_next_token_loss(
        Pos, Embed, Vocab, logits, lm_head, tokens, loss_weight=ex_no_ignore.loss_weight
    )

    assert no_ignore_loss.item() >= ignored_loss.item() + 100 / Pos.size


def test_unnamed_lm_example_parity_with_named():
    Pos = hax.Axis("position", 10)
    tokens = hax.arange(Pos, dtype=jnp.int32)

    named = LmExample.causal(tokens, ignore_id=6, eos_id=9)
    grug_example = GrugLmExample.causal(tokens.array, ignore_id=6, eos_id=9)
    converted = named_lm_example_from_grug(grug_example, Pos)

    np.testing.assert_array_equal(converted.tokens.array, named.tokens.array)
    np.testing.assert_array_equal(converted.loss_weight.array, named.loss_weight.array)
    assert converted.attn_mask.is_causal == named.attn_mask.is_causal
    assert converted.attn_mask.sliding_window == named.attn_mask.sliding_window
    np.testing.assert_array_equal(converted.attn_mask.segment_ids[0].array, named.attn_mask.segment_ids[0].array)
    np.testing.assert_array_equal(converted.attn_mask.segment_ids[1].array, named.attn_mask.segment_ids[1].array)


def test_named_unnamed_lm_example_roundtrip():
    Pos = hax.Axis("position", 8)
    named = LmExample.causal(hax.arange(Pos, dtype=jnp.int32), eos_id=7)

    grug_example = grug_lm_example_from_named(named)
    converted_back = named_lm_example_from_grug(grug_example, Pos)

    np.testing.assert_array_equal(converted_back.tokens.array, named.tokens.array)
    np.testing.assert_array_equal(converted_back.loss_weight.array, named.loss_weight.array)
    np.testing.assert_array_equal(
        converted_back.attn_mask.segment_ids[0].array,
        named.attn_mask.segment_ids[0].array,
    )


def test_merge_split_encodings(local_gpt2_tokenizer):
    tokenizer = local_gpt2_tokenizer
    # make this very short for testing

    lorem = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""

    short_batch_tokenizer = BatchTokenizer(tokenizer, _workaround_len=len(lorem) // 3)
    # force this
    short_batch_tokenizer._needs_long_sequence_workaround = True

    batch_tokenizer = BatchTokenizer(tokenizer, _workaround_len=50000)
    batch = [{"text": lorem}]

    short_out = short_batch_tokenizer(batch)
    reg_out = batch_tokenizer(batch)

    assert short_out == reg_out


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_llama_tokenizer_needs_long_sequence_workaround():
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    batch_tokenizer = BatchTokenizer(tokenizer)
    assert batch_tokenizer._needs_long_sequence_workaround


def test_prebuilt_cache_with_loss_weights(tmp_path):
    records = [
        {"input_ids": [1, 2, 3, 4], "loss_weights": [1.0, 0.5, 0.0, 1.0]},
        {"input_ids": [5, 6, 7, 8], "loss_weights": [0.0, 1.0, 1.0, 1.0]},
    ]
    data_path = tmp_path / "prebuilt.jsonl"
    with data_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    component = DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[str(data_path)], validation_urls=[]),
        format=PrebuiltLmDatasetFormat(
            loss_weights_key="loss_weights",
            loss_weight_transform=lambda weights: weights * 2.0,
        ),
        cache_dir=str(tmp_path),
    )
    config = LmDataConfig(
        components={"prebuilt": component},
        tokenizer="passthrough",
        vocab_size=16,
    )

    cache = config.build_caches("train")["prebuilt"]
    Pos = hax.Axis("position", 4)
    ds = dataset_for_component(
        component,
        Pos,
        cache,
        eos_id=None,
        block_cross_document_attention=config.block_cross_document_attention,
    ).as_sync_dataset()

    example = ds[0]
    np.testing.assert_array_equal(np.asarray(example.tokens), np.array(records[0]["input_ids"], dtype=np.int32))
    expected_loss_weight = np.array([2.0, 1.0, 0.0, 0.0], dtype=np.asarray(example.loss_weight).dtype)
    np.testing.assert_array_equal(np.asarray(example.loss_weight), expected_loss_weight)


def test_prebuilt_cache_without_loss_weights(tmp_path):
    records = [{"input_ids": [1, 2, 3, 4]}]
    data_path = tmp_path / "prebuilt_no_weights.jsonl"
    with data_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    component = DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[str(data_path)], validation_urls=[]),
        format=PrebuiltLmDatasetFormat(),
        cache_dir=str(tmp_path),
    )
    config = LmDataConfig(
        components={"prebuilt": component},
        tokenizer="passthrough",
        vocab_size=16,
    )

    cache = config.build_caches("train")["prebuilt"]
    Pos = hax.Axis("position", 4)
    ds = dataset_for_component(
        component,
        Pos,
        cache,
        eos_id=None,
        block_cross_document_attention=config.block_cross_document_attention,
    ).as_sync_dataset()

    example = ds[0]
    expected_loss_weight = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.asarray(example.loss_weight).dtype)
    np.testing.assert_array_equal(np.asarray(example.loss_weight), expected_loss_weight)


def test_train_set_last_mile_wraps_to_named(tmp_path):
    records = [{"input_ids": [1, 2, 3, 4]}]
    data_path = tmp_path / "prebuilt_train.jsonl"
    with data_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    component = DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[str(data_path)], validation_urls=[]),
        format=PrebuiltLmDatasetFormat(),
        cache_dir=str(tmp_path),
    )
    config = LmDataConfig(
        components={"prebuilt": component},
        tokenizer="passthrough",
        vocab_size=16,
    )

    Pos = hax.Axis("position", 4)
    train_sets = config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))
    grug_example = train_sets["prebuilt"].as_sync_dataset()[0]
    assert isinstance(grug_example, GrugLmExample)

    named_train_set = config.train_set(Pos, BatchSchedule(1), key=jax.random.PRNGKey(0)).as_sync_dataset()
    named_example = named_train_set[0]
    assert isinstance(named_example, LmExample)


@pytest.fixture
def dummy_chat_data():
    messages = [
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there, how can I help?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "To get to the other side."},
                {"role": "assistant", "content": "No, the other side."},
            ]
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "chat.jsonl"
        with path.open("w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")
        yield str(path)


def assert_loss_weight_matches_all_assistants(example, tokenizer):
    """
    Assert that loss_weight == 1 exactly over assistant‑content spans.

    A span starts at the newline that follows
    "<|start_header_id|>assistant<|end_header_id|>"
    and ends just before the next "<|eot_id|>".
    """
    # ok we want to be sure we're predicting the assistant tokens
    # This is very fiddly, so we want to be careful.
    # In Levanter, the loss_weight is 1 for positions we compute loss on, 0 for positions we don't
    # that means we compute loss (have 1 loss weight) on the positions before each assistant token
    # our current chat template inserts a newline after each role
    # (consistent with Olmo's)
    # Unfortunately, if we change the
    # decoded = tokenizer.decode(ex.tokens.array, skip_special_tokens=False)
    # print(decoded)
    # Hello!<|eot_id|>
    # <|start_header_id|>assistant<|end_header_id|>
    # Hi there, how can I help?<|eot_id|>
    # <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    # Tell me a joke.<|eot_id|>
    # <|start_header_id|>assistant<|end_header_id|>
    # Why did the chicken cross the road?<|eot_id|>
    # <|start_header_id|>user<|end_header_id|>
    # To get to the other side.<|eot_id|>
    # <|start_header_id|>assistant<|end_header_id|>
    # No, the other side.<|eot_id|>
    tok_arr = np.asarray(example.tokens)
    loss_weight = np.asarray(example.loss_weight)

    start_hdr_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_hdr_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    assistant_ids: list[int] = tokenizer.encode("assistant", add_special_tokens=False)

    expected = np.zeros_like(loss_weight, dtype=loss_weight.dtype)

    # iterate over every position that holds <|start_header_id|>
    for idx in np.where(tok_arr == start_hdr_id)[0]:
        # pattern should be:
        # idx                -> <|start_header_id|>
        # idx+1 .. idx+k     -> "assistant" (one or more tokens)
        # idx+k+1            -> <|end_header_id|>
        # idx+k+2            -> newline
        k = len(assistant_ids)
        if idx + k + 2 >= len(tok_arr):
            continue  # out of bounds (shouldn't happen in valid template)

        if (
            np.array_equal(tok_arr[idx + 1 : idx + 1 + k], assistant_ids)
            and tok_arr[idx + 1 + k] == end_hdr_id
            and tok_arr[idx + 2 + k] == newline_id
        ):
            span_start = idx + 2 + k  # newline position (inclusive)

            # find next <|eot_id|>
            rel = np.where(tok_arr[span_start:] == eot_id)[0]
            assert rel.size, "assistant span not terminated by <|eot_id|>"
            span_end = span_start + int(rel[0])  # exclusive

            expected[span_start:span_end] = 1

    # Final check
    assert np.array_equal(loss_weight, expected), "loss_weight does not match assistant spans"


@pytest.mark.ray
def test_chat_dataset_build_and_pack(dummy_chat_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = tmpdir

        tokenizer = AutoTokenizer.from_pretrained(
            "stanford-crfm/marin-tokenizer", revision="49a09e626c220e9daae74124ea41be1bf5cd331d"
        )

        component = DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=[dummy_chat_data]),
            format=ChatLmDatasetFormat(messages_field="messages"),
            cache_dir=cache_dir,
        )

        processor = preprocessor_for_format(component.format, tokenizer)

        # test the processor
        source = component.source.get_shard_source("train")  # type: ignore
        processed = []
        for doc in source.open_shard(source.shard_names[0]):
            processed += processor([doc])

        assert len(processed) == 2

        # test the caching
        ds = build_lm_dataset_cache(cache_dir, source, component.format, tokenizer)
        ds_sync = ds.as_sync_dataset()
        assert len(ds_sync) == 2
        sample = next(iter(ds))

        # these are ProcessedChatDicts
        assert sample["assistant_masks"].shape == sample["input_ids"].shape
        assert 8 < sample["assistant_masks"].sum() <= 10
        # assert sample["input_ids"].shape[0] > 20
        assert (
            tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
            == "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHello!<|eot_id|>\n<|start_header_id|>assistant"
            "<|end_header_id|>\nHi there, how can I help?<|eot_id|>\n"
        )

        # now test packing
        Pos = hax.Axis("position", 100)
        packed_ds = ChatDataset(ds, Pos, max_segments_per_example=2)
        packed_ds = packed_ds.as_sync_dataset()

        assert len(packed_ds) == 1

        ex = packed_ds[0]
        assert ex.tokens.shape == (Pos.size,)
        assert ex.loss_weight.shape == (Pos.size,)
        assert ex.attn_mask.segment_ids[0].shape == (Pos.size,)

        assert_loss_weight_matches_all_assistants(ex, tokenizer)

        # test no packing
        packed_ds = ChatDataset(ds, Pos, max_segments_per_example=1).as_sync_dataset()

        # we supplied two conversations, so we should still have two examples
        assert len(packed_ds) == 2

        for ex in packed_ds:
            # basic structural checks
            assert ex.tokens.shape == (Pos.size,)
            assert ex.loss_weight.shape == (Pos.size,)
            assert ex.attn_mask.segment_ids[0].shape == (Pos.size,)

            # loss_weight should coincide with assistant tokens only
            assert_loss_weight_matches_all_assistants(ex, tokenizer)
