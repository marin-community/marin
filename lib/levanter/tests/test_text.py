# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import os
import tempfile
from pathlib import Path


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax

from levanter.data import ListAsyncDataset
import levanter.data.text.datasets as text_datasets
from levanter.data.text import (
    BatchTokenizer,
    ChatLmDatasetFormat,
    ChatDataset,
    DatasetComponent,
    DirectDatasetComponent,
    GrugLmExample,
    HierarchicalMixtureDatasetComponent,
    LmDataConfig,
    LmDatasetFormatBase,
    PreferenceChatLmDatasetFormat,
    PreferenceChatProcessor,
    PrebuiltLmDatasetFormat,
    UrlDatasetSourceConfig,
    build_lm_dataset_cache,
    dataset_for_component,
    grug_lm_example_from_named,
    named_lm_example_from_grug,
    preprocessor_for_format,
)
from levanter.tokenizers import load_tokenizer
from levanter.models.lm_model import LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.schedule import BatchSchedule


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


def test_merge_split_encodings(local_gpt2_marin_tokenizer):
    tokenizer = local_gpt2_marin_tokenizer

    lorem = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""

    short_batch_tokenizer = BatchTokenizer(tokenizer, _workaround_len=len(lorem) // 3, long_string_workaround=True)

    batch_tokenizer = BatchTokenizer(tokenizer, _workaround_len=50000)
    batch = [{"text": lorem}]

    short_out = short_batch_tokenizer(batch)
    reg_out = batch_tokenizer(batch)

    assert short_out == reg_out


# ---------------------------------------------------------------------------
# BOS / EOS handling — regression tests for #5034
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def marin_tokenizer_with_bos():
    """Real marin-tokenizer (Llama-3 BPE, BOS-only TemplateProcessing post-processor)."""
    try:
        return load_tokenizer("marin-community/marin-tokenizer")
    except Exception as e:
        pytest.skip(f"Cannot load marin-community/marin-tokenizer: {e}")


def test_batch_tokenizer_prepends_bos(marin_tokenizer_with_bos):
    """Every doc in the output must start with BOS when enforce_bos=True.

    Regression test for https://github.com/marin-community/marin/issues/5034:
    BatchTokenizer silently stopped prepending BOS after the kitoken backend
    migration because the init probe ran with add_special_tokens=True but the
    hot-path encode_batch did not, inverting the prepend decision.
    """
    tokenizer = marin_tokenizer_with_bos
    bos_id = tokenizer.bos_token_id
    assert bos_id is not None

    bt = BatchTokenizer(tokenizer, enforce_bos=True, enforce_eos=False)
    batch = [{"text": "hello world"}, {"text": "Anonymous 01/04/19 posted"}, {"text": "x"}]
    out = bt(batch)

    for i, item in enumerate(out):
        ids = item["input_ids"]
        assert ids[0] == bos_id, f"doc {i}: expected BOS at position 0, got {ids[:5]}"
        assert ids.count(bos_id) == 1, f"doc {i}: expected exactly one BOS, got {ids.count(bos_id)}"


def test_batch_tokenizer_appends_eos(marin_tokenizer_with_bos):
    """Every doc must end with EOS when enforce_eos=True."""
    tokenizer = marin_tokenizer_with_bos
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None

    bt = BatchTokenizer(tokenizer, enforce_bos=False, enforce_eos=True)
    batch = [{"text": "hello world"}, {"text": "another doc"}]
    out = bt(batch)

    for i, item in enumerate(out):
        ids = item["input_ids"]
        assert ids[-1] == eos_id, f"doc {i}: expected EOS at end, got {ids[-5:]}"


def test_batch_tokenizer_bos_disabled(marin_tokenizer_with_bos):
    """With enforce_bos=False, doc must not start with BOS even though the tokenizer has one."""
    tokenizer = marin_tokenizer_with_bos
    bos_id = tokenizer.bos_token_id

    bt = BatchTokenizer(tokenizer, enforce_bos=False, enforce_eos=False)
    out = bt([{"text": "hello world"}])
    ids = out[0]["input_ids"]
    assert bos_id not in ids


def test_batch_tokenizer_output_equals_direct_encode_plus_bos(marin_tokenizer_with_bos):
    """BatchTokenizer(enforce_bos=True, enforce_eos=False) must equal [BOS] + encode(text)."""
    tokenizer = marin_tokenizer_with_bos
    bos_id = tokenizer.bos_token_id

    bt = BatchTokenizer(tokenizer, enforce_bos=True, enforce_eos=False)
    texts = ["hello world", "Anonymous 01/04/19", "The quick brown fox", "  leading space"]
    out = bt([{"text": t} for t in texts])

    for t, item in zip(texts, out):
        expected = [bos_id] + tokenizer.encode(t, add_special_tokens=False)
        assert item["input_ids"] == expected, f"text={t!r}: got {item['input_ids'][:10]}, expected {expected[:10]}"


def test_batch_tokenizer_metadata_reflects_user_intent(marin_tokenizer_with_bos):
    """metadata['append_bos'/'append_eos'] must track user intent, not internal detail.

    Post-migration caches stored append_bos=False despite enforce_bos=True; this
    invariant makes the metadata field a reliable signal for cache comparability.
    """
    tokenizer = marin_tokenizer_with_bos

    bt_on = BatchTokenizer(tokenizer, enforce_bos=True, enforce_eos=True)
    assert bt_on.metadata["append_bos"] is True
    assert bt_on.metadata["append_eos"] is True

    bt_off = BatchTokenizer(tokenizer, enforce_bos=False, enforce_eos=False)
    assert bt_off.metadata["append_bos"] is False
    assert bt_off.metadata["append_eos"] is False


def test_batch_tokenizer_bos_survives_long_string_split(marin_tokenizer_with_bos):
    """Long-string workaround must not duplicate or drop BOS across chunk boundaries."""
    tokenizer = marin_tokenizer_with_bos
    bos_id = tokenizer.bos_token_id

    lorem = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et "
        "dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip "
        "ex ea commodo consequat."
    )

    bt_split = BatchTokenizer(
        tokenizer,
        enforce_bos=True,
        enforce_eos=False,
        _workaround_len=len(lorem) // 3,
        long_string_workaround=True,
    )
    bt_whole = BatchTokenizer(
        tokenizer,
        enforce_bos=True,
        enforce_eos=False,
        _workaround_len=50000,
    )
    batch = [{"text": lorem}]

    split_out = bt_split(batch)
    whole_out = bt_whole(batch)

    assert split_out == whole_out
    ids = split_out[0]["input_ids"]
    assert ids[0] == bos_id
    assert ids.count(bos_id) == 1


def test_batch_tokenizer_no_bos_when_tokenizer_has_none(local_gpt2_marin_tokenizer):
    """Tokenizer without a BOS id: enforce_bos=True is a no-op (and doesn't crash)."""
    tokenizer = local_gpt2_marin_tokenizer
    assert tokenizer.bos_token_id is None

    bt = BatchTokenizer(tokenizer, enforce_bos=True, enforce_eos=False)
    assert bt.metadata["append_bos"] is False

    out = bt([{"text": "hello world"}])
    assert len(out[0]["input_ids"]) > 0


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


def test_hierarchical_component_defers_child_cache_loads_until_first_batch(tmp_path, monkeypatch):
    child_paths = []
    for index, tokens in enumerate(([1, 2, 3, 4], [5, 6, 7, 8])):
        data_path = tmp_path / f"prebuilt_{index}.jsonl"
        with data_path.open("w") as f:
            f.write(json.dumps({"input_ids": tokens}) + "\n")
        child_paths.append(data_path)

    hierarchical = HierarchicalMixtureDatasetComponent(
        components={
            "left": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[str(child_paths[0])], validation_urls=[]),
                format=PrebuiltLmDatasetFormat(),
                cache_dir=str(tmp_path / "cache_left"),
            ),
            "right": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[str(child_paths[1])], validation_urls=[]),
                format=PrebuiltLmDatasetFormat(),
                cache_dir=str(tmp_path / "cache_right"),
            ),
        },
        train_weights={"left": 0.5, "right": 0.5},
    )

    for child_component in hierarchical.components.values():
        source = child_component.source
        assert source is not None
        shard_source = source.get_shard_source("train")
        assert shard_source is not None
        build_lm_dataset_cache(
            os.path.join(child_component.cache_dir, "train"),
            shard_source,
            child_component.format,
            text_datasets.PassthroughTokenizer(16),
        )

    cache_calls: list[str] = []
    original_build_or_load = text_datasets.build_lm_dataset_cache

    def tracked_build_or_load(*args, **kwargs):
        cache_calls.append(args[0])
        return original_build_or_load(*args, **kwargs)

    monkeypatch.setattr("levanter.data.text.datasets.build_lm_dataset_cache", tracked_build_or_load)

    config = LmDataConfig(
        components={"group": hierarchical},
        train_weights={"group": 1.0},
        tokenizer="passthrough",
        vocab_size=16,
        shuffle=False,
    )
    Pos = hax.Axis("position", 4)

    train_sets = config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))

    assert cache_calls == []
    first = train_sets["group"].as_sync_dataset()[0]
    assert isinstance(first, GrugLmExample)
    assert len(cache_calls) == 1


def test_hierarchical_component_first_few_examples_stay_on_one_child_cache(tmp_path, monkeypatch):
    child_paths = []
    for index, tokens in enumerate(([1, 2, 3, 4], [5, 6, 7, 8])):
        data_path = tmp_path / f"clustered_{index}.jsonl"
        with data_path.open("w") as f:
            f.write(json.dumps({"input_ids": tokens}) + "\n")
        child_paths.append(data_path)

    hierarchical = HierarchicalMixtureDatasetComponent(
        components={
            "left": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[str(child_paths[0])], validation_urls=[]),
                format=PrebuiltLmDatasetFormat(),
                cache_dir=str(tmp_path / "clustered_cache_left"),
            ),
            "right": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[str(child_paths[1])], validation_urls=[]),
                format=PrebuiltLmDatasetFormat(),
                cache_dir=str(tmp_path / "clustered_cache_right"),
            ),
        },
        train_weights={"left": 0.5, "right": 0.5},
    )

    for child_component in hierarchical.components.values():
        source = child_component.source
        assert source is not None
        shard_source = source.get_shard_source("train")
        assert shard_source is not None
        build_lm_dataset_cache(
            os.path.join(child_component.cache_dir, "train"),
            shard_source,
            child_component.format,
            text_datasets.PassthroughTokenizer(16),
        )

    cache_calls: list[str] = []
    original_build_or_load = text_datasets.build_lm_dataset_cache

    def tracked_build_or_load(*args, **kwargs):
        cache_calls.append(args[0])
        return original_build_or_load(*args, **kwargs)

    monkeypatch.setattr("levanter.data.text.datasets.build_lm_dataset_cache", tracked_build_or_load)

    config = LmDataConfig(
        components={"group": hierarchical},
        train_weights={"group": 1.0},
        tokenizer="passthrough",
        vocab_size=16,
        shuffle=False,
    )
    Pos = hax.Axis("position", 4)

    train_sets = config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))
    sync_dataset = train_sets["group"].as_sync_dataset()

    for index in range(8):
        example = sync_dataset[index]
        assert isinstance(example, GrugLmExample)

    assert len(cache_calls) == 1


def test_hierarchical_component_uses_token_counts_for_simulated_epoching_without_loading_children(monkeypatch):
    hierarchical = HierarchicalMixtureDatasetComponent(
        components={
            "left": DatasetComponent(cache_dir="/tmp/does-not-exist-left"),
            "right": DatasetComponent(cache_dir="/tmp/does-not-exist-right"),
        },
        train_weights={"left": 0.5, "right": 0.5},
        token_counts={"left": 8192, "right": 4096},
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("load_lm_dataset_cache should not be called during hierarchical startup")

    monkeypatch.setattr("levanter.data.text.datasets.load_lm_dataset_cache", fail_load)

    config = LmDataConfig(
        components={"group": hierarchical},
        train_weights={"group": 1.0},
        tokenizer="gpt2",
        shuffle=False,
        experiment_budget=1024,
        target_budget=2048,
    )
    Pos = hax.Axis("position", 2048)

    train_sets = config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))

    assert train_sets["group"].is_finite()
    assert len(train_sets["group"].as_sync_dataset()) > 0


def test_simulated_epoching_fixed_subset_seed_stabilizes_direct_dataset_membership():
    direct = DirectDatasetComponent(datasets={"train": ListAsyncDataset(list(range(40)))})
    config = LmDataConfig(
        components={"group": direct},
        train_weights={"group": 1.0},
        tokenizer="passthrough",
        vocab_size=64,
        shuffle=True,
        experiment_budget=20,
        target_budget=40,
    )
    fixed_subset_config = dataclasses.replace(config, simulated_epoch_subset_seed=97)
    Pos = hax.Axis("position", 4)

    run0 = config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))["group"].as_sync_dataset()
    run1 = config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(1))["group"].as_sync_dataset()
    fixed_run0 = fixed_subset_config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))[
        "group"
    ].as_sync_dataset()
    fixed_run1 = fixed_subset_config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(1))[
        "group"
    ].as_sync_dataset()

    run0_values = [run0[i] for i in range(len(run0))]
    run1_values = [run1[i] for i in range(len(run1))]
    fixed_run0_values = [fixed_run0[i] for i in range(len(fixed_run0))]
    fixed_run1_values = [fixed_run1[i] for i in range(len(fixed_run1))]

    assert set(run0_values) != set(run1_values)
    assert set(fixed_run0_values) == set(fixed_run1_values)
    assert fixed_run0_values != fixed_run1_values


def test_simulated_epoching_fixed_subset_seed_stabilizes_hierarchical_membership(monkeypatch):
    hierarchical = HierarchicalMixtureDatasetComponent(
        components={
            "left": DatasetComponent(cache_dir="/tmp/does-not-exist-left"),
            "right": DatasetComponent(cache_dir="/tmp/does-not-exist-right"),
        },
        train_weights={"left": 0.5, "right": 0.5},
        token_counts={"left": 20, "right": 20},
    )
    config = LmDataConfig(
        components={"group": hierarchical},
        train_weights={"group": 1.0},
        tokenizer="passthrough",
        vocab_size=64,
        shuffle=True,
        experiment_budget=8,
        target_budget=40,
    )
    fixed_subset_config = dataclasses.replace(config, simulated_epoch_subset_seed=97)
    Pos = hax.Axis("position", 1)

    original_build = text_datasets.LmDataConfig._build_token_dataset_for_component

    def fake_build(self, name, component, Pos, *, split, caches):
        if isinstance(component, DatasetComponent):
            if name.endswith("/left"):
                return ListAsyncDataset(list(range(100, 120)))
            if name.endswith("/right"):
                return ListAsyncDataset(list(range(200, 220)))
        return original_build(self, name, component, Pos, split=split, caches=caches)

    monkeypatch.setattr(text_datasets.LmDataConfig, "_build_token_dataset_for_component", fake_build)

    run0 = config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))["group"].as_sync_dataset()
    run1 = config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(1))["group"].as_sync_dataset()
    fixed_run0 = fixed_subset_config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))[
        "group"
    ].as_sync_dataset()
    fixed_run1 = fixed_subset_config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(1))[
        "group"
    ].as_sync_dataset()

    run0_values = [run0[i] for i in range(len(run0))]
    run1_values = [run1[i] for i in range(len(run1))]
    fixed_run0_values = [fixed_run0[i] for i in range(len(fixed_run0))]
    fixed_run1_values = [fixed_run1[i] for i in range(len(fixed_run1))]

    assert set(run0_values) != set(run1_values)
    assert set(fixed_run0_values) == set(fixed_run1_values)
    assert fixed_run0_values != fixed_run1_values


def test_hierarchical_component_uses_metadata_length_when_tiny_child_weight_rounds_to_zero(monkeypatch):
    hierarchical = HierarchicalMixtureDatasetComponent(
        components={
            "large": DatasetComponent(cache_dir="/tmp/does-not-exist-large"),
            "tiny": DatasetComponent(cache_dir="/tmp/does-not-exist-tiny"),
        },
        train_weights={"large": 40_000, "tiny": 4},
        token_counts={"large": 40_000, "tiny": 4},
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("load_lm_dataset_cache should not be called during hierarchical startup")

    monkeypatch.setattr("levanter.data.text.datasets.load_lm_dataset_cache", fail_load)

    config = LmDataConfig(
        components={"group": hierarchical},
        train_weights={"group": 1.0},
        tokenizer="gpt2",
        shuffle=False,
        experiment_budget=1024,
        target_budget=2048,
        mixture_block_size=2048,
    )
    Pos = hax.Axis("position", 4)

    train_sets = config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))

    assert train_sets["group"].is_finite()
    assert len(train_sets["group"].as_sync_dataset()) == 5000


def test_hierarchical_component_skips_empty_validation_split(tmp_path):
    train_path = tmp_path / "train.jsonl"
    with train_path.open("w") as f:
        f.write(json.dumps({"input_ids": [1, 2, 3, 4]}) + "\n")

    hierarchical = HierarchicalMixtureDatasetComponent(
        components={
            "left": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[str(train_path)], validation_urls=[]),
                format=PrebuiltLmDatasetFormat(),
                cache_dir=str(tmp_path / "cache_left"),
            ),
            "right": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[str(train_path)], validation_urls=[]),
                format=PrebuiltLmDatasetFormat(),
                cache_dir=str(tmp_path / "cache_right"),
            ),
        },
        train_weights={"left": 0.5, "right": 0.5},
    )

    config = LmDataConfig(
        components={"group": hierarchical},
        train_weights={"group": 1.0},
        tokenizer="passthrough",
        vocab_size=16,
        shuffle=False,
    )
    Pos = hax.Axis("position", 4)

    assert config.validation_sets(Pos) == {}


def test_hierarchical_component_validation_uses_available_children_only(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    with train_path.open("w") as f:
        f.write(json.dumps({"input_ids": [1, 2, 3, 4]}) + "\n")
    with validation_path.open("w") as f:
        f.write(json.dumps({"input_ids": [5, 6, 7, 8]}) + "\n")

    hierarchical = HierarchicalMixtureDatasetComponent(
        components={
            "left": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[str(train_path)], validation_urls=[]),
                format=PrebuiltLmDatasetFormat(),
                cache_dir=str(tmp_path / "cache_left"),
            ),
            "right": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[str(train_path)], validation_urls=[str(validation_path)]),
                format=PrebuiltLmDatasetFormat(),
                cache_dir=str(tmp_path / "cache_right"),
            ),
        },
        train_weights={"left": 0.5, "right": 0.5},
    )

    config = LmDataConfig(
        components={"group": hierarchical},
        train_weights={"group": 1.0},
        tokenizer="passthrough",
        vocab_size=16,
        shuffle=False,
    )

    validation_sets = config.validation_grug_sets(seq_len=4)

    assert set(validation_sets) == {"group"}
    dataset = validation_sets["group"]
    assert dataset.is_finite()
    assert len(dataset.as_sync_dataset()) == 1
    assert isinstance(dataset.as_sync_dataset()[0], GrugLmExample)


def test_dataset_for_component_rejects_preference_format():
    component = DatasetComponent(format=PreferenceChatLmDatasetFormat())
    Pos = hax.Axis("position", 8)
    with pytest.raises(ValueError, match="Unknown format"):
        dataset_for_component(
            component,
            Pos,
            None,  # type: ignore[arg-type]
            eos_id=None,
            block_cross_document_attention=True,
        )


def test_preprocessor_for_format_dispatches_preference_format():
    class _DummyTokenizer:
        chat_template = "{% generation %}"
        name_or_path = "dummy"
        vocab_size = 128
        bos_token_id = None
        eos_token_id = None
        pad_token_id = None
        bos_token = None
        eos_token = None

        def __len__(self):
            return 128

        def encode(self, text, *, add_special_tokens=False):
            return [11, 12, 13]

        def decode(self, ids, *, skip_special_tokens=False):
            return "dummy"

        def encode_batch(self, texts, *, add_special_tokens=False):
            return [[11, 12, 13] for _ in texts]

        def get_vocab(self):
            return {}

        def apply_chat_template(self, conversation, *, tokenize=True, add_generation_prompt=False, **kwargs):
            return [11, 12, 13]

        def apply_chat_template_with_masks(self, conversations, *, chat_template=None, **kwargs):
            return {
                "input_ids": [[11, 12, 13] for _ in conversations],
                "assistant_masks": [[0, 1, 1] for _ in conversations],
            }

    tokenizer = _DummyTokenizer()
    format = PreferenceChatLmDatasetFormat()

    processor = preprocessor_for_format(format, tokenizer)  # type: ignore[arg-type]

    assert isinstance(processor, PreferenceChatProcessor)

    output = processor(
        [
            {
                "chosen": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
                "rejected": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "No"},
                ],
            }
        ]
    )

    assert len(output) == 1
    row = output[0]
    assert set(row.keys()) == {
        "chosen_input_ids",
        "chosen_assistant_masks",
        "rejected_input_ids",
        "rejected_assistant_masks",
    }
    assert row["chosen_input_ids"].shape == row["chosen_assistant_masks"].shape
    assert row["rejected_input_ids"].shape == row["rejected_assistant_masks"].shape


def test_preprocessor_for_format_rejects_unknown_format():
    class _UnknownFormat(LmDatasetFormatBase):
        pass

    class _DummyTokenizer:
        chat_template = "{% generation %}"
        name_or_path = "dummy"

        def __len__(self):
            return 128

    with pytest.raises(ValueError, match="Unknown format"):
        preprocessor_for_format(_UnknownFormat(), _DummyTokenizer())  # type: ignore[arg-type]


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

    vocab = tokenizer.get_vocab()
    start_hdr_id = vocab["<|start_header_id|>"]
    end_hdr_id = vocab["<|end_header_id|>"]
    eot_id = vocab["<|eot_id|>"]
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

        tokenizer = load_tokenizer("marin-community/marin-tokenizer")

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
        expected_rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there, how can I help?"},
            ],
            tokenize=False,
        )
        assert tokenizer.decode(sample["input_ids"], skip_special_tokens=False) == expected_rendered

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
