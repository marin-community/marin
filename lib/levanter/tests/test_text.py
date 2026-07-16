# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import tempfile
from pathlib import Path


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax

from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.data.text.cache import build_lm_dataset_cache
from levanter.data.text.datasets import (
    ChatDataset,
    DatasetComponent,
    LmDataConfig,
    UrlDatasetSourceConfig,
    count_corpus_sizes,
    dataset_for_component,
)
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named, named_lm_example_from_grug
from levanter.data.text.formats import (
    ChatLmDatasetFormat,
    LmDatasetFormatBase,
    PrebuiltLmDatasetFormat,
    SupervisedLmDatasetFormat,
    TextLmDatasetFormat,
    preprocessor_for_format,
)
from levanter.data.text.preference import PreferenceChatLmDatasetFormat, PreferenceChatProcessor
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


def test_count_corpus_sizes_handles_empty_train_cache(monkeypatch):
    class EmptyCache:
        def flat_field_length(self, _field):
            return 0

        async def async_flat_field_length(self, _field):
            return 0

        def flat_field_num_rows(self, _field):
            return 0

    config = LmDataConfig(
        components={"empty": DatasetComponent()},
        tokenizer="passthrough",
        vocab_size=64,
    )

    def build_caches(_self, split):
        if split == "train":
            return {"empty": EmptyCache()}
        return {}

    monkeypatch.setattr(LmDataConfig, "build_caches", build_caches)

    stats = count_corpus_sizes(config)

    prefix = "data/stats/train/empty/"
    assert stats[f"{prefix}total_tokens"] == 0
    assert stats[f"{prefix}total_docs"] == 0
    assert stats[f"{prefix}total_seqs"] == 0
    assert f"{prefix}padding_fraction" not in stats
    assert f"{prefix}truncation_fraction" not in stats


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


def test_long_string_workaround_matches_whole_encoding_across_many_chunks(local_gpt2_marin_tokenizer):
    """Cursor-based long-string splitting must match whole-string encoding over many chunks.

    ``_encode_long_string`` walks a cursor over the original text instead of
    re-slicing the unconsumed tail. This exercises hundreds of cursor advances
    (~650 chunks at ``_workaround_len=500``) and asserts the ids are byte-for-byte
    identical to the single-shot path, guarding the O(N) rewrite against drift.
    """
    tokenizer = local_gpt2_marin_tokenizer
    text = "lorem ipsum dolor sit amet " * 12_000  # ~324k chars

    split_tok = BatchTokenizer(tokenizer, _workaround_len=500, long_string_workaround=True)
    # _workaround_len above any realistic length => never splits => reference path.
    whole_tok = BatchTokenizer(tokenizer, _workaround_len=10**9, long_string_workaround=True)
    batch = [{"text": text}]

    assert split_tok(batch) == whole_tok(batch)


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


def test_build_caches_surfaces_component_failure(tmp_path):
    """A component that cannot be classified must raise out of build_caches, not
    be swallowed by the classification thread pool.

    #6954: a worker failure that fails to surface strands the process and, in a
    multi-host gang, silently desyncs the survivors into a multi-minute collective
    hang. build_caches classifies components concurrently, so this guards that a
    worker exception still propagates to the caller.
    """
    data_path = tmp_path / "docs.jsonl"
    with data_path.open("w") as f:
        f.write(json.dumps({"input_ids": [1, 2, 3, 4]}) + "\n")

    def component(cache_subdir: str) -> DatasetComponent:
        return DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=[str(data_path)], validation_urls=[]),
            format=PrebuiltLmDatasetFormat(),
            cache_dir=str(tmp_path / cache_subdir),
        )

    config = LmDataConfig(
        components={"a": component("a"), "b": component("b"), "c": component("c")},
        tokenizer="passthrough",
        vocab_size=16,
        auto_build_caches=False,  # a missing cache must raise, not build on the fly
    )

    with pytest.raises(FileNotFoundError):
        config.build_caches("train")


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


def test_supervised_text_cache_masks_target_tokens_for_training_and_eval(tmp_path):
    records = [
        {"input": "1 2 ", "target": "3 4"},
        {"input": "5 ", "target": "6 7 8"},
    ]
    data_path = tmp_path / "supervised.jsonl"
    with data_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    component = DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[str(data_path)], validation_urls=[str(data_path)]),
        format=SupervisedLmDatasetFormat(input_key="input", target_key="target"),
        cache_dir=str(tmp_path),
    )
    config = LmDataConfig(
        components={"supervised": component},
        tokenizer="passthrough",
        vocab_size=16,
    )

    train_cache = config.build_caches("train")["supervised"]
    first_row = train_cache.as_sync_dataset()[0]
    np.testing.assert_array_equal(first_row["input_ids"], np.array([1, 2, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(first_row["loss_weights"], np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32))

    Pos = hax.Axis("position", 4)
    train_example = config.train_sets(Pos, initial_batch_size=1, key=jax.random.PRNGKey(0))[
        "supervised"
    ].as_sync_dataset()[0]
    np.testing.assert_array_equal(np.asarray(train_example.tokens), np.array([1, 2, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(train_example.loss_weight), np.array([0.0, 1.0, 1.0, 0.0]))

    tagged_eval_sets = config.tagged_eval_sets(Pos)
    assert tagged_eval_sets[0][1] == ["supervised"]
    eval_example = tagged_eval_sets[0][0].as_sync_dataset()[0]
    np.testing.assert_array_equal(eval_example.tokens.array, np.array([1, 2, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(eval_example.loss_weight.array, np.array([0.0, 1.0, 1.0, 0.0]))


def test_supervised_text_processor_preserves_input_target_token_boundary():
    class BoundaryMergingTokenizer:
        name_or_path = "boundary-merging"
        vocab_size = 128
        bos_token_id = None
        eos_token_id = None
        pad_token_id = None
        bos_token = None
        eos_token = None
        chat_template = None

        def __len__(self):
            return self.vocab_size

        def encode(self, text, *, add_special_tokens=False):
            if text == "ab":
                return [99]
            if text == "a":
                return [1]
            if text == "b":
                return [2]
            raise ValueError(f"Unexpected text: {text!r}")

    processor = preprocessor_for_format(
        SupervisedLmDatasetFormat(input_key="input", target_key="target"),
        BoundaryMergingTokenizer(),  # type: ignore[arg-type]
        enforce_bos=False,
        enforce_eos=False,
    )

    row = processor([{"input": "a", "target": "b"}])[0]

    np.testing.assert_array_equal(row["input_ids"], np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(row["loss_weights"], np.array([1.0, 0.0], dtype=np.float32))


def test_supervised_text_packing_preserves_document_loss_boundaries(tmp_path):
    records = [
        {"input": "1 ", "target": "2"},
        {"input": "3 ", "target": "4"},
    ]
    data_path = tmp_path / "supervised_pack.jsonl"
    with data_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    component = DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[str(data_path)]),
        format=SupervisedLmDatasetFormat(input_key="input", target_key="target"),
        cache_dir=str(tmp_path),
        pack=2,
    )
    config = LmDataConfig(
        components={"supervised": component},
        tokenizer="passthrough",
        vocab_size=16,
    )

    cache = config.build_caches("train")["supervised"]
    Pos = hax.Axis("position", 4)
    dataset = dataset_for_component(
        component,
        Pos,
        cache,
        eos_id=None,
        block_cross_document_attention=config.block_cross_document_attention,
    ).as_sync_dataset()

    example = dataset[0]
    np.testing.assert_array_equal(np.asarray(example.tokens), np.array([1, 2, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(example.loss_weight), np.array([1.0, 0.0, 1.0, 0.0]))


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


# --- one example per document ----------------------------------------------
#
# A falsy pack (for chat/trace) and pack=1 select one document per example, padded to
# Pos. These tests drive the config-level dispatch (dataset_for_component /
# dataset_for_trace_chat_format) rather than the dataset classes directly, since the
# crash and bool/int coercion defects lived in that dispatch.


def _build_train_cache(component, tokenizer):
    source = component.source.get_shard_source("train")
    return build_lm_dataset_cache(component.cache_dir, source, component.format, tokenizer)


def assert_padding_never_contributes_loss(example):
    """The pad value must not leak into the objective.

    GreedyPrepackedDataset marks padding positions with segment id -1. Both a padding
    position and any position whose successor is padding (i.e. that would predict a pad
    token) must carry zero loss weight.
    """
    segment_ids = np.asarray(example.attn_mask.segment_ids[0])
    loss_weight = np.asarray(example.loss_weight)
    is_padding = segment_ids == -1
    assert is_padding.any(), "expected the document to be shorter than Pos, leaving padding"
    predicts_padding = np.roll(segment_ids, -1) == -1
    leaking = loss_weight[is_padding | predicts_padding]
    np.testing.assert_array_equal(leaking, np.zeros_like(leaking))


@pytest.mark.parametrize("pack", [False, 1])
def test_dataset_for_component_chat_unpacked_yields_one_padded_example_per_conversation(
    dummy_chat_data, tmp_path, pack
):
    tokenizer = load_tokenizer("marin-community/marin-tokenizer")
    component = DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[dummy_chat_data]),
        format=ChatLmDatasetFormat(messages_field="messages"),
        cache_dir=str(tmp_path),
        pack=pack,
    )
    cache = _build_train_cache(component, tokenizer)
    Pos = hax.Axis("position", 128)
    ds = dataset_for_component(
        component, Pos, cache, eos_id=None, block_cross_document_attention=True
    ).as_sync_dataset()

    # dummy_chat_data holds two conversations; unpacked mode never merges them
    assert len(ds) == 2
    for ex in ds:
        assert ex.tokens.shape == (Pos.size,)
        assert ex.loss_weight.shape == (Pos.size,)
        assert_padding_never_contributes_loss(ex)
        # assistant spans survive intact
        assert_loss_weight_matches_all_assistants(ex, tokenizer)


def test_dataset_for_component_supervised_unpacked_masks_padding_and_prompt(tmp_path):
    records = [
        {"input": "1 2 ", "target": "3 4"},
        {"input": "5 ", "target": "6 7 8"},
    ]
    data_path = tmp_path / "supervised_pad.jsonl"
    with data_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    component = DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[str(data_path)]),
        format=SupervisedLmDatasetFormat(input_key="input", target_key="target"),
        cache_dir=str(tmp_path),
        pack=1,
    )
    config = LmDataConfig(components={"s": component}, tokenizer="passthrough", vocab_size=16)
    cache = config.build_caches("train")["s"]
    Pos = hax.Axis("position", 8)
    ds = dataset_for_component(
        component, Pos, cache, eos_id=None, block_cross_document_attention=True
    ).as_sync_dataset()

    # one example per supervised record, none merged
    assert len(ds) == 2
    first = ds[0]
    np.testing.assert_array_equal(np.asarray(first.tokens)[:4], np.array([1, 2, 3, 4], dtype=np.int32))
    # loss on the target tokens only: prompt (0,1) and padding masked
    np.testing.assert_array_equal(np.asarray(first.loss_weight), np.array([0, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32))
    for ex in ds:
        assert ex.tokens.shape == (Pos.size,)
        assert_padding_never_contributes_loss(ex)


def test_dataset_for_component_text_unpacked_masks_padding(tmp_path):
    """Regression: one-document-per-example raw text must not train on padding.

    PackedTokenDataset supplies loss_weight=1 everywhere for raw text, so without the
    padding-aware masking the pad positions (and the final real token that would predict
    a pad token) would leak into the loss.
    """
    records = [{"text": "Hello world"}, {"text": "Short"}]
    data_path = tmp_path / "text_pad.jsonl"
    with data_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    component = DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[str(data_path)]),
        format=TextLmDatasetFormat(text_key="text"),
        cache_dir=str(tmp_path),
        pack=1,
    )
    tokenizer = load_tokenizer("marin-community/marin-tokenizer")
    cache = _build_train_cache(component, tokenizer)
    Pos = hax.Axis("position", 16)
    ds = dataset_for_component(
        component, Pos, cache, eos_id=None, block_cross_document_attention=True
    ).as_sync_dataset()

    assert len(ds) == 2
    for ex in ds:
        assert ex.tokens.shape == (Pos.size,)
        assert_padding_never_contributes_loss(ex)


# --- LmDataConfig.build_caches ---------------------------------------------


def _write_prebuilt_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _prebuilt_train_component(jsonl_path: Path) -> DatasetComponent:
    return DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[str(jsonl_path)], validation_urls=[]),
        format=PrebuiltLmDatasetFormat(),
    )


def test_build_caches_propagates_exception_from_one_component(tmp_path):
    p_good = tmp_path / "good.jsonl"
    _write_prebuilt_jsonl(p_good, [{"input_ids": [1, 2, 3, 4]}])
    good = _prebuilt_train_component(p_good)
    bad = DatasetComponent(
        source=None,
        cache_dir=str(tmp_path / "bad_missing"),
        format=PrebuiltLmDatasetFormat(),
    )
    config = LmDataConfig(
        components={"good": good, "bad": bad},
        cache_dir=str(tmp_path / "caches"),
        tokenizer="passthrough",
        vocab_size=16,
    )
    with pytest.raises(ValueError, match="No source and no cache"):
        config.build_caches("train")


def test_build_caches_rebuilds_on_unloadable_cache(tmp_path):
    """A cache dir that exists but won't load (no shard_ledger.json — the
    leftover of a cache build killed before it finished) must not crash-loop:
    with auto_build_caches on, build_caches catches the FileNotFoundError and
    falls through to rebuild rather than propagating it.
    """
    records = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5, 6, 7, 8]}]
    data_path = tmp_path / "data.jsonl"
    with data_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    component = DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[str(data_path)], validation_urls=[]),
        format=PrebuiltLmDatasetFormat(),
        cache_dir=str(tmp_path),
    )
    config = LmDataConfig(components={"c": component}, tokenizer="passthrough", vocab_size=16)
    assert config.auto_build_caches  # precondition: rebuild path is enabled

    config.build_caches("train")  # first build → complete cache

    # Reduce the cache dir to "exists but unloadable": keep the directory so
    # fsspec exists() is True, but empty it so load_lm_dataset_cache raises
    # FileNotFoundError (no ledger). This is the partial state a killed build
    # leaves behind.
    cache_path = next(tmp_path.glob("**/shard_ledger.json")).parent
    for child in cache_path.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    assert cache_path.exists() and not any(cache_path.iterdir())

    rebuilt = config.build_caches("train")["c"]
    assert (cache_path / "shard_ledger.json").exists(), "rebuild should restore the ledger"
    Pos = hax.Axis("position", 4)
    ds = dataset_for_component(
        component, Pos, rebuilt, eos_id=None, block_cross_document_attention=config.block_cross_document_attention
    ).as_sync_dataset()
    np.testing.assert_array_equal(np.asarray(ds[0].tokens), np.array(records[0]["input_ids"], dtype=np.int32))
