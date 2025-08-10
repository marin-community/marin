import json
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import haliax as hax

from levanter.data.text import (
    BatchTokenizer,
    ChatLmDatasetFormat,
    MultiturnChatDataset,
    SupervisedDataset,
    SupervisedLmDatasetFormat,
    UrlSingleDatasetLMConfig,
    build_lm_dataset_cache,
    preprocessor_for_format,
)
from levanter.models.lm_model import LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from tests.test_utils import skip_if_hf_model_not_accessible


def test_dont_blow_up_without_validation_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = UrlSingleDatasetLMConfig(
            train_urls=["kaa"],
            validation_urls=[],
            cache_dir=tmpdir,
            tokenizer="passthrough",
            vocab_size=64,
        )

        Pos = hax.Axis("position", 10)
        # mostly just making sure this doesn't blow up
        assert config.validation_set(Pos) is None


def test_lm_example_handles_ignore_id():
    Pos = hax.Axis("position", 10)
    Vocab = hax.Axis("vocab", Pos.size + 1)
    Embed = hax.Axis("embed", 10)
    tokens = hax.arange(Pos, dtype=jnp.int32)

    ignore_id = 6
    eos_id = 10

    ex_ignore = LmExample.causal(tokens, ignore_id=ignore_id, eos_id=eos_id)
    ex_no_ignore = LmExample.causal(tokens, eos_id=eos_id)
    assert ex_ignore.loss_mask[Pos, ignore_id - 1] == 0

    logits = hax.ones((Pos, Embed))
    lm_head = hax.zeros((Embed, Vocab))
    lm_head = lm_head.at[Vocab, ignore_id].set(-100)

    ignored_loss = maybe_fused_next_token_loss(
        Pos, Embed, Vocab, logits, lm_head, tokens, loss_mask=ex_ignore.loss_mask
    )
    no_ignore_loss = maybe_fused_next_token_loss(
        Pos, Embed, Vocab, logits, lm_head, tokens, loss_mask=ex_no_ignore.loss_mask
    )

    assert no_ignore_loss.item() >= ignored_loss.item() + 100 / Pos.size


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


def test_make_sequence_mask_basic_two_segments():
    make_output_mask = SupervisedDataset._make_sequence_mask
    segment_ids = np.array([0, 0, 1, 1, 1])
    segment_source_len = np.array([1, 2])
    expected = np.array([0, 1, 0, 0, 1], dtype=np.int32)
    assert_array_equal(make_output_mask(segment_ids, segment_source_len), expected)


def test_make_sequence_mask_all_input_tokens():
    make_output_mask = SupervisedDataset._make_sequence_mask
    segment_ids = np.array([0, 0, 1, 1])
    segment_source_len = np.array([2, 2])
    expected = np.array([0, 0, 0, 0], dtype=np.int32)
    assert_array_equal(make_output_mask(segment_ids, segment_source_len), expected)


def test_make_sequence_mask_all_output_tokens():
    make_output_mask = SupervisedDataset._make_sequence_mask
    segment_ids = np.array([0, 0, 1, 1])
    segment_source_len = np.array([0, 0])
    expected = np.array([1, 1, 1, 1], dtype=np.int32)
    assert_array_equal(make_output_mask(segment_ids, segment_source_len), expected)


def test_make_sequence_mask_alternating_inputs_outputs():
    make_output_mask = SupervisedDataset._make_sequence_mask
    segment_ids = np.array([0, 0, 0, 1, 1, 1])
    segment_source_len = np.array([2, 1])
    expected = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
    assert_array_equal(make_output_mask(segment_ids, segment_source_len), expected)


def test_make_sequence_mask_single_segment_mixed():
    make_output_mask = SupervisedDataset._make_sequence_mask
    segment_ids = np.array([0, 0, 0, 0])
    segment_source_len = np.array([2])
    expected = np.array([0, 0, 1, 1], dtype=np.int32)
    assert_array_equal(make_output_mask(segment_ids, segment_source_len), expected)


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


def assert_loss_mask_matches_all_assistants(example, tokenizer):
    """
    Assert that loss_mask == 1 exactly over assistant‑content spans.

    A span starts at the newline that follows
    "<|start_header_id|>assistant<|end_header_id|>"
    and ends just before the next "<|eot_id|>".
    """
    # ok we want to be sure we're predicting the assistant tokens
    # This is very fiddly, so we want to be careful.
    # In Levanter, the loss_mask is 1 for positions we compute loss on, 0 for positions we don't
    # that means we compute loss (have 1 loss mask) on the positions before each assistant token
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
    tok_arr = example.tokens.array
    loss_mask = example.loss_mask.array

    start_hdr_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_hdr_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    assistant_ids: List[int] = tokenizer.encode("assistant", add_special_tokens=False)

    expected = np.zeros_like(loss_mask, dtype=loss_mask.dtype)

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
    assert np.array_equal(loss_mask, expected), "loss_mask does not match assistant spans"


@pytest.mark.ray
def test_chat_dataset_build_and_pack(dummy_chat_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = tmpdir

        tokenizer = AutoTokenizer.from_pretrained(
            "stanford-crfm/marin-tokenizer", revision="49a09e626c220e9daae74124ea41be1bf5cd331d"
        )

        config = UrlSingleDatasetLMConfig(
            train_urls=[dummy_chat_data], format=ChatLmDatasetFormat(messages_field="messages")
        )

        processor = preprocessor_for_format(config.format, tokenizer)

        # test the processor
        source = config.get_shard_source("train")
        processed = []
        for doc in source.open_shard(source.shard_names[0]):
            processed += processor([doc])

        assert len(processed) == 2

        # test the caching
        ds = build_lm_dataset_cache(cache_dir, source, config.format, tokenizer)
        ds.await_finished()
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
        packed_ds = MultiturnChatDataset(ds, Pos, max_segments_per_example=2)
        packed_ds = packed_ds.as_sync_dataset()

        assert len(packed_ds) == 1

        ex = packed_ds[0]
        assert ex.tokens.axes == (Pos,)
        assert ex.loss_mask.axes == (Pos,)
        assert ex.attn_mask.segment_ids.axes == (Pos,)

        assert_loss_mask_matches_all_assistants(ex, tokenizer)

        # test no packing
        packed_ds = MultiturnChatDataset(ds, Pos, max_segments_per_example=1).as_sync_dataset()

        # we supplied two conversations, so we should still have two examples
        assert len(packed_ds) == 2

        for ex in packed_ds:
            # basic structural checks
            assert ex.tokens.axes == (Pos,)
            assert ex.loss_mask.axes == (Pos,)
            assert ex.attn_mask.segment_ids.axes == (Pos,)

            # loss_mask should coincide with assistant tokens only
            assert_loss_mask_matches_all_assistants(ex, tokenizer)


@pytest.fixture(scope="module")
def hf_tokenizer():
    return AutoTokenizer.from_pretrained(
        "stanford-crfm/marin-tokenizer",
        revision="49a09e626c220e9daae74124ea41be1bf5cd331d",
    )


@pytest.fixture
def dummy_supervised_file(tmp_path_factory) -> str:
    """Write two tiny supervised examples to jsonl and return the path."""
    data = [
        {"prompt": "Translate to French: Hello", "answer": "Bonjour"},
        {"prompt": "Translate to French: Yes", "answer": "Oui"},
    ]
    fp: Path = tmp_path_factory.mktemp("sup") / "sup.jsonl"
    with fp.open("w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    return str(fp)


@pytest.mark.ray
def test_supervised_processor_and_cache(dummy_supervised_file, hf_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = UrlSingleDatasetLMConfig(
            train_urls=[dummy_supervised_file],
            format=SupervisedLmDatasetFormat(
                input_field="prompt",
                output_field="answer",
                separate_with="\n",
            ),
        )

        source = cfg.get_shard_source("train")
        ds_cache = build_lm_dataset_cache(tmpdir, source, cfg.format, hf_tokenizer)
        ds_cache.await_finished()

        # there are two separate items, one per line
        sync_ds = ds_cache.as_sync_dataset()
        assert len(sync_ds) == 2

        for ex in sync_ds:
            # shape / type checks
            assert ex["input_ids"].ndim == 1
            assert ex["sources_len"].size == 1
            # specializing to the particular data: <begin_of_text>Translate to French: <single word>\n
            assert int(ex["sources_len"]) == 7

        # Now pack 2 conversations into one example
        Pos = hax.Axis("position", 128)
        packed_ds = SupervisedDataset(
            ds_cache,
            Pos,
            max_segments_per_example=2,
            mask_inputs=True,
        ).as_sync_dataset()

        # We had 2 examples, max_segments_per_example=2 → should pack into 1
        assert len(packed_ds) == 1
        ex: LmExample = packed_ds[0]

        # Axis checks
        assert ex.tokens.axes == (Pos,)
        assert ex.loss_mask.axes == (Pos,)
        assert ex.attn_mask.segment_ids.axes == (Pos,)

        # -----------------------------------------------------------
        #  Verify that for every segment:
        #    * leading tokens (input) have loss_mask==0
        #    * trailing tokens (answer) have loss_mask==1
        # -----------------------------------------------------------
        seg_ids: np.ndarray = ex.attn_mask.segment_ids.array
        mask: np.ndarray = ex.loss_mask.array

        for seg in np.unique(seg_ids):
            if seg < 0:  # skip padding segment (-1)
                continue

            idx = np.where(seg_ids == seg)[0]
            seg_mask = mask[idx]

            # Must contain at least one prompt and one answer token
            ones_idx = np.where(seg_mask == 1)[0]
            assert len(ones_idx) > 0, "segment has no answer‑token losses"
            first_one = ones_idx[0]

            # All positions before first answer token must be masked 0
            assert not seg_mask[:first_one].any(), "prompt tokens should have loss_mask == 0"
            # All positions from first answer token onward must be masked 1
            assert seg_mask[first_one:].all(), "answer tokens should have loss_mask == 1"

        # now try no packing

        packed_ds = SupervisedDataset(
            ds_cache,
            Pos,
            max_segments_per_example=1,
            mask_inputs=True,
        ).as_sync_dataset()

        # we supplied two conversations, so we should still have two examples
        assert len(packed_ds) == 2

        for idx, (raw_ex, ex) in enumerate(zip(sync_ds, packed_ds, strict=True)):
            # basic structural checks
            assert ex.tokens.axes == (Pos,)
            assert ex.loss_mask.axes == (Pos,)
            assert ex.attn_mask.segment_ids.axes == (Pos,)

            assert set(int(i) for i in np.unique(ex.attn_mask.segment_ids.array)) == {idx, -1}

            assert ex.loss_mask.array.sum() == len(ex.attn_mask.segment_ids.array) - raw_ex["sources_len"] - np.sum(
                ex.attn_mask.segment_ids.array == -1
            )
