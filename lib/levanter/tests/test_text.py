# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax

from levanter.data.text import (
    BatchTokenizer,
    ChatLmDatasetFormat,
    ChatDataset,
    DatasetComponent,
    GrugLmExample,
    LmDataConfig,
    LmDatasetFormatBase,
    PreferenceChatLmDatasetFormat,
    PreferenceChatProcessor,
    PrebuiltLmDatasetFormat,
    TraceChatEvaluationFormat,
    TraceChatProcessor,
    TraceLmExample,
    UrlDatasetSourceConfig,
    build_lm_dataset_cache,
    build_trace_chat_dataset_cache,
    dataset_for_component,
    dataset_for_trace_chat_format,
    grug_lm_example_from_named,
    named_lm_example_from_grug,
    preprocessor_for_format,
)
from levanter.tokenizers import load_tokenizer
from levanter.models.lm_model import LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.schedule import BatchSchedule


_SIMPLE_TRACE_CHAT_TEMPLATE = """\
{%- for message in messages -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{%- if message['role'] == 'assistant' -%}
{% generation %}{{ message['content'] }}{% endgeneration %}
{%- else -%}
{{ message['content'] }}
{%- endif -%}
<|eot_id|>
{%- endfor -%}
"""


_TOOL_AWARE_TRACE_CHAT_TEMPLATE = """\
{%- if tools %}tools available{% endif -%}
{%- for message in messages -%}
{{ message.role }}:
{%- if message.role == 'assistant' -%}
{% generation %}{{ message.content }}{% endgeneration %}
{%- if message.tool_calls -%}
{% generation %}<tool_call>{{ message.tool_calls|length }}</tool_call>{% endgeneration %}
{%- endif -%}
{%- else -%}
{{ message.content }}
{%- endif -%}
{%- endfor -%}
"""


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


def _decode_masked_targets(tokenizer, tokens, loss_mask) -> str:
    tokens = np.asarray(tokens)
    loss_mask = np.asarray(loss_mask)
    target_positions = np.where(loss_mask > 0)[0] + 1
    target_positions = target_positions[target_positions < tokens.shape[0]]
    return tokenizer.decode(tokens[target_positions].tolist(), skip_special_tokens=False)


def test_trace_chat_processor_emits_named_masks(local_gpt2_marin_tokenizer):
    processor = TraceChatProcessor(
        local_gpt2_marin_tokenizer,
        chat_template=_SIMPLE_TRACE_CHAT_TEMPLATE,
        loss_tags=("assistant", "tool", "observation", "bash", "patch", "final_assistant"),
    )
    processed = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Fix the bug."},
                    {"role": "assistant", "content": "pytest -q", "loss_tags": ["bash"]},
                    {"role": "tool", "content": "test failed"},
                    {"role": "assistant", "content": "diff --git a/x.py b/x.py", "loss_tags": ["patch"]},
                ]
            }
        ]
    )[0]

    rendered = local_gpt2_marin_tokenizer.decode(processed["input_ids"], skip_special_tokens=False)
    assert "pytest -q" in rendered
    assert "test failed" in rendered
    assert "diff --git" in rendered

    masks = processed["trace_masks"]
    assert masks["assistant"].sum() > masks["patch"].sum()
    assert masks["tool"].sum() > 0
    assert masks["observation"].sum() == masks["tool"].sum()

    patch_tokens = local_gpt2_marin_tokenizer.decode(
        np.asarray(processed["input_ids"])[np.asarray(masks["patch"]) > 0].tolist(), skip_special_tokens=False
    )
    bash_tokens = local_gpt2_marin_tokenizer.decode(
        np.asarray(processed["input_ids"])[np.asarray(masks["bash"]) > 0].tolist(), skip_special_tokens=False
    )
    assert "diff --git" in patch_tokens
    assert "pytest -q" in bash_tokens


def test_trace_chat_processor_fills_common_chat_template_defaults(local_gpt2_marin_tokenizer):
    processor = TraceChatProcessor(
        local_gpt2_marin_tokenizer,
        chat_template=_TOOL_AWARE_TRACE_CHAT_TEMPLATE,
        loss_tags=("assistant", "tool_call"),
    )

    processed = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Call a tool."},
                    {"role": "assistant", "content": "I will do that."},
                ],
            }
        ]
    )[0]

    rendered = local_gpt2_marin_tokenizer.decode(processed["input_ids"], skip_special_tokens=False)
    assert "I will do that." in rendered
    assert processed["trace_masks"]["assistant"].sum() > 0
    assert processed["trace_masks"]["tool_call"].sum() == 0


def test_trace_chat_dataset_packs_and_shifts_masks(tmp_path, local_gpt2_marin_tokenizer):
    records = [
        {
            "messages": [
                {"role": "user", "content": "Fix A."},
                {"role": "assistant", "content": "pytest a", "loss_tags": ["bash"]},
                {"role": "tool", "content": "A failed"},
                {"role": "assistant", "content": "patch A", "loss_tags": ["patch"]},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Fix B."},
                {"role": "assistant", "content": "pytest b", "loss_tags": ["bash"]},
                {"role": "tool", "content": "B failed"},
                {"role": "assistant", "content": "patch B", "loss_tags": ["patch"]},
            ]
        },
    ]
    data_path = tmp_path / "traces.jsonl"
    with data_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    trace_format = TraceChatEvaluationFormat(
        messages_field="messages",
        chat_template=_SIMPLE_TRACE_CHAT_TEMPLATE,
        loss_tags=("bash", "patch", "tool", "observation"),
        pack=2,
    )
    source = UrlDatasetSourceConfig(train_urls=[str(data_path)], validation_urls=[]).get_shard_source("train")
    assert source is not None
    cache = build_trace_chat_dataset_cache(str(tmp_path / "cache"), source, trace_format, local_gpt2_marin_tokenizer)
    Pos = hax.Axis("position", 256)
    dataset = dataset_for_trace_chat_format(
        trace_format,
        Pos,
        cache,
        block_cross_document_attention=True,
    ).as_sync_dataset()

    example = dataset[0]
    assert isinstance(example, TraceLmExample)
    assert example.loss_masks.shape == (len(trace_format.loss_tags), Pos.size)
    assert example.attn_mask.segment_ids[0].shape == (Pos.size,)

    target_names = list(trace_format.loss_tags)
    bash_text = ""
    patch_text = ""
    for index in range(len(dataset)):
        example = dataset[index]
        bash_text += _decode_masked_targets(
            local_gpt2_marin_tokenizer,
            example.tokens,
            example.loss_masks[target_names.index("bash")],
        )
        patch_text += _decode_masked_targets(
            local_gpt2_marin_tokenizer,
            example.tokens,
            example.loss_masks[target_names.index("patch")],
        )
    assert "pytest a" in bash_text
    assert "pytest b" in bash_text
    assert "patch A" in patch_text
    assert "patch B" in patch_text


def test_trace_chat_dataset_right_slice_preserves_tail_masks(tmp_path, local_gpt2_marin_tokenizer):
    record = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": " ".join(["longtrace"] * 128)},
            {"role": "assistant", "content": "CORRECT", "loss_tags": ["outcome"]},
        ]
    }
    data_path = tmp_path / "right_slice_trace.jsonl"
    data_path.write_text(json.dumps(record) + "\n")

    trace_format = TraceChatEvaluationFormat(
        messages_field="messages",
        chat_template=_SIMPLE_TRACE_CHAT_TEMPLATE,
        loss_tags=("outcome",),
        pack=1,
        slice_strategy="right",
    )
    source = UrlDatasetSourceConfig(train_urls=[str(data_path)], validation_urls=[]).get_shard_source("train")
    assert source is not None
    cache = build_trace_chat_dataset_cache(str(tmp_path / "cache"), source, trace_format, local_gpt2_marin_tokenizer)
    Pos = hax.Axis("position", 32)
    dataset = dataset_for_trace_chat_format(
        trace_format,
        Pos,
        cache,
        block_cross_document_attention=True,
    ).as_sync_dataset()

    example = dataset[0]
    outcome_text = _decode_masked_targets(
        local_gpt2_marin_tokenizer,
        example.tokens,
        example.loss_masks[0],
    )
    assert "CORRECT" in outcome_text


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


@pytest.mark.ray
def test_chat_dataset_build_and_pack(dummy_chat_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = tmpdir

        tokenizer = load_tokenizer("stanford-crfm/marin-tokenizer")

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
