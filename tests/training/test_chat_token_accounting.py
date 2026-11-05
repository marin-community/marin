# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
from typing import Any

import numpy as np
import pytest
from haliax import Axis
from levanter.data.text.datasets import DatasetComponent, dataset_for_component
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.store.cache import SerialCacheWriter
from marin.training.chat_token_accounting import (
    RowAccounting,
    RowIdentity,
    TurnTokens,
    account_chat_row,
    account_ordered_occurrences,
)

TEMPLATE = """{{ bos_token }}{% for message in messages %}
[{{ message.role }}]
{% if message.role == 'assistant' %}{% generation %}{{ message.get('reasoning_content', '') }}
{{ message.content }}{{ eos_token }}{% endgeneration %}
{% else %}{{ message.content }}{% endif %}{% endfor %}"""


@pytest.mark.parametrize("format_pack", [False, True])
def test_multiturn_counts_match_actual_causal_loss_and_preserve_inputs(gpt2_tokenizer, tmp_path, format_pack):
    chat_format = ChatLmDatasetFormat(chat_template=TEMPLATE, pack=format_pack, system_prompt="system")
    component = DatasetComponent(format=chat_format, pack=False)
    row: dict[str, Any] = {
        "system": "Be concise.",
        "messages": [
            {"role": "user", "content": "Add two and three."},
            {"role": "assistant", "reasoning_content": "Adding gives five.", "content": "Five."},
            {"role": "user", "content": "Name a color."},
            {"role": "assistant", "content": "Blue."},
        ],
    }
    original = copy.deepcopy(row)
    identity = RowIdentity("occ", "sample", "prompt", "target", "history")
    result = account_chat_row(
        row,
        identity,
        tokenizer=gpt2_tokenizer,
        component=component,
        context_length=256,
        expected_template_sha256=hashlib.sha256(TEMPLATE.encode()).hexdigest(),
    )
    processed = chat_format.build_preprocessor(gpt2_tokenizer)([row])[0]
    with SerialCacheWriter(str(tmp_path / "cache"), processed) as writer:
        writer.write_batch([processed])
    dataset = dataset_for_component(
        component, Axis("position", 256), writer.result(), eos_id=None, block_cross_document_attention=True
    ).as_sync_dataset()
    example = dataset[0]
    assert sum(turn.supervised_tokens for turn in result.turns) == int(example.loss_weight.sum())
    assert (
        result.target_mask_sha256 == hashlib.sha256(np.asarray(example.loss_weight, dtype="<i4").tobytes()).hexdigest()
    )
    assert [turn.label for turn in result.turns] == ["history", "target"]
    assert all(turn.supervised_tokens > 0 for turn in result.turns)
    assert result.turns[0].supervised_tokens > result.turns[1].supervised_tokens
    assert row == original


def test_component_pack_override_rejects_accounting_for_packed_updates(gpt2_tokenizer, tmp_path):
    chat_format = ChatLmDatasetFormat(chat_template=TEMPLATE, pack=False, slice_strategy="raise")
    component = DatasetComponent(format=chat_format, pack=True)
    row = {"messages": [{"role": "user", "content": "Add."}, {"role": "assistant", "content": "Two."}]}
    processed = chat_format.build_preprocessor(gpt2_tokenizer)([row])[0]
    with SerialCacheWriter(str(tmp_path / "cache"), processed) as writer:
        writer.write_batch([processed, processed])
    dataset = dataset_for_component(
        component, Axis("position", 256), writer.result(), eos_id=None, block_cross_document_attention=True
    ).as_sync_dataset()
    # Two occurrences occupy one packed sequence, so row-counted updates would be wrong.
    assert len(dataset) == 1
    segments = np.asarray(dataset[0].attn_mask.segment_ids[0])
    assert set(segments[segments >= 0]) == {0, 1}
    with pytest.raises(ValueError, match="unpacked"):
        account_chat_row(
            row,
            RowIdentity("occ", "sample", "prompt", "target", "history"),
            tokenizer=gpt2_tokenizer,
            component=component,
            context_length=256,
            expected_template_sha256=hashlib.sha256(TEMPLATE.encode()).hexdigest(),
        )


def test_ordered_mass_is_not_mean_of_row_fractions_or_gradient_share():
    short = RowAccounting(
        RowIdentity("short", "sample1", "prompt", "id", "replay"),
        8,
        "pin",
        "ids1",
        "mask1",
        (TurnTokens(1, "id", 1), TurnTokens(3, "replay", 1)),
    )
    long = RowAccounting(
        RowIdentity("long", "sample2", "prompt", "replay", "replay"),
        80,
        "pin",
        "ids2",
        "mask2",
        (TurnTokens(1, "replay", 18),),
    )
    report = account_ordered_occurrences([short, long], ["short", "short", "long", "long"], batch_size=2)
    assert report.pooled.supervised_tokens == 40
    assert report.pooled.classes["id"].ce_coefficient_mass.value == 2 / 40
    assert [step.classes["id"].ce_coefficient_mass.value for step in report.updates] == [0.5, 0]
    assert report.mean_update_ce_coefficient_mass["id"].value == 0.25
    assert report.pooled.unique_occurrences == 2
    assert report.pooled.unique_samples == 2
    assert report.pooled.unique_prompts == 1
    assert report.pooled.classes["id"].occurrences_with_targets == 2
    assert report.pooled.classes["id"].unique_samples == 1
    empty = account_ordered_occurrences([short, long], [], batch_size=8)
    assert empty.pooled.classes["id"].ce_coefficient_mass.value is None
    assert empty.mean_update_ce_coefficient_mass["id"].value is None
