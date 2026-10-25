# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Account for supervised chat targets without changing rows or their sampling order."""

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.tokenizers import MarinTokenizer


@dataclass(frozen=True)
class RowIdentity:
    occurrence_id: str
    sample_id: str
    prompt_id: str
    final_label: str
    history_label: str


@dataclass(frozen=True)
class TurnTokens:
    message_index: int
    label: str
    supervised_tokens: int


@dataclass(frozen=True)
class RowAccounting:
    identity: RowIdentity
    rendered_tokens: int
    template_sha256: str
    input_ids_sha256: str
    target_mask_sha256: str
    turns: tuple[TurnTokens, ...]


@dataclass(frozen=True)
class CoefficientMass:
    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None


@dataclass(frozen=True)
class ClassAccounting:
    occurrences_with_targets: int
    unique_samples: int
    unique_prompts: int
    supervised_tokens: int
    ce_coefficient_mass: CoefficientMass


@dataclass(frozen=True)
class ScopeAccounting:
    occurrences: int
    unique_occurrences: int
    unique_samples: int
    unique_prompts: int
    final_class_rows: dict[str, int]
    rendered_tokens: int
    supervised_tokens: int
    classes: dict[str, ClassAccounting]


@dataclass(frozen=True)
class OrderedAccounting:
    ordered_occurrence_ids: tuple[str, ...]
    pooled: ScopeAccounting
    updates: tuple[ScopeAccounting, ...]
    mean_update_ce_coefficient_mass: dict[str, CoefficientMass]


def account_chat_row(
    row: dict[str, Any],
    identity: RowIdentity,
    *,
    tokenizer: MarinTokenizer,
    chat_format: ChatLmDatasetFormat,
    context_length: int,
    expected_template_sha256: str,
) -> RowAccounting:
    """Count exact causal targets, labeling prior and final assistant turns separately.

    Uses the training processor and public message-span instrumentation, requiring
    identical token IDs and masks. Only non-packed, untruncated, assistant-masked
    conversations ending in an assistant reply are supported. Labels are caller
    provenance, not inferred intent. No input fields are changed.
    """
    if not all(
        (identity.occurrence_id, identity.sample_id, identity.prompt_id, identity.final_label, identity.history_label)
    ):
        raise ValueError("Occurrence, sample, prompt IDs and both labels must be nonempty")
    if (
        context_length <= 0
        or chat_format.pack is not False
        or chat_format.slice_strategy != "raise"
        or not chat_format.mask_user_turns
    ):
        raise ValueError("Accounting requires positive context, pack=false, slice=raise and assistant masking")
    template = chat_format.chat_template or tokenizer.chat_template
    if template is None or hashlib.sha256(template.encode()).hexdigest() != expected_template_sha256:
        raise ValueError("Chat template does not match the pinned hash")
    messages = list(row[chat_format.messages_field])
    if chat_format.system_prompt is not None and row.get(chat_format.system_prompt) is not None:
        system = row[chat_format.system_prompt]
        if isinstance(system, Mapping):
            system_message = {**system, "role": "system"}
            if "content" not in system_message:
                raise ValueError("System prompt mapping must include content")
        else:
            system_message = {"role": "system", "content": system}
        messages = [system_message, *messages]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Conversation must end with the final assistant target")
    raw_kwargs = row.get(chat_format.chat_template_kwargs) if chat_format.chat_template_kwargs is not None else None
    if raw_kwargs is not None and not isinstance(raw_kwargs, Mapping):
        raise ValueError("Chat template kwargs must be a mapping")
    kwargs = dict(raw_kwargs or {})
    if any(key in kwargs for key in ("tokenize", "return_assistant_tokens_mask", "return_dict", "return_message_spans")):
        raise ValueError("Row overrides a tokenizer accounting control")
    if kwargs.pop("chat_template", template) != template:
        raise ValueError("Row overrides the pinned chat template")
    kwargs.pop("add_generation_prompt", None)
    processed = chat_format.build_preprocessor(tokenizer)([row])[0]
    traced = tokenizer.apply_chat_template_with_masks(
        [messages], chat_template=template, return_message_spans=True, **kwargs
    )
    ids = np.asarray(processed["input_ids"], dtype=np.int32)
    mask = np.asarray(processed["assistant_masks"], dtype=np.int32)
    if not np.array_equal(ids, traced["input_ids"][0]) or not np.array_equal(mask, traced["assistant_masks"][0]):
        raise ValueError("Message instrumentation changed training IDs or masks")
    if len(ids) > context_length:
        raise ValueError(f"Rendered length {len(ids)} exceeds context {context_length}")
    if mask.shape != ids.shape or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Invalid assistant mask")
    spans = traced["message_spans"][0]
    if len(spans) != len(messages):
        raise ValueError("Message-span coverage mismatch")
    owners = np.full(len(ids), -1, dtype=np.int32)
    for index, (message, (start, end)) in enumerate(zip(messages, spans, strict=True)):
        if not 0 <= start <= end <= len(ids):
            raise ValueError("Message span is outside the rendered sequence")
        positions = np.arange(start, end)[mask[start:end] != 0]
        if len(positions) and message.get("role") != "assistant":
            raise ValueError("Non-assistant message has supervised targets")
        if np.any(owners[positions] != -1):
            raise ValueError("Supervised token belongs to overlapping message spans")
        owners[positions] = index
    if not np.array_equal(owners != -1, mask != 0):
        raise ValueError("Some supervised targets have no message owner")
    target_mask = np.zeros(context_length, dtype=np.int32)
    target_mask[: max(0, len(mask) - 1)] = mask[1:]
    effective_owners = owners[1:][mask[1:] != 0]
    turns = tuple(
        TurnTokens(
            index,
            identity.final_label if index == len(messages) - 1 else identity.history_label,
            int(np.sum(effective_owners == index)),
        )
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    )
    if not target_mask.sum() or not turns[-1].supervised_tokens:
        raise ValueError("Conversation or final assistant has no effective supervised targets")
    return RowAccounting(
        identity,
        len(ids),
        expected_template_sha256,
        hashlib.sha256(ids.astype("<i4").tobytes()).hexdigest(),
        hashlib.sha256(target_mask.astype("<i4").tobytes()).hexdigest(),
        turns,
    )


def _scope(rows: Sequence[RowAccounting], labels: Sequence[str]) -> ScopeAccounting:
    totals = Counter[str]()
    per_row = []
    for row in rows:
        counts = Counter[str]()
        for turn in row.turns:
            counts[turn.label] += turn.supervised_tokens
        totals.update(counts)
        per_row.append(counts)
    total = sum(totals.values())
    classes = {}
    for label in labels:
        participating = [row.identity for row, counts in zip(rows, per_row, strict=True) if counts[label] > 0]
        classes[label] = ClassAccounting(
            len(participating),
            len({row.sample_id for row in participating}),
            len({row.prompt_id for row in participating}),
            totals[label],
            CoefficientMass(totals[label], total),
        )
    return ScopeAccounting(
        len(rows),
        len({row.identity.occurrence_id for row in rows}),
        len({row.identity.sample_id for row in rows}),
        len({row.identity.prompt_id for row in rows}),
        dict(Counter(row.identity.final_label for row in rows)),
        sum(row.rendered_tokens for row in rows),
        total,
        classes,
    )


def account_ordered_occurrences(
    rows: Sequence[RowAccounting], ordered_occurrence_ids: Sequence[str], *, batch_size: int
) -> OrderedAccounting:
    """Report pooled and per-update CE masses for an explicitly supplied occurrence stream.

    Repeated IDs in the stream represent repeated exposure. Input row IDs must be
    unique. This function neither samples nor establishes that this order is the
    trainer's order. CE coefficient mass is not gradient share. Empty rates are
    undefined; incomplete batches and unknown IDs fail rather than being padded.
    """
    if batch_size <= 0 or len(ordered_occurrence_ids) % batch_size:
        raise ValueError("Require a positive batch size and complete updates")
    by_id = {row.identity.occurrence_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate materialized occurrence ID")
    unknown = set(ordered_occurrence_ids) - by_id.keys()
    if unknown:
        raise ValueError(f"Unknown occurrence IDs: {sorted(unknown)}")
    selected = [by_id[identity] for identity in ordered_occurrence_ids]
    labels = sorted({turn.label for row in rows for turn in row.turns})
    updates = tuple(
        _scope(selected[start : start + batch_size], labels) for start in range(0, len(selected), batch_size)
    )
    if any(update.supervised_tokens <= 0 for update in updates):
        raise ValueError("An update has no supervised targets")
    means = {}
    for label in labels:
        if not updates:
            means[label] = CoefficientMass(0, 0)
            continue
        mean = sum(
            (Fraction(update.classes[label].supervised_tokens, update.supervised_tokens) for update in updates),
            Fraction(),
        ) / len(updates)
        means[label] = CoefficientMass(mean.numerator, mean.denominator)
    return OrderedAccounting(tuple(ordered_occurrence_ids), _scope(selected, labels), updates, means)
