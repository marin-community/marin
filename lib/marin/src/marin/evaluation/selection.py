# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a reproducible evaluation cohort without truncating canonical prompts."""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict
from taskcompendium.evaluation import (
    ChatGeneration,
    EvaluationProtocol,
    GraphWalksContextSelection,
    MRCRContextSelection,
    TextGeneration,
)
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.evaluation import EvaluationCase, evaluation_cases, evaluation_items
from taskcompendium.submission import conversation_messages
from verifyit.adapters.evalchemy_long_context import (
    GRAPHWALKS_CONTEXT_MARGIN,
    GRAPHWALKS_PREFIX_CHARS_PER_TOKEN,
    GRAPHWALKS_PREFIX_TOKEN_MARGIN,
    graphwalks_output_budget,
    interleave_mrcr_indices,
)

SELECTIONS_TABLE = "evaluation_selection"
CONTEXT_MARGIN = 64


class EvaluationTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def apply_chat_template(
        self, conversation: list[dict[str, Any]], *, tokenize: bool, add_generation_prompt: bool, **kwargs: Any
    ) -> list[int]: ...


class SelectionStatus(StrEnum):
    SELECTED = "selected"
    EXCLUDED = "excluded"
    UNINSPECTED = "uninspected"


class Selection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    source_row: str
    source_ordinal: int
    status: SelectionStatus
    reason: str | None
    rank: int | None
    context_window: int
    max_tokens: int | None
    prompt_tokens: int | None
    reference_tokens: int | None


@dataclass(frozen=True)
class SelectionDecision:
    record: Selection
    case: EvaluationCase | None


def wire_messages(task: TaskSpec, operation: ChatGeneration) -> list[dict[str, Any]]:
    """Render source transport fields that do not change the canonical conversation."""
    messages = conversation_messages(task.context.events)
    if operation.historical_assistant_reasoning == "empty":
        messages = [
            {**message, "reasoning_content": ""} if message["role"] == "assistant" else message for message in messages
        ]
    return messages


def prompt_variant(
    case: EvaluationCase, context_window: int, max_tokens: int, tokenizer: EvaluationTokenizer | None
) -> TaskSpec:
    operation = case.item.operation
    if not isinstance(operation, ChatGeneration) or operation.prompt_selection == "fixed":
        return case.task
    if tokenizer is None:
        raise ValueError("Prompt selection requires the run's pinned tokenizer")
    for variant in sorted(operation.variants, key=lambda variant: variant.fewshot, reverse=True):
        text = "".join(message["content"] for message in conversation_messages(variant.context.events))
        # The source stops reducing examples at zero shots. Ordinary context preflight
        # then lowers the output cap or excludes a prompt that cannot fit at all.
        if variant.fewshot == 0 or len(tokenizer.encode(text, add_special_tokens=False)) < context_window - max_tokens:
            return case.task.model_copy(update={"context": variant.context})
    raise ValueError(f"No declared prompt variant fits the model context: {case.task.id}")


def prompt_tokens(
    case: EvaluationCase,
    protocol: EvaluationProtocol,
    tokenizer: EvaluationTokenizer,
    *,
    prefix_chars: int | None = None,
) -> int:
    operation = case.item.operation
    if isinstance(operation, TextGeneration):
        return len(tokenizer.encode(operation.context, add_special_tokens=False))
    assert isinstance(operation, ChatGeneration)
    assert protocol.sampling is not None
    messages = wire_messages(case.task, operation)
    if prefix_chars is not None:
        (message,) = messages
        messages = [{**message, "content": message["content"][:prefix_chars]}]
    return len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **protocol.sampling.chat_template_kwargs,
        )
    )


def _context_decision(
    case: EvaluationCase,
    protocol: EvaluationProtocol,
    dataset_id: str,
    context_window: int,
    max_tokens: int,
    tokenizer: EvaluationTokenizer | None,
    rank: int,
) -> SelectionDecision:
    record = Selection(
        dataset=dataset_id,
        source_row=case.item.source_row,
        source_ordinal=case.item.source_ordinal,
        status=SelectionStatus.SELECTED,
        reason=None,
        rank=rank,
        context_window=context_window,
        max_tokens=max_tokens if protocol.sampling is not None else None,
        prompt_tokens=None,
        reference_tokens=None,
    )
    if protocol.sampling is None:
        return SelectionDecision(record, case)
    maximum = min(max_tokens, protocol.sampling.max_tokens or max_tokens)
    case = replace(case, task=prompt_variant(case, context_window, maximum, tokenizer))
    if tokenizer is None:
        if protocol.context_policy is not None:
            raise ValueError("Context-selected evaluations require the served model's pinned tokenizer")
        return SelectionDecision(record.model_copy(update={"max_tokens": maximum}), case)
    metadata = case.item.context_selection
    if isinstance(metadata, GraphWalksContextSelection):
        reference_tokens = len(tokenizer.encode(metadata.gold_output, add_special_tokens=False))
        maximum = graphwalks_output_budget(reference_tokens)
        record = record.model_copy(update={"reference_tokens": reference_tokens})
        if maximum > min(max_tokens, protocol.sampling.max_tokens or max_tokens):
            return SelectionDecision(
                record.model_copy(
                    update={
                        "status": SelectionStatus.EXCLUDED,
                        "reason": "output_cap",
                        "max_tokens": maximum,
                    }
                ),
                None,
            )
        allowance = context_window - maximum - GRAPHWALKS_CONTEXT_MARGIN
        if allowance <= 0:
            return SelectionDecision(
                record.model_copy(
                    update={
                        "status": SelectionStatus.EXCLUDED,
                        "reason": "context_window",
                        "max_tokens": maximum,
                    }
                ),
                None,
            )
        if metadata.prompt_chars > allowance * GRAPHWALKS_PREFIX_CHARS_PER_TOKEN:
            prefix = prompt_tokens(case, protocol, tokenizer, prefix_chars=allowance * GRAPHWALKS_PREFIX_CHARS_PER_TOKEN)
            if prefix > allowance + GRAPHWALKS_PREFIX_TOKEN_MARGIN:
                return SelectionDecision(
                    record.model_copy(
                        update={
                            "status": SelectionStatus.EXCLUDED,
                            "reason": "context_prefix",
                            "max_tokens": maximum,
                            "prompt_tokens": prefix,
                        }
                    ),
                    None,
                )
        size = prompt_tokens(case, protocol, tokenizer)
        if size > allowance:
            return SelectionDecision(
                record.model_copy(
                    update={
                        "status": SelectionStatus.EXCLUDED,
                        "reason": "context_window",
                        "max_tokens": maximum,
                        "prompt_tokens": size,
                    }
                ),
                None,
            )
    else:
        size = prompt_tokens(case, protocol, tokenizer)
        maximum = min(maximum, context_window - size - CONTEXT_MARGIN)
        if maximum < 1:
            return SelectionDecision(
                record.model_copy(
                    update={
                        "status": SelectionStatus.EXCLUDED,
                        "reason": "context_window",
                        "max_tokens": None,
                        "prompt_tokens": size,
                    }
                ),
                None,
            )
    return SelectionDecision(record.model_copy(update={"max_tokens": maximum, "prompt_tokens": size}), case)


def _mrcr_cases(
    root: str,
    protocol: EvaluationProtocol,
    dataset_id: str,
    context_window: int,
    max_tokens: int,
    limit: int | None,
    tokenizer: EvaluationTokenizer | None,
) -> Iterator[SelectionDecision]:
    items = list(evaluation_items(root))
    cells = defaultdict(list)
    for index, item in enumerate(items):
        metadata = item.context_selection
        assert isinstance(metadata, MRCRContextSelection)
        if metadata.bin_upper is not None:
            cells[(metadata.bin_upper, metadata.needles)].append(index)
    ordered = (
        interleave_mrcr_indices(cells, context_window) if any(upper <= context_window for upper, _ in cells) else []
    )
    ranks = {items[index].source_row: rank for rank, index in enumerate(ordered)}
    decisions: dict[str, Selection] = {}
    eligible: list[str] = []
    batch_size = limit or max(1, len(ordered))
    for offset in range(0, len(ordered), batch_size):
        batch = {items[index].source_row for index in ordered[offset : offset + batch_size]}
        for case in evaluation_cases(root, source_rows=batch):
            decision = _context_decision(
                case, protocol, dataset_id, context_window, max_tokens, tokenizer, ranks[case.item.source_row]
            )
            decisions[case.item.source_row] = decision.record
            if decision.case is not None:
                eligible.append(case.item.source_row)
        if limit is not None and len(eligible) >= limit:
            break
    # Keep only small decisions while preflighting; decode the selected TaskSpecs for execution.
    eligible.sort(key=ranks.__getitem__)
    selected_rows = frozenset(eligible if limit is None else eligible[:limit])
    cases = evaluation_cases(root, source_rows=selected_rows)
    for item in items:
        record = decisions.get(item.source_row)
        if item.source_row in selected_rows:
            case = next(cases)
            assert case.item.source_row == item.source_row and record is not None
            yield SelectionDecision(record, case)
        elif record is not None and record.status is SelectionStatus.EXCLUDED:
            yield SelectionDecision(record, None)
        else:
            rank = ranks.get(item.source_row)
            yield SelectionDecision(
                Selection(
                    dataset=dataset_id,
                    source_row=item.source_row,
                    source_ordinal=item.source_ordinal,
                    status=(
                        SelectionStatus.EXCLUDED if rank is None or record is not None else SelectionStatus.UNINSPECTED
                    ),
                    reason="official_bin" if rank is None else "source_limit",
                    rank=rank,
                    context_window=context_window,
                    max_tokens=record.max_tokens if record is not None else None,
                    prompt_tokens=record.prompt_tokens if record is not None else None,
                    reference_tokens=None,
                ),
                None,
            )


def select_cases(
    root: str,
    protocol: EvaluationProtocol,
    *,
    dataset_id: str,
    context_window: int,
    max_tokens: int,
    limit: int | None,
    tokenizer: EvaluationTokenizer | None,
) -> Iterator[SelectionDecision]:
    """Yield source-order decisions, retaining MRCR's interleaved cohort rank separately."""
    if protocol.context_window is not None:
        context_window = min(context_window, protocol.context_window)
    policy = protocol.context_policy
    if policy is not None and policy.max_context is not None:
        context_window = min(context_window, policy.max_context)
    selected_rows: set[str] | None = None
    if policy is not None and policy.kind == "mrcr":
        yield from _mrcr_cases(root, protocol, dataset_id, context_window, max_tokens, limit, tokenizer)
        return
    elif policy is None:
        selected_rows = set()
        for index, item in enumerate(evaluation_items(root)):
            if limit is not None and index >= limit:
                break
            selected_rows.add(item.source_row)
    cases = evaluation_cases(root, source_rows=selected_rows)
    selected = 0
    for index, item in enumerate(evaluation_items(root)):
        rank = selected if policy is not None else index
        if limit is not None and rank >= limit:
            yield SelectionDecision(
                Selection(
                    dataset=dataset_id,
                    source_row=item.source_row,
                    source_ordinal=item.source_ordinal,
                    status=SelectionStatus.UNINSPECTED,
                    reason="source_limit",
                    rank=None,
                    context_window=context_window,
                    max_tokens=None,
                    prompt_tokens=None,
                    reference_tokens=None,
                ),
                None,
            )
            continue
        case = next(cases)
        if case.item.source_row != item.source_row:
            raise ValueError("Evaluation metadata and canonical tasks have different source positions")
        decision = _context_decision(case, protocol, dataset_id, context_window, max_tokens, tokenizer, rank or 0)
        if decision.case is not None:
            selected += 1
        yield decision
