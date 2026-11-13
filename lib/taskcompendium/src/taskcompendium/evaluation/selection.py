# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select reproducible cohorts from task annotations and declarative token budgets."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import zip_longest
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from taskcompendium.evaluation.corpus import EvaluationCase, EvaluationCorpus, TaskRow, source_order, text_prompt
from taskcompendium.evaluation.spec import (
    ChatGeneration,
    CohortOrder,
    CohortPolicy,
    ContextPolicy,
    EvaluationProtocol,
    LimitStage,
    OutputFit,
    TextGeneration,
)
from taskcompendium.models import TaskSpec
from taskcompendium.submission import conversation_messages

SELECTIONS_TABLE = "evaluation_selection"


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


def wire_messages(messages: Sequence[Mapping[str, Any]], operation: ChatGeneration) -> list[dict[str, Any]]:
    """Render source transport fields that do not change the canonical conversation."""
    if operation.historical_assistant_reasoning == "empty":
        return [
            {**message, "reasoning_content": ""} if message["role"] == "assistant" else dict(message)
            for message in messages
        ]
    return [dict(message) for message in messages]


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
        # Zero shots is the source's final fallback; rendered-chat preflight still
        # reduces its output budget or excludes it when no output token can fit.
        if variant.fewshot == 0 or len(tokenizer.encode(text, add_special_tokens=False)) < context_window - max_tokens:
            return case.task.model_copy(update={"context": variant.context})
    raise ValueError(f"No declared prompt variant fits the model context: {case.task.id}")


def prompt_tokens(
    case: EvaluationCase,
    protocol: EvaluationProtocol,
    tokenizer: EvaluationTokenizer,
    *,
    prefix_chars: int | None = None,
    prefix_message: int = 0,
) -> int:
    operation = case.item.operation
    if isinstance(operation, TextGeneration):
        prompt = text_prompt(case.task)
        return len(tokenizer.encode(prompt if prefix_chars is None else prompt[:prefix_chars], add_special_tokens=False))
    assert isinstance(operation, ChatGeneration)
    assert protocol.sampling is not None
    messages = wire_messages(conversation_messages(case.task.context.events), operation)
    if prefix_chars is not None:
        message = messages[prefix_message]
        messages[prefix_message] = {**message, "content": message["content"][:prefix_chars]}
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
    cap = min(max_tokens, protocol.sampling.max_tokens or max_tokens)
    case = replace(case, task=prompt_variant(case, context_window, cap, tokenizer))
    if tokenizer is None:
        if protocol.context_policy is not None:
            raise ValueError("Context-selected evaluations require the served model's pinned tokenizer")
        return SelectionDecision(record.model_copy(update={"max_tokens": cap}), case)
    policy = protocol.context_policy or ContextPolicy()
    maximum = cap
    if policy.output_budget is not None:
        assert case.item.reference_text is not None
        reference_tokens = len(tokenizer.encode(case.item.reference_text, add_special_tokens=False))
        budget = policy.output_budget
        maximum = budget.multiplier * max(budget.minimum, reference_tokens + budget.reserve)
        record = record.model_copy(update={"reference_tokens": reference_tokens})
        if maximum > cap and policy.output_fit is OutputFit.REQUIRE:
            return SelectionDecision(
                record.model_copy(
                    update={"status": SelectionStatus.EXCLUDED, "reason": "output_cap", "max_tokens": maximum}
                ),
                None,
            )
        maximum = min(maximum, cap)
    allowance = context_window - maximum - policy.context_reserve
    if policy.output_fit is OutputFit.REQUIRE and allowance <= 0:
        return SelectionDecision(
            record.model_copy(
                update={"status": SelectionStatus.EXCLUDED, "reason": "context_window", "max_tokens": maximum}
            ),
            None,
        )
    probe = policy.prefix_probe
    if probe is not None:
        assert case.item.prompt_characters is not None
        if case.item.prompt_characters > allowance * probe.characters_per_token:
            prefix = prompt_tokens(
                case,
                protocol,
                tokenizer,
                prefix_chars=allowance * probe.characters_per_token,
                prefix_message=probe.message_index,
            )
            if prefix > allowance + probe.token_margin:
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
    if policy.output_fit is OutputFit.REQUIRE:
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
        maximum = min(maximum, context_window - size - policy.context_reserve)
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


def _candidate_rows(rows: Sequence[TaskRow], policy: CohortPolicy, context_window: int) -> list[TaskRow]:
    if not policy.groups:
        return list(rows)
    groups = [
        group for group in policy.groups if group.minimum_context is None or group.minimum_context <= context_window
    ]
    cells: list[list[TaskRow]] = [[] for _ in groups]
    for row in rows:
        matches = [
            index
            for index, group in enumerate(groups)
            if all(tag in row.tags for tag in group.tags)
            and all(row.labels.get(key) == value for key, value in group.labels.items())
        ]
        if len(matches) > 1:
            raise ValueError(f"Task matches overlapping cohort groups: {row.source_row}")
        if matches:
            cells[matches[0]].append(row)
    missing = [group for group, cell in zip(groups, cells, strict=True) if not cell]
    if missing:
        raise ValueError(f"Evaluation corpus has no examples for requested groups: {missing}")
    if policy.order is CohortOrder.SOURCE:
        return sorted((row for cell in cells for row in cell), key=lambda row: source_order(row.source_row))
    return [row for cohort_round in zip_longest(*cells) for row in cohort_round if row is not None]


def select_cases(
    root: str,
    protocol: EvaluationProtocol,
    *,
    dataset_id: str,
    context_window: int,
    max_tokens: int,
    limit: int | None,
    tokenizer: EvaluationTokenizer | None,
    chat_template_kwargs: Mapping[str, bool | str] | None = None,
) -> Iterator[SelectionDecision]:
    """Yield a full source-order ledger with task-group ordering and the declared limit stage."""
    corpus = EvaluationCorpus(root)
    if protocol.fingerprint != corpus.protocol.fingerprint:
        raise ValueError("Selection protocol differs from its published corpus")
    rendering = protocol
    if protocol.sampling is not None and chat_template_kwargs is not None:
        rendering = protocol.model_copy(
            update={
                "sampling": protocol.sampling.model_copy(update={"chat_template_kwargs": dict(chat_template_kwargs)})
            }
        )
    if protocol.context_window is not None:
        context_window = min(context_window, protocol.context_window)
    rows = corpus.rows()
    policy = protocol.cohort
    ordered = _candidate_rows(rows, policy, context_window)
    ranks = {row.source_row: rank for rank, row in enumerate(ordered)}
    candidates = ordered[:limit] if policy.limit_stage is LimitStage.CANDIDATES and limit is not None else ordered
    decisions: dict[str, Selection] = {}
    variants: dict[str, int] = {}
    eligible: list[str] = []
    batch_size = limit or max(1, len(candidates))
    for offset in range(0, len(candidates), batch_size):
        batch = {row.source_row for row in candidates[offset : offset + batch_size]}
        for case in corpus.cases(source_rows=batch):
            rank = ranks[case.item.source_row]
            if policy.order is CohortOrder.SOURCE and policy.limit_stage is LimitStage.ELIGIBLE:
                rank = len(eligible)
            decision = _context_decision(case, rendering, dataset_id, context_window, max_tokens, tokenizer, rank)
            decisions[case.item.source_row] = decision.record
            if decision.case is not None:
                eligible.append(case.item.source_row)
                operation = case.item.operation
                if isinstance(operation, ChatGeneration) and operation.prompt_selection == "fit_context":
                    variants[case.item.source_row] = next(
                        index
                        for index, variant in enumerate(operation.variants)
                        if variant.context == decision.case.task.context
                    )
            if (
                policy.order is CohortOrder.SOURCE
                and policy.limit_stage is LimitStage.ELIGIBLE
                and limit is not None
                and len(eligible) >= limit
            ):
                break
        if limit is not None and len(eligible) >= limit:
            break
    eligible.sort(key=ranks.__getitem__)
    selected = frozenset(eligible if limit is None else eligible[:limit])
    cases = corpus.cases(source_rows=selected)
    for row in rows:
        record = decisions.get(row.source_row)
        if row.source_row in selected:
            case = next(cases)
            assert case.item.source_row == row.source_row and record is not None
            if row.source_row in variants:
                operation = case.item.operation
                assert isinstance(operation, ChatGeneration)
                case = replace(
                    case,
                    task=case.task.model_copy(update={"context": operation.variants[variants[row.source_row]].context}),
                )
            yield SelectionDecision(record, case)
        elif record is not None and record.status is SelectionStatus.EXCLUDED:
            yield SelectionDecision(record, None)
        else:
            rank = ranks.get(row.source_row)
            yield SelectionDecision(
                Selection(
                    dataset=dataset_id,
                    source_row=row.source_row,
                    source_ordinal=source_order(row.source_row)[1],
                    status=(
                        SelectionStatus.EXCLUDED if rank is None or record is not None else SelectionStatus.UNINSPECTED
                    ),
                    reason="group_filter" if rank is None else "source_limit",
                    rank=rank if policy.order is CohortOrder.ROUND_ROBIN else None,
                    context_window=context_window,
                    max_tokens=record.max_tokens if record is not None else None,
                    prompt_tokens=record.prompt_tokens if record is not None else None,
                    reference_tokens=record.reference_tokens if record is not None else None,
                ),
                None,
            )
