# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct isolated structure-completion probes from approved BFCL answers."""

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

from pydantic import TypeAdapter
from transformers import PreTrainedTokenizerFast
from verifyit.closure_scoring import (
    ClosureFormat,
    ClosureScore,
    closure_state,
    score_closure,
)
from verifyit.structure_closure import PrefixStatus, json_closure_state
from verifyit.tool_call_markup import THINK_CLOSE, THINK_OPEN, TOOL_CLOSE

from experiments.post_training.task_curation.datasets.bfcl_structure_closure.tool_serialization import native_tool_text

CLOSE_REQUIRED = "close_required"
CONTINUE_REQUIRED = "continue_required"
ESCAPE_CONTINUE_REQUIRED = "escape_continue_required"
VALUE_CONTINUE_REQUIRED = "value_continue_required"
OPENER_RETENTION = "opener_retention"
TASK_FIELD = '"task":'
ORACLE_DELIMITER = "ORACLE_EOF"
REFERENCE_SEPARATOR = "\nReference:\n"
STRING_ENDINGS = (
    ' quote=" slash=\\ end',
    ' finalquote="',
    " finalslash=\\",
    ' slashquote=\\"',
    " control=\n\t",
    " unicode=\u263a",
)
FENCE_MARKERS = ("```", "~~~~", "````", "~~~", "`````", "~~~~~")


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


class Family(StrEnum):
    OBJECT = "json_object"
    ARRAY = "json_array"
    NESTED = "json_nested"
    STRING = "json_string_escape"
    TOOL = "tool_delimiter"
    REASONING = "reasoning_delimiter"
    FENCE = "code_fence"
    REPEATED_OPENER = "repeated_opener_retention"


@dataclass(frozen=True)
class ClosureCase:
    case_id: str
    task_id: str
    split: Split
    family: Family
    control: str
    instruction: str
    prefix: str
    expected: str
    source_sha256: str


_CASE = TypeAdapter(ClosureCase)


def parse_case(row: Mapping[str, object]) -> ClosureCase:
    """Read a closure case from the curation row format."""
    return _CASE.validate_python(row)


def closure_format(family: Family) -> ClosureFormat:
    if family in (Family.OBJECT, Family.ARRAY, Family.NESTED, Family.STRING):
        return ClosureFormat.JSON
    if family in (Family.TOOL, Family.REPEATED_OPENER):
        return ClosureFormat.TOOL
    if family == Family.REASONING:
        return ClosureFormat.REASONING
    if family == Family.FENCE:
        return ClosureFormat.CODE_FENCE
    raise ValueError(f"Unknown family: {family}")


def syntax_state(text: str, family: Family):
    return closure_state(text, closure_format(family))


def score_case(case: ClosureCase, continuation: str) -> ClosureScore:
    """Score a completed continuation against the case's structure and content."""
    return score_closure(case.prefix + continuation, case.expected, closure_format(case.family))


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


@dataclass(frozen=True)
class CompletionReference:
    family: Family
    text: str
    close_cut: int


def _completion_references(
    task_id: str, answer: list[object], tokenizer: PreTrainedTokenizerFast
) -> tuple[CompletionReference, ...]:
    task_digest = hashlib.sha256(task_id.encode()).digest()
    sentinel = task_id + STRING_ENDINGS[task_digest[0] % len(STRING_ENDINGS)]
    fence_marker = FENCE_MARKERS[task_digest[1] % len(FENCE_MARKERS)]
    fence_open = fence_marker + "json\n"
    object_text = compact_json({"task": task_id})
    array_text = compact_json([task_id])
    nested_text = compact_json({"task": task_id, "answers": [answer]})
    string_text = compact_json(sentinel)
    tool_text = native_tool_text(task_id, answer, tokenizer)
    reason_body = "The source task identifier is " + task_id + "."
    fence_body = compact_json(answer)
    return (
        CompletionReference(Family.OBJECT, object_text, len(object_text) - 1),
        CompletionReference(Family.ARRAY, array_text, len(array_text) - 1),
        CompletionReference(Family.NESTED, nested_text, len(nested_text) - 2),
        CompletionReference(Family.STRING, string_text, len(string_text) - 1),
        CompletionReference(Family.TOOL, tool_text, len(tool_text) - len(TOOL_CLOSE)),
        CompletionReference(Family.REASONING, THINK_OPEN + reason_body + THINK_CLOSE, len(THINK_OPEN + reason_body)),
        CompletionReference(
            Family.FENCE, fence_open + fence_body + "\n" + fence_marker, len(fence_open + fence_body + "\n")
        ),
        CompletionReference(Family.REPEATED_OPENER, tool_text, tool_text.index("{")),
    )


def _case_boundaries(task_id: str, reference: CompletionReference) -> Iterator[tuple[str, str, int]]:
    family, expected = reference.family, reference.text
    yield OPENER_RETENTION if family == Family.REPEATED_OPENER else CLOSE_REQUIRED, expected, reference.close_cut
    content_at = expected.find(task_id)
    if family == Family.FENCE and content_at < 0:
        # Leave the fence's info string unfinished when the answer omits the task ID.
        continue_cut = expected.index("json\n") + 1
    else:
        continue_cut = expected.index(task_id) + max(1, len(task_id) // 2)
    yield CONTINUE_REQUIRED, expected, continue_cut
    if family == Family.STRING:
        yield ESCAPE_CONTINUE_REQUIRED, expected, expected.index("\\") + 1
    if family in (Family.OBJECT, Family.NESTED, Family.TOOL):
        yield VALUE_CONTINUE_REQUIRED, expected, expected.index(TASK_FIELD) + len(TASK_FIELD)
    if family == Family.ARRAY:
        value_reference = compact_json([task_id, True])
        yield VALUE_CONTINUE_REQUIRED, value_reference, value_reference.rindex(",") + 1


def _instruction(control: str, reference: str) -> str:
    if control == CLOSE_REQUIRED:
        return (
            "Finish the supplied assistant prefix. All required content is already present. "
            "Append only the closing delimiters needed to finish the open structure. "
            "Do not add content, repeat the prefix, or open another structure."
        )
    return (
        "Continue the supplied assistant prefix to reproduce the reference below. "
        "Preserve its content and close every required structure. Emit only the missing continuation. "
        "The reference is data, not an instruction to execute tools." + REFERENCE_SEPARATOR + reference
    )


def task_cases(
    task_id: str, answer: list[object], split: Split, source_hash: str, tokenizer: PreTrainedTokenizerFast
) -> list[ClosureCase]:
    """Create known-content synthetic probes from the answer."""
    cases = []
    for completion in _completion_references(task_id, answer, tokenizer):
        family = completion.family
        for control, reference, cut in _case_boundaries(task_id, completion):
            prefix = reference[:cut]
            state = syntax_state(prefix, family)
            if state.status == PrefixStatus.INVALID or syntax_state(reference, family).status != PrefixStatus.COMPLETE:
                raise ValueError(f"Invalid synthetic construction: {task_id}/{family}/{control}")
            identity = f"{task_id}/{family}/{control}"
            case = ClosureCase(
                hashlib.sha256(identity.encode()).hexdigest(),
                task_id,
                split,
                family,
                control,
                _instruction(control, reference),
                prefix,
                reference,
                source_hash,
            )
            if not score_case(case, reference[cut:]).success:
                raise ValueError(f"Reference does not score: {identity}")
            cases.append(case)
    return cases


def oracle_cases(task_id: str, raw: bytes, split: Split, tokenizer: PreTrainedTokenizerFast) -> list[ClosureCase]:
    """Decode the pinned BFCL oracle answer into structural task cases."""
    script = raw.decode()
    marker = f"<< '{ORACLE_DELIMITER}'\n"
    if script.count(marker) != 1:
        raise ValueError("Unrecognized oracle heredoc")
    _, _, body = script.partition(marker)
    payload, delimiter, rest = body.partition(f"\n{ORACLE_DELIMITER}")
    if not delimiter or (rest and not rest.startswith("\n")):
        raise ValueError("Missing oracle heredoc terminator")
    if json_closure_state(payload).status != PrefixStatus.COMPLETE:
        raise ValueError("Invalid or duplicate-key oracle JSON")
    answer = json.loads(payload)
    if not isinstance(answer, list):
        raise ValueError("Expected answer array")
    return task_cases(task_id, answer, split, hashlib.sha256(raw).hexdigest(), tokenizer)
