# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve reference-free source rubric judges for curation and later binding."""

import base64
import json
import re
import tomllib
from collections.abc import Callable

from taskcompendium.datasets.direct_contracts import source_contract_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    ImportFailureKind,
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    ReviewRubric,
    TaskPolicy,
    VerificationReport,
)

NUMBERED = re.compile(r"^\s*\d+[.)]\s*")
DELIVERY_SENTENCE = (
    "After you have completed your analysis and formulated your answer, you MUST write your final, "
    "comprehensive response into a file named 'response.txt' in the current directory."
)
DELIVERY_HEADER = "\n\n## Submitting your answer (IMPORTANT)\n"
HEREDOC_GUIDANCE = re.compile(
    r"  - To write the response from a shell, use a heredoc, e\.g\.:\n"
    r"        cat > /app/response\.txt <<'EOF'\n        [^\n]+\n        EOF\n"
)


def response_instruction(instruction: str) -> str:
    """Replace observed source file-delivery wrappers while preserving the underlying request."""
    instruction = instruction.replace(DELIVERY_SENTENCE, "Return your final, comprehensive assistant response.")
    body, marker, footer = instruction.rpartition(DELIVERY_HEADER)
    if marker and footer.startswith("You are a terminal agent."):
        instruction = body + "\n\nReturn your complete answer in the assistant response."
    prefix, separator, request = instruction.partition("\n---\n")
    if not separator:
        return instruction
    prefix = HEREDOC_GUIDANCE.sub("", prefix)
    prefix = prefix.replace(
        "  - Verify with `cat /app/response.txt` before marking the task complete. ",
        "  - ",
    ).replace("  - Verify with `cat /app/response.txt` before marking the task complete.\n", "")
    prefix = prefix.replace("to the file `/app/response.txt` inside the sandbox", "in the assistant response")
    prefix = prefix.replace("to `/app/response.txt`", "in the assistant response")
    prefix = prefix.replace("Empty or missing files score 0.", "An empty response scores 0.")
    return prefix + separator + request


def normalized_task(row: RawRow, contract: dict) -> NormalizedTask | ImportRejection:
    """Bind a private judge contract to the public response task."""
    instruction = row.data.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT, reason="missing_instruction", detail="Public instruction is required"
        )
    replacement = response_instruction(instruction)
    changes = (
        ()
        if replacement == instruction
        else (
            NormalizationChange(
                field="instruction",
                reason="Replace source response-file delivery with the assistant response convention",
                original=instruction,
                replacement=replacement,
            ),
        )
    )
    package = source_contract_package(
        "source rubric judge", row.source.revision, contract, ("Source semantic judge binding",)
    )
    return NormalizedTask(
        TaskSpec(
            id=row.id,
            source=row.source,
            context=ConversationInput(events=(TextMessage(role="user", content=replacement),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            answer_format=PlainText(),
            grader=package.grader,
            resources=ResourceGroups(verifier=package.resources),
        ),
        changes,
    )


def normalize(row: RawRow) -> NormalizedTask | ImportRejection:
    """Keep the original criteria, judge question, and source aggregation configuration."""
    data = row.data.get("verifier_data")
    if not isinstance(data, dict):
        files = row.data.get("files", {})
        encoded = files.get("tests/verifier_data.json")
        if encoded is None:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="missing_judge_data",
                detail="Source verifier_data.json is required",
            )
        data = json.loads(base64.b64decode(encoded, validate=True))
    rubric = data.get("rubric")
    if isinstance(rubric, list):
        criteria = tuple(entry.get("criteria", "").strip() for entry in rubric if isinstance(entry, dict))
    else:
        principle = data.get("principle")
        criteria = (
            tuple(NUMBERED.sub("", line).strip() for line in principle.splitlines() if line.strip())
            if isinstance(principle, str)
            else ()
        )
    question = data.get("instruction")
    if not isinstance(question, str) or not question.strip() or not criteria or any(not item for item in criteria):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="invalid_rubric",
            detail="Source question and nonempty criteria are required",
        )
    encoded_toml = row.data.get("files", {}).get("tests/judge.toml")
    if encoded_toml is None:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="missing_judge_config",
            detail="Original source judge.toml is required",
        )
    source_toml = base64.b64decode(encoded_toml, validate=True).decode()
    config = tomllib.loads(source_toml)
    contract = {
        "mode": (
            "holistic_numeric"
            if any(item.get("type") == "numeric" for item in config.get("criterion", []))
            else "checklist"
        ),
        "question": question,
        "criteria": list(criteria),
        "aggregation": config,
        "source_judge_data": data,
        "source_judge_toml": source_toml,
    }
    try:
        return normalized_task(row, contract)
    except (TypeError, ValueError) as error:
        return ImportRejection(kind=ImportFailureKind.UNSUPPORTED, reason="invalid_rubric", detail=str(error))


def normalize_identity(row: RawRow) -> NormalizedTask | ImportRejection:
    """Remove the identity source's terminal completion flag after response adaptation."""
    result = normalize(row)
    if isinstance(result, ImportRejection):
        return result
    instruction = result.task.context.events[0].content
    assert instruction is not None
    prefix, separator, request = instruction.partition("\n---\n")
    replacement = prefix.replace("  - Set `task_complete: true` once `/app/response.txt` is written.\n", "")
    replacement += separator + request
    if replacement == instruction:
        return result
    return NormalizedTask(
        result.task.model_copy(
            update={"context": ConversationInput(events=(TextMessage(role="user", content=replacement),))}
        ),
        (
            *result.changes,
            NormalizationChange(
                field="instruction",
                reason="Remove the source's terminal-only completion flag",
                original=instruction,
                replacement=replacement,
            ),
        ),
    )


def verification_report(_task: TaskSpec) -> VerificationReport:
    return VerificationReport(
        checks=[
            CheckResult(
                check="source_semantic_rubric_judge",
                status=CheckStatus.UNSUPPORTED,
                detail="The original source rubric and aggregation are preserved; its semantic judge is not bound",
            )
        ]
    )


STACK_EXCHANGE_COMMON_CRITERIA = (
    "Require a complete public request and any code, prior turns, or external passages needed to answer it.",
    "Check each private criterion against the public request; flag invented constraints or incorrect premises.",
    "The source uses one holistic numeric judge over four criteria and has no gold answer; its runtime "
    "judge is unbound.",
)

RUBRICS: dict[str, ReviewRubric] = {
    "identity_following": ReviewRubric(
        id="identity-following-answerability",
        version="1",
        criteria=(
            "Treat the explicit public identity and language requirements as the task's assigned persona. "
            "Preserve its Marin/NVIDIA wording; a different preferred training identity is not a content defect.",
            "Check that the user request can be answered under the public persona and language constraints. "
            "A user asking about another model or requesting a conflicting identity may be the intended challenge.",
            "Compare the original private principle, judge prompt and numeric scoring configuration with the "
            "public requirements. Flag hidden constraints, contradictions and ambiguous pass conditions.",
            "No canonical answer is supplied. The source semantic judge remains unbound; assess static task "
            "quality without treating that runtime limitation as a task defect.",
        ),
    ),
    "codereview": ReviewRubric(
        id="codereview-answerability",
        version="1",
        criteria=(
            *STACK_EXCHANGE_COMMON_CRITERIA,
            "Require the code under review and its intended behavior; inspect flattened source for lost "
            "comparisons, "
            "markup, links, and abrupt truncation before judging it complete.",
        ),
    ),
    "glaive_code": ReviewRubric(
        id="glaive_code-answerability",
        version="1",
        criteria=(
            *STACK_EXCHANGE_COMMON_CRITERIA,
            "Check required code, table schemas, input, output, dependencies, and runtime assumptions; generic "
            "programming questions can be answerable, but references to absent specific inputs are defects.",
        ),
    ),
    "safety": ReviewRubric(
        id="safety-answerability",
        version="1",
        criteria=(
            "Compare each safety principle with the actual request; refusal on a benign request is a rubric mismatch.",
            "Reject missing inputs, contradictions, and rubric requirements absent from the public request.",
            "The source grades one holistic numeric reward; preserve its rubric, judge policy, and threshold.",
            "No reference answer is supplied; do not invent one.",
            "An unavailable semantic judge is a verification limitation, not evidence that the content is bad.",
        ),
    ),
    "stack_overflow": ReviewRubric(
        id="stack_overflow-answerability",
        version="1",
        criteria=(
            *STACK_EXCHANGE_COMMON_CRITERIA,
            "Check that error reports include relevant code, versions, input, and observed behavior; distinguish "
            "plausible advice from an answer justified by supplied context.",
        ),
    ),
    "superuser": ReviewRubric(
        id="superuser-answerability",
        version="1",
        criteria=(
            *STACK_EXCHANGE_COMMON_CRITERIA,
            "Check operating system, application, privileges, and device assumptions; missing environment "
            "details can make a checklist demand impossible or unsafe.",
        ),
    ),
    "tezos": ReviewRubric(
        id="tezos-answerability",
        version="1",
        criteria=(
            "Check that Tezos questions supply necessary code, transaction details, versions, and error context.",
            "Reject missing inputs, contradictions, and rubric requirements absent from the public request.",
            "The source grades one holistic numeric reward; preserve its rubric, judge policy, and threshold.",
            "No reference answer is supplied; do not invent one.",
            "An unavailable semantic judge is a verification limitation, not evidence that the content is bad.",
        ),
    ),
    "unix": ReviewRubric(
        id="unix-answerability",
        version="1",
        criteria=(
            *STACK_EXCHANGE_COMMON_CRITERIA,
            "Check shell, distribution, filesystem, quoting, permissions, and tool assumptions; different valid "
            "commands must not be excluded by an arbitrary checklist.",
        ),
    ),
    "wizard_orca": ReviewRubric(
        id="wizard_orca-answerability",
        version="2",
        criteria=(
            "Trace supplied code with each stated example, including actual printed strings, divisions, "
            "return values, and arithmetic. A purported correct example that disagrees with the code is a "
            "defect unless the public task explicitly asks to debug or correct that discrepancy.",
            "Check the complete instruction, facts, and requested reasoning against the original private rubric.",
            "Reject missing inputs, contradictions, and rubric requirements absent from the public request.",
            "The source grades one holistic numeric reward; preserve its rubric, judge policy, and threshold.",
            "No reference answer is supplied; do not invent one.",
            "An unavailable semantic judge is a verification limitation, not evidence that the content is bad.",
        ),
    ),
}


def policy(
    rubric: ReviewRubric,
    normalize_row: Callable[[RawRow], NormalizedTask | ImportRejection] = normalize,
) -> TaskPolicy:
    """Build the original semantic-judge review policy for one rubric source."""
    return TaskPolicy(
        normalize=normalize_row,
        rubric=rubric,
        check_suite=CheckSuite(
            id="source-rubric-with-unbound-judge", revision="1", parameters={}, run=verification_report
        ),
    )
