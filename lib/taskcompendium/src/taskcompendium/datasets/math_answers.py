# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed math normalization and review policies."""

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any, Literal

from pydantic import JsonValue
from rigging.filesystem.storage_path import StoragePath
from verifyit.modes.extract import extract_boxed
from verifyit.spec import MathSpec, MathType

from taskcompendium.grader import grader_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    CheckSuite,
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_witness
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.task_grading import resolve_verifier

NUMINA_PROOF_REQUEST = re.compile(r"\b(?:prove|show)\s+that\b", re.IGNORECASE)
NUMINA_INEQUALITY = re.compile(r"[<>≤≥]|\\(?:leqslant|geqslant|leq|geq|le|ge|lt|gt)\b")


def answer_type(expected: str) -> Literal["scalar", "equation", "interval", "set", "tuple", "list"]:
    """Keep structured mathematical answers distinct from scalar values."""
    if expected.startswith("[") and expected.endswith("]"):
        return "list"
    if expected.startswith(("[", "(")) and expected.endswith(("]", ")")) and "," in expected:
        return "tuple" if expected.startswith("(") and expected.endswith(")") else "interval"
    if expected.startswith(r"\{") or expected.startswith("{"):
        return "set"
    if "=" in expected or r"\approx" in expected:
        return "equation"
    return "scalar"


def normalize_math(row: RawRow, problem_field: str, reference_field: str) -> TaskSpec | ImportRejection:
    problem, reference = row.data.get(problem_field), row.data.get(reference_field)
    if not isinstance(problem, str) or not problem.strip():
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt",
            detail=f"{problem_field} must be a nonempty string",
        )
    if not isinstance(reference, str) or not reference.strip():
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="invalid_reference",
            detail=f"{reference_field} must be a nonempty string",
        )
    expected = extract_boxed(reference) or reference.strip()
    spec = MathSpec(expected=expected, math_type=MathType(answer_type(expected)))
    private = json.dumps(
        {key: row.data[key] for key in ("solution", "answer_type", "extracted_answer", "source") if key in row.data},
        ensure_ascii=False,
    ).encode()
    package = grader_package(spec, (inline_resource("reference/source-evidence.json", private),))
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=problem),)),
        environment_requirements=EnvironmentRequirements(),
        resources=ResourceGroups(verifier=package.resources),
        answer_type=AnswerType.TEXT,
        verifier=package.verifier,
    )


def math_controls(task: TaskSpec) -> VerificationReport:
    spec = resolve_verifier(task.verifier)
    assert isinstance(spec, MathSpec)
    return VerificationReport(checks=verify_witness(task, rf"\boxed{{{spec.expected}}}", "__incorrect_math_answer__"))


def math_task(
    row: RawRow, events: tuple[TextMessage, ...], expected: str, evidence: dict[str, JsonValue]
) -> TaskSpec | ImportRejection:
    """Bind an extracted reference and private evidence to public source messages."""
    problem = "\n\n".join(event.content for event in events)
    task = normalize_math(replace(row, data={"problem": problem, "answer": expected}), "problem", "answer")
    if isinstance(task, ImportRejection):
        return task
    resource = inline_resource("reference/source-evidence.json", json.dumps(evidence, ensure_ascii=False).encode())
    package = grader_package(resolve_verifier(task.verifier), (resource,))
    return task.model_copy(
        update={
            "context": ConversationInput(events=events),
            "resources": ResourceGroups(verifier=package.resources),
        }
    )


def field_math_task(
    row: RawRow, problem_key: str, answer_key: str, evidence_keys: tuple[str, ...]
) -> TaskSpec | ImportRejection:
    """Extract a direct problem/reference pair while retaining source evidence."""
    problem, expected = row.data.get(problem_key), row.data.get(answer_key)
    if not isinstance(problem, str) or not isinstance(expected, str):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_reference",
            detail=f"{problem_key} and {answer_key} strings are required",
        )
    evidence = {key: row.data[key] for key in evidence_keys}
    return math_task(row, (TextMessage(role="user", content=problem),), expected, evidence)


MATH500_RUBRIC = ReviewRubric(
    id="math500-quality",
    version="1",
    criteria=(
        "Preserve tuple order, intervals, units, and mathematical domains. This official test subset is "
        "evaluation data.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_math500(row: RawRow) -> TaskSpec | ImportRejection:
    return field_math_task(row, "problem", "answer", ("answer", "solution", "subject", "level", "unique_id"))


AIME_1983_2024_RUBRIC = ReviewRubric(
    id="aime_1983_2024-quality",
    version="1",
    criteria=(
        "Historical AIME answers are integers; preserve contest year and problem number privately and reserve "
        "this benchmark for evaluation.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_aime_1983_2024(row: RawRow) -> TaskSpec | ImportRejection:
    return field_math_task(row, "Question", "Answer", ("Answer", "ID", "Year", "Problem Number", "Part"))


GSM8K_RUBRIC = ReviewRubric(
    id="gsm8k-quality",
    version="1",
    criteria=(
        "The final #### answer is the numeric key; retain the worked derivation privately and check its "
        "arithmetic against the question.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_gsm8k(row: RawRow) -> TaskSpec | ImportRejection:
    problem, answer = row.data.get("question"), row.data.get("answer")
    if not isinstance(problem, str) or not isinstance(answer, str) or "####" not in answer:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_reference",
            detail="question and answer with #### final separator are required",
        )
    expected = answer.rsplit("####", 1)[-1].strip()
    return math_task(row, (TextMessage(role="user", content=problem),), expected, {"answer": answer})


ASDIV_RUBRIC = ReviewRubric(
    id="asdiv-quality",
    version="1",
    criteria=(
        "The Body and Question jointly specify the problem. Answer parentheses contain units, which must "
        "agree with the public quantity.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_asdiv(row: RawRow) -> TaskSpec | ImportRejection:
    body, question, answer = (row.data.get(key) for key in ("Body", "Question", "Answer"))
    if not all(isinstance(value, str) and value.strip() for value in (body, question, answer)):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_reference",
            detail="Body, Question and Answer are required",
        )
    expected = re.sub(r"\s*\([^)]*\)\s*$", "", str(answer)).strip()
    evidence = {key: row.data[key] for key in ("Answer", "Formula", "Solution-Type", "Source", "Grade", "ID")}
    return math_task(row, (TextMessage(role="user", content=f"{body}\n\n{question}"),), expected, evidence)


def asdiv_rows(path: StoragePath) -> Iterator[dict[str, Any]]:
    """Read ASDiv Problem elements into the fields used by its converter."""
    with path.open("rb") as stream:
        root = ET.parse(stream).getroot()
    yield from ({**item.attrib, **{child.tag: child.text or "" for child in item}} for item in root.iter("Problem"))


DAPO_MATH_RUBRIC = ReviewRubric(
    id="dapo_math-quality",
    version="1",
    criteria=(
        "Preserve every prompt message and reward_model ground truth; source scorer parity is not implied by "
        "matching a reference.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_dapo_math(row: RawRow) -> TaskSpec | ImportRejection:
    messages, reward = row.data.get("prompt"), row.data.get("reward_model")
    if not isinstance(messages, list) or not messages or not isinstance(reward, dict) or "ground_truth" not in reward:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_reference",
            detail="prompt messages and reward_model ground_truth are required",
        )
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    evidence = {key: row.data[key] for key in ("reward_model", "data_source", "ability", "extra_info")}
    return math_task(row, events, str(reward["ground_truth"]), evidence)


RLVR_MATH_RUBRIC = ReviewRubric(
    id="rlvr_math-quality",
    version="1",
    criteria=(
        "Retain public few-shot worked examples and distinguish them from the final question. This selected split is "
        "training data.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_rlvr_math(row: RawRow) -> TaskSpec | ImportRejection:
    messages, expected = row.data.get("messages"), row.data.get("ground_truth")
    if not isinstance(messages, list) or not messages or not isinstance(expected, str):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_reference",
            detail="messages and ground_truth strings are required",
        )
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    evidence = {key: row.data[key] for key in ("ground_truth", "dataset", "constraint_type", "constraint")}
    return math_task(row, events, expected, evidence)


NUMINA_MATH_RUBRIC = ReviewRubric(
    id="numina_math-quality",
    version="1",
    criteria=(
        "The boxed solution conclusion is a generated reference; assess its derivation rather than assuming "
        "its correctness.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_numina_math(row: RawRow) -> TaskSpec | ImportRejection:
    problem, solution = row.data.get("problem"), row.data.get("solution")
    if not isinstance(problem, str) or not isinstance(solution, str):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_reference",
            detail="problem and solution strings are required",
        )
    expected = extract_boxed(solution)
    if not expected:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED, reason="missing_final_answer", detail="Solution lacks a boxed conclusion"
        )
    if NUMINA_PROOF_REQUEST.search(problem) and NUMINA_INEQUALITY.search(expected):
        # A boxed inequality is a proof conclusion, not a scalar-answer witness.
        # The cleanup comparator cannot verify the requested derivation.
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_proof_contract",
            detail="The requested inequality proof is outside the typed answer comparator's capabilities",
        )
    return math_task(
        row, (TextMessage(role="user", content=problem),), expected, {"solution": solution, "source": row.data["source"]}
    )


def normalize_hardmath(row: RawRow) -> TaskSpec | ImportRejection:
    return normalize_math(row, "question", "ground_truths")


def normalize_hendrycks_math(row: RawRow) -> TaskSpec | ImportRejection:
    solution = row.data.get("solution")
    if isinstance(solution, str) and r"\boxed" not in solution:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="missing_final_answer",
            detail="A boxed final answer is required in solution",
        )
    return normalize_math(row, "problem", "solution")


def normalize_deepscaler(row: RawRow) -> TaskSpec | ImportRejection:
    return normalize_math(row, "problem", "answer")


def math_policy(
    normalize: Callable[[RawRow], TaskSpec | ImportRejection],
    rubric: ReviewRubric,
    check_id: str,
) -> TaskPolicy:
    """Apply the typed math comparator and its witness controls."""
    return TaskPolicy(
        normalize=normalize,
        rubric=rubric,
        check_suite=CheckSuite(
            id=check_id,
            revision="1",
            parameters={"comparator": "cleanup-math-verify"},
            run=math_controls,
        ),
    )


HARDMATH_RUBRIC = ReviewRubric(
    id="hardmath-asymptotics",
    version="1",
    criteria=(
        "Check the question, private solution, and ground_truths together. Symbolic "
        "regimes, approximations, boundary conditions, and equations are part of the "
        "task.",
        "Compare every requested regime or deliverable with the reference list. A key "
        "omitting a requested asymptotic regime is a concrete mismatch.",
        "Check limiting powers and coefficients before certifying asymptotic formulas; "
        "do not confuse necessary and sufficient regimes or invent precision "
        "requirements.",
        "The training source is not an evaluation benchmark binding. The cleanup typed "
        "math comparator is used without claiming source scorer parity; hard mathematics "
        "alone is not a defect.",
    ),
)

HENDRYCKS_MATH_RUBRIC = ReviewRubric(
    id="hendrycks-math-algebra-train",
    version="1",
    criteria=(
        "Preserve the complete algebra problem, domains, quantifiers, units, and requested answer form.",
        "The last boxed solution answer is private reference evidence. Check short "
        "calculations and contradictions; ordered pairs and half-open intervals are "
        "different contracts.",
        "Only the algebra train split is bound. MATH test and MATH-500 remain evaluation "
        "sources and must not be merged through this binding.",
        "Assess well-posedness independently of difficulty. The cleanup typed math "
        "comparator is used without claiming original scorer parity.",
    ),
)

DEEPSCALER_RUBRIC = ReviewRubric(
    id="deepscaler-math",
    version="1",
    criteria=(
        "Check the problem and private answer for a unique mathematical result; preserve " "domains, LaTeX, and units.",
        "The answer field is the key. A solution ending with an option letter does not "
        "override a numeric answer; compare the actual arithmetic before alleging a "
        "contradiction.",
        "This training blend includes historical AIME/AMC, Omni-MATH, and Still "
        "problems. Retain provenance and do not treat it as uncontaminated evaluation "
        "data.",
        "The cleanup typed math comparator is used; source reward-scorer parity is not "
        "claimed. Difficulty and inability to solve immediately are not defects.",
    ),
)
