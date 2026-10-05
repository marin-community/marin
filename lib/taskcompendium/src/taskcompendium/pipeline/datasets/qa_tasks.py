# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot adapters for TaskTrove knowledge and science open-ended QA."""

from pathlib import Path

from verifyit.modes.grade_judge import normalize as normalize_reference

from taskcompendium.grader import grader_config, script_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPipeline,
    VerificationReport,
)
from taskcompendium.pipeline.verification import answer_checks

GRADER_SCRIPT = (Path(__file__).with_name("grader_scripts") / "references.py").read_bytes()

SOURCE_DELIVERY = "Write your concise final answer to `/app/response.txt`."
SCIENCE_DELIVERY = "Work through it and write your full answer to the file `/app/response.txt` inside the sandbox."
SCIENCE_SHELL_GUIDANCE = (
    "To write the response from a shell, use a heredoc, e.g.:\n"
    "    cat > /app/response.txt <<'EOF'\n"
    "    <your full answer, ending with \\boxed{<final answer>}>\n"
    "    EOF\n"
    "Verify with `cat /app/response.txt` before completing. An empty or missing file scores 0."
)
RUBRIC = ReviewRubric(
    id="open-qa-reference-grounding",
    version="1",
    criteria=(
        "Require a complete, understandable question with all referenced passages, diagrams, choices, and prior "
        "turns supplied. Do not invent omitted source context or infer an intended question from a topic fragment.",
        "Check each reference against the question, including assumptions, scope, units, dates, and requested "
        "level of explanation. Flag a reference that is unsupported, contradictory, or answers a different question.",
        "Distinguish alternate valid factual answers from ambiguity that prevents a reasonable response. "
        "A semantic reference judge may accept paraphrases, but it cannot repair a wrong or incomplete reference.",
        "Science questions may need detailed reasoning or domain expertise. Difficulty alone is not a defect. "
        "Flag missing experimental conditions, misleading scientific premises, or references that omit key findings.",
        "Quoted answers, units, notation, and multilingual content can be coherent. Check whether the required "
        "boxed format can express the substantive answer without changing its meaning.",
        "Compare the wrapper's boxed-answer requirement with the question's own final-answer delimiters. "
        "Record conflicting formatting requirements rather than silently choosing one or rewriting the question.",
        "The verifier uses the source's normalized exact gate followed by a semantic judge, which is currently "
        "unbound. Model quality review does not itself implement or certify that grading fallback.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    instruction, data = row.data.get("instruction"), row.data.get("verifier_data")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return ImportRejection(reason="missing_input", detail="Instruction and verifier_data are required")
    expected = data.get("expected_answers")
    if isinstance(expected, list):
        references = tuple(answer.strip().strip("*").strip() for answer in expected if isinstance(answer, str))
    else:
        answer = data.get("reference_answer")
        references = (answer.strip().strip("*").strip(),) if isinstance(answer, str) else ()
    references = tuple(answer for answer in references if answer)
    question = data.get("instruction")
    if not isinstance(question, str) or not question.strip():
        return ImportRejection(reason="missing_question", detail="The semantic judge requires its source question")
    if not references:
        return ImportRejection(reason="invalid_references", detail="At least one reference is required")
    if any(not normalize_reference(reference) for reference in references):
        return ImportRejection(reason="invalid_references", detail="A reference becomes empty under normalization")
    package = script_package(GRADER_SCRIPT, {"references": references, "question": question, "source_judge_data": data})
    instruction = instruction.replace(SOURCE_DELIVERY, "Return your concise final answer in the assistant response.")
    instruction = instruction.replace(
        SCIENCE_DELIVERY, "Work through it and return your full answer in the assistant response."
    )
    instruction = instruction.replace(SCIENCE_SHELL_GUIDANCE, "An empty response scores 0.")
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=package,
        source=row.source,
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    """Exercise the exact gate and preserve semantic fallback as unresolved."""
    config = grader_config(task)
    checks = answer_checks(task, (("empty", "", 0.0), ("reference", config["references"][0], 1.0)))
    checks.append(
        CheckResult(
            check="semantic_reference_judge",
            status=CheckStatus.UNSUPPORTED,
            detail="Nonmatching responses require the unbound source semantic judge; exact controls are insufficient",
        )
    )
    return VerificationReport(checks=checks)


def pipeline() -> TaskPipeline:
    """Build the source normalization and review policy."""
    return TaskPipeline(
        normalize=normalize,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="openqa-exact-gate-with-unbound-judge", revision="1", parameters={}, run=verification_report
        ),
    )
