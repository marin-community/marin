# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot adapters for TaskTrove knowledge and science open-ended QA."""

from pydantic import ValidationError

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.source_definitions import tasktrove_inputs
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    VerificationReport,
)
from taskcompendium.pipeline.verification import answer_checks
from taskcompendium.verifiers.reference_answers import ReferenceAnswersVerifier

KNOWLEDGE_CONFIG = "laion__nemotron-gym-knowledge-openqa-v4"
SCIENCE_CONFIG = "laion__nemotron-gym-science-so-openq-v3"
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
    try:
        verifier = ReferenceAnswersVerifier(references=references, question=question, source_judge_data=data)
    except (ValidationError, ValueError) as error:
        return ImportRejection(reason="invalid_references", detail=str(error))
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
        verifier=VerifierSpec(kind=VerifierKind.REFERENCE_ANSWERS, parameters_json=verifier.model_dump_json()),
        source=row.source,
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    """Exercise the exact gate and preserve semantic fallback as unresolved."""
    verifier = ReferenceAnswersVerifier.model_validate_json(task.verifier.parameters_json)
    checks = answer_checks(task, verifier, (("empty", "", 0.0), ("reference", verifier.references[0], 1.0)))
    checks.append(
        CheckResult(
            check="semantic_reference_judge",
            status=CheckStatus.UNSUPPORTED,
            detail="Nonmatching responses require the unbound source semantic judge; exact controls are insufficient",
        )
    )
    return VerificationReport(checks=checks)


def knowledge_recipe() -> DatasetRecipe:
    return _recipe("knowledge-openqa", KNOWLEDGE_CONFIG)


def science_recipe() -> DatasetRecipe:
    return _recipe("science-openqa", SCIENCE_CONFIG)


def _recipe(name: str, config: str) -> DatasetRecipe:
    return DatasetRecipe(
        name=name,
        version=f"{name}-v1",
        source=HFSource("open-thoughts/TaskTrove", REVISION, config, "train"),
        inputs=tasktrove_inputs(config, REVISION),
        normalize=normalize,
        rubric=RUBRIC,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id="openqa-exact-gate-with-unbound-judge", revision="1", parameters={}, run=verification_report
        ),
    )
