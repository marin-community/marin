# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize TaskTrove IFEval without guessing missing requests."""

import re

from verifyit.grade import InvalidTask
from verifyit.modes.grade_ifeval import resolve_checks
from verifyit.spec import Constraint, IfevalSpec

from taskcompendium.grader import grader_package
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
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_task
from taskcompendium.runtime.task_grading import resolve_verifier

NON_LATIN_LANGUAGES = frozenset({"ar", "bg", "bn", "he", "hi", "ja", "ko", "ne", "ru", "ta", "te", "th", "zh"})
LATIN_WORD = re.compile(r"[A-Za-z]+")
POSITIONAL_WORDS = {
    "first_word:first_word_answer": "first_word",
    "first_word:first_word_sent": "first_word",
    "last_word:last_word_answer": "last_word",
    "last_word:last_word_sent": "last_word",
    "length_constraints:nth_paragraph_first_word": "first_word",
}
RUBRIC = ReviewRubric(
    id="instruction-following-answerability",
    version="3",
    criteria=(
        "Identify the underlying content request independently of the formatting constraints. A topic fragment, "
        "random text, or missing question is not an answerable task; do not invent what the user meant.",
        "Check that the content request and ALL constraints can be satisfied together. Reject contradictory counts "
        "and answer formats that cannot express the requested answer, even if the formal checker can pass.",
        "Reject requests depending on absent documents, profiles, prior turns, or unavailable tools. Asking for "
        "clarification is not a complete answer to a missing-input task.",
        "Distinguish an actual missing-input deliverable from a capability/setup question such as 'could you review "
        "an email for me?'; that question can be answered by requesting the email. A promised but absent article "
        "needed for an actual summary or analysis remains missing context.",
        "Apply language requirements literally. ENTIRELY or ONLY in one language conflicts with mandatory words "
        "from another language unless an explicit exception is supplied. Do not invent an exception. A question "
        "written in Chinese does not by itself require an exclusively Chinese answer.",
        "Structural markers (P.S., markdown delimiters), mathematical notation, and code syntax are not foreign "
        "lexical words. Treat unclear marker case or delimiter precedence as uncertainty; do not invent an "
        "exception to an explicit prohibition on capitals, punctuation, or trailing characters.",
        "Multilingual requests, spelling errors, and hard but feasible constraints alone are not defects. Require "
        "a meaningful, comprehensible answer; flag constraints that destroy that answer rather than merely making "
        "it difficult or unusual.",
        "Check the private constraint identifiers and parameters against the public wording. The checker measures "
        "formal compliance only; it does not certify factual correctness or semantic usefulness.",
        "For a constraint-only verifier, reference_status concerns agreement between public constraints and "
        "checker configuration. There is no canonical answer key; its absence alone is not missing context, "
        "a reason for reference_status=unknown, or a reason to downgrade a clearly answerable task.",
        "Distinguish a contradiction in the written instructions from impossibility under the checker. Some shared "
        "checks approximate language, word counts, or sentence structure; passing them does not repair a bad prompt.",
        "Shared checker details: copy:repeat_phrase requires at least N unchanged case-insensitive substring "
        "occurrences, so transformed variants do not satisfy it. letters:letter_counting counts ASCII word "
        "matches rather than letters. count:count_unique requires at least five unique ASCII words and 50% "
        "uniqueness, not all words unique. startend:end_checker uses literal endswith, including closing delimiters. "
        "Use these facts to identify concrete prompt/checker mismatches; under-enforcement alone does not make "
        "an otherwise meaningful content request unanswerable.",
        "Assess whether a short answer can still convey the requested content. Do not infer impossibility just "
        "because a story must be compressed or a constraint makes the task difficult. Use quality=good and high "
        "confidence when there is no material defect; reserve uncertainty for a specific unresolved issue.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    instruction = row.data.get("instruction")
    data = row.data.get("verifier_data")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_input",
            detail="Instruction and verifier_data are required",
        )
    # This source has a fixed shell-delivery preamble; the direct-chat adapter changes only delivery.
    if instruction.startswith("You are running in a shell-based sandbox."):
        _, separator, instruction = instruction.partition("\n---\n")
        if not separator:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="unknown_wrapper",
                detail="The shell preamble has no task separator",
            )
    names, parameters = data.get("instruction_id_list"), data.get("kwargs")
    if not isinstance(names, list) or not isinstance(parameters, list) or len(names) != len(parameters):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="invalid_constraints",
            detail="Constraint names and parameters must align",
        )
    if any(
        not isinstance(name, str) or not isinstance(params, dict) for name, params in zip(names, parameters, strict=True)
    ):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="invalid_constraints",
            detail="Constraint names and parameters must be objects",
        )
    try:
        constraints = tuple(Constraint(name=name, params=params) for name, params in zip(names, parameters, strict=True))
        resolve_checks(constraints)
        package = grader_package(IfevalSpec(constraints=constraints))
    except (InvalidTask, ValueError) as error:
        return ImportRejection(kind=ImportFailureKind.UNSUPPORTED, reason="invalid_constraints", detail=str(error))
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction.strip()),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=package.verifier,
        source=row.source,
    )


def policy() -> TaskPolicy:
    """Build the source normalization and review policy."""
    return TaskPolicy(
        normalize=normalize,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="ifeval-contract-and-controls",
            revision="1",
            parameters={},
            run=verification_report,
        ),
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    """Catch the source's explicit language/positional-word contradiction.

    This bounded rule covers non-Latin response languages with mandatory Latin
    positional words. It does not infer a language from the question or attempt
    to classify foreign words across languages sharing a script.
    """
    event = task.context.events[0]
    assert isinstance(event, TextMessage)
    verifier = resolve_verifier(task.verifier)
    assert isinstance(verifier, IfevalSpec)
    languages = set()
    for constraint in verifier.constraints:
        language = constraint.params.get("language")
        if constraint.name == "language:response_language" and isinstance(language, str):
            languages.add(language)
    checks = []
    if "Your ENTIRE response should be in " in event.content and "no other language is allowed" in event.content:
        if languages & NON_LATIN_LANGUAGES:
            for constraint in verifier.constraints:
                parameter = POSITIONAL_WORDS.get(constraint.name)
                word = constraint.params.get(parameter) if parameter is not None else None
                if isinstance(word, str) and LATIN_WORD.fullmatch(word):
                    checks.append(
                        CheckResult(
                            check="exclusive_language_positional_word",
                            status=CheckStatus.FAIL,
                            detail=f"Exclusive non-Latin response language conflicts with mandatory Latin word {word!r}",
                        )
                    )
    return VerificationReport(checks=[*checks, *verify_task(task)])
