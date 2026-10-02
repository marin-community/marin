# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot ingestors for typed math, calculations, MCQA, and abstention QA."""

import base64
import re

from pydantic import ValidationError
from verifyit.grade import InvalidTask

from taskcompendium.grading import numeric_answer
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceVisibility,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
    task_resource,
)
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.source_definitions import tasktrove_source
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
from taskcompendium.pipeline.verification import verify_task, verify_witness
from taskcompendium.verifiers.atlas_answers import AbstentionAnswersVerifier, MathAnswerVerifier
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer
from taskcompendium.verifiers.reference_answers import ReferenceAnswersVerifier

CONFIGS = {
    "math_openreasoning": "laion__nemotron-gym-math-openmathreasoning-v2",
    "advanced_calculations": "laion__nemotron-gym-math-advanced-calculations-v4",
    "knowledge_mcqa": "laion__nemotron-gym-knowledge-mcqa-v2",
    "web_search_mcqa": "laion__nemotron-gym-knowledge-web-search-mcqa-v2",
    "qa_abstention": "laion__nemotron-gym-qa-abstention-v4",
}
OPTION_LINE = re.compile(r"^[ \t]*([A-Z])[.):][ \t]", re.MULTILINE)
MCQ_REGEX = r"Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])\s*"
MCQ_BOXED_REGEX = r"\\boxed\{\s*([A-Za-z0-9])\s*\}"
MCQ_FORMAT_PREFIX = (
    "Answer the following multiple choice question. The last line of your response should be in the following format: "
)
ABSTENTION_SUBMISSION = "\n## Submitting your answer (IMPORTANT)\n"
MATH_SUBMISSION = "\n## Submitting the answer\n"


def _instruction(instruction: str, name: str, options: int | None = None) -> str:
    """Normalize the recognized file delivery wrapper, preserving problem scope."""
    if name in {"knowledge_mcqa", "web_search_mcqa"}:
        _, separator, problem = instruction.partition("\n---\n\n")
        if not separator:
            raise ValueError("Unrecognized MCQA delivery wrapper")
        if not problem.strip() or options is None:
            raise ValueError("Missing MCQA question/options")
        first, separator, rest = problem.partition("\n\n")
        if first.startswith(MCQ_FORMAT_PREFIX):
            letters = tuple(chr(65 + index) for index in range(options))
            option_list = "/".join(letters)
            listed = re.search(r"'Answer: (?:\\boxed\{)?([A-Z](?:/[A-Z])+)", first)
            option_lists = {option_list}
            # A generated wrapper can repeat a label already present in the actual choices.
            if listed is not None and set(listed.group(1).split("/")) == set(letters):
                option_lists.add(listed.group(1))
            valid_formats = {
                f"{MCQ_FORMAT_PREFIX}'Answer: {wrapper.format(listed_options)}' "
                f"(e.g. 'Answer: {wrapper.format(example)}')."
                for wrapper in ("{}", "\\boxed{{{}}}")
                for listed_options in option_lists
                for example in letters
            }
            if first not in valid_formats or not separator or not rest.strip():
                raise ValueError("Unsupported MCQA format wrapper")
            problem = rest
        return f"{problem.strip()}\n\nReturn one option letter from A through {chr(64 + options)}."
    if name == "qa_abstention":
        instruction = instruction.partition(ABSTENTION_SUBMISSION)[0]
    if name == "math_openreasoning":
        instruction = instruction.partition(MATH_SUBMISSION)[0]
    instruction = instruction.replace(
        "write your final answer at the path `/app/answer.txt`", "return your final answer"
    )
    instruction = instruction.replace(
        "Write your final answer to the path `/app/answer.txt`", "Return your final answer"
    )
    instruction = instruction.replace(
        "write the final numeric answer (a single number) to `/app/answer.txt`",
        "return the final numeric answer (a single number)",
    )
    instruction = instruction.replace(
        "write ONLY the value of the LAST one to `/app/answer.txt`", "return ONLY the value of the LAST one"
    )
    instruction = instruction.replace("the answer file", "the assistant response")
    instruction = instruction.replace("your answer file", "your assistant response")
    return instruction.strip()


def normalize(row: RawRow, name: str) -> TaskSpec | ImportRejection:
    instruction, data = row.data.get("instruction"), row.data.get("verifier_data")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return ImportRejection(reason="missing_input", detail="Instruction and verifier_data are required")
    try:
        options = None
        if name == "math_openreasoning":
            expected = data["expected_answer"]
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError("A nonempty typed math reference is required")
            verifier = MathAnswerVerifier(expected=expected, math_type=data["answer_type"])
            spec = VerifierSpec(kind=VerifierKind.MATH_ANSWER, parameters_json=verifier.model_dump_json())
        elif name == "advanced_calculations":
            spec = numeric_answer(
                float(data["expected_value"]), float(data["tolerance_abs"]), float(data["tolerance_rel"])
            )
        elif name in {"knowledge_mcqa", "web_search_mcqa"}:
            supported_patterns = {
                pattern for regex in (MCQ_REGEX, MCQ_BOXED_REGEX) for pattern in (regex, regex.replace("\\", "\\\\"))
            }
            if data["output_regex"] not in supported_patterns:
                raise ValueError("Unsupported source MCQA extraction regex")
            letters = {match.group(1) for match in OPTION_LINE.finditer(instruction)}
            options = max((ord(letter) - 64 for letter in letters), default=0)
            if letters != {chr(65 + index) for index in range(options)} or not letters:
                raise ValueError("Options must be a contiguous labeled sequence beginning at A")
            spec = multiple_choice_answer(data["expected_answer"], options)
        elif name == "qa_abstention":
            reference = ReferenceAnswersVerifier(
                references=(data["expected_answer"],), question=data["question"], source_judge_data=data
            )
            verifier = AbstentionAnswersVerifier(reference=reference, abstention_token=data["abstention_token"])
            spec = VerifierSpec(kind=VerifierKind.ABSTENTION_ANSWERS, parameters_json=verifier.model_dump_json())
        else:
            raise ValueError(f"Unknown source: {name}")
        public = _instruction(instruction, name, options)
    except (KeyError, ValueError, TypeError, ValidationError) as error:
        return ImportRejection(reason="unsupported_answer_contract", detail=str(error))
    files = row.data.get("files", {})
    resources = tuple(
        task_resource(
            "/source/" + path,
            base64.b64decode(encoded, validate=True),
            ResourceVisibility.VERIFIER if path.startswith("tests/") else ResourceVisibility.CONTROL,
        )
        for path, encoded in files.items()
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=public),)),
        environment_requirements=EnvironmentRequirements(),
        resources=resources,
        answer_type=AnswerType.TEXT,
        verifier=spec,
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    if task.verifier.kind == VerifierKind.MATH_ANSWER:
        verifier = MathAnswerVerifier.model_validate_json(task.verifier.parameters_json)
        try:
            checks = verify_witness(task, rf"\boxed{{{verifier.expected}}}", "__incorrect_math_answer__")
        except InvalidTask as error:
            checks = [CheckResult(check="cleanup_math_reference", status=CheckStatus.UNSUPPORTED, detail=str(error))]
        checks.append(
            CheckResult(
                check="original_math_comparator",
                status=CheckStatus.UNSUPPORTED,
                detail="Cleanup math-verify comparator is not certified equivalent to original SymPy scorer",
            )
        )
    elif task.verifier.kind == VerifierKind.ABSTENTION_ANSWERS:
        verifier = AbstentionAnswersVerifier.model_validate_json(task.verifier.parameters_json)
        checks = verify_witness(task, verifier.reference.references[0], r"\boxed{[IDK]}")
        checks.append(
            CheckResult(
                check="semantic_reference_judge",
                status=CheckStatus.UNSUPPORTED,
                detail="Nonmatching non-abstention responses require the unbound source judge",
            )
        )
    else:
        checks = verify_task(task)
    return VerificationReport(checks=checks)


def recipe(name: str, *, rubric: ReviewRubric) -> DatasetRecipe:
    """Bind one pinned source and its content rubric to the common pipeline."""

    def normalize_row(row: RawRow) -> TaskSpec | ImportRejection:
        return normalize(row, name)

    return DatasetRecipe(
        name=f"tasktrove-{name}",
        version=f"tasktrove-{name}-v1",
        source=HFSource("open-thoughts/TaskTrove", REVISION, CONFIGS[name], "train"),
        normalize=normalize_row,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(id=f"{name}-answer-controls", revision="1", parameters={}, run=verification_report),
    )


SOURCES = {
    "advanced_calculations": tasktrove_source(
        config=CONFIGS["advanced_calculations"],
        revision=REVISION,
        rubric=ReviewRubric(
            id="advanced_calculations-answerability",
            version="1",
            criteria=(
                "The wrapper grades only the final requested expression. Preserve that scope, but reject "
                "contradictory requests for multiple answers or methods requiring tools absent from the "
                "public task.",
                "Independently compute the requested final quantity when feasible. Check radians versus "
                "degrees, units, domain errors, precision, rounding, and the declared absolute/relative "
                "tolerance.",
            ),
        ),
    ),
    "knowledge_mcqa": tasktrove_source(
        config=CONFIGS["knowledge_mcqa"],
        revision=REVISION,
        rubric=ReviewRubric(
            id="knowledge_mcqa-answerability",
            version="1",
            criteria=(
                "Require a complete question and all labeled options. Check whether exactly one option is "
                "defensible from the stated context, and whether the reference selects it. Overlapping "
                "answers or unstated assumptions behind strongest/best claims are concrete defects.",
                "Specialized medical or scientific knowledge is allowed. Unsupported specificity, "
                "contradictory premises, and fabricated distinctions between near-identical options are "
                "defects; unfamiliarity alone is not.",
            ),
        ),
    ),
    "math_openreasoning": tasktrove_source(
        config=CONFIGS["math_openreasoning"],
        revision=REVISION,
        rubric=ReviewRubric(
            id="math_openreasoning-answerability",
            version="1",
            criteria=(
                "Check the full mathematical problem, givens, notation, units, diagrams, and requested "
                "result. Reject absent diagrams, contradictory assumptions, or a private key inconsistent "
                "with a demonstrated solution.",
                "Independently verify short calculations. For long proofs, assess whether the problem is "
                "well posed; difficulty and inability to solve immediately are not defects. Do not invent a"
                " reference conflict.",
                "The private typed math key is graded by the cleanup math-verify comparator. Original SymPy"
                " comparator parity has not been established. Scalar, equation, interval, set, and ordered "
                "sequence distinctions matter.",
            ),
        ),
    ),
    "qa_abstention": tasktrove_source(
        config=CONFIGS["qa_abstention"],
        revision=REVISION,
        rubric=ReviewRubric(
            id="qa_abstention-answerability",
            version="1",
            criteria=(
                "Judge answerability from the actual question. The wrapper's claim that every question is "
                "knowable does not supply omitted passages, diagrams, personal facts, or needed "
                "experimental conditions.",
                "An optional [IDK] response does not repair an unanswerable task. The source rejects "
                "abstention for ordinary answerable rows. Check reference accuracy and whether it fully "
                "answers the question.",
                "The semantic paraphrase judge is unbound. That is a grading integration annotation, not a "
                "reason to reject an otherwise coherent static task. Do not confuse output delivery with "
                "subject matter.",
            ),
        ),
    ),
    "web_search_mcqa": tasktrove_source(
        config=CONFIGS["web_search_mcqa"],
        revision=REVISION,
        rubric=ReviewRubric(
            id="web_search_mcqa-answerability",
            version="1",
            criteria=(
                "Require a complete question, labeled options, and one defensible answer. The dataset name "
                "does not provide a browser, search results, or citations. Reject questions that require "
                "missing live evidence.",
                "Check overlapping options and unsupported strongest/best claims. A question answerable "
                "from stable knowledge does not require a browsing tool merely because of its dataset name.",
            ),
        ),
    ),
}


def recipe_for_source(
    name: str,
) -> DatasetRecipe:
    source = SOURCES[name]
    return recipe(name, rubric=source.rubric)
