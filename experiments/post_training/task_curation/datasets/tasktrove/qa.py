# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove question-answering sources: open-ended answers judged against references, and MCQA.

The open-ended sources grade with the Nemotron harness's exact gate followed by a semantic judge;
verifyit's ``reference`` judge rubric is a port of that harness. Judged tasks wait for a judge
endpoint, so those sources have no offline controls. The multiple-choice source grades the option
letter in process.
"""

import re

from taskcompendium.convert.answers import mcq_task, source_defect, unsupported
from taskcompendium.convert.conversation import conversation_task
from taskcompendium.convert.delivery import replace_phrases, rewritten_task
from taskcompendium.convert.tasktrove import archive_resources
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.controls import reference_reply, wrong_reply
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, NormalizedTask, RawRow
from verifyit.modes.grade_judge import normalize as normalize_reference
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation.datasets.tasktrove import tasktrove_source
from experiments.post_training.task_curation.images import REWARDKIT_IMAGE
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim

OPENQA_REWRITE_REASON = "Replace source response-file delivery with the assistant response convention"
MCQA_REWRITE_REASON = "Replace the source's answer-file wrapper with a request for one option letter"
OPENQA_DELIVERY = (
    (
        "Write your concise final answer to `/app/response.txt`.",
        "Return your concise final answer in the assistant response.",
    ),
    (
        "Work through it and write your full answer to the file `/app/response.txt` inside the sandbox.",
        "Work through it and return your full answer in the assistant response.",
    ),
    (
        "To write the response from a shell, use a heredoc, e.g.:\n"
        "    cat > /app/response.txt <<'EOF'\n"
        "    <your full answer, ending with \\boxed{<final answer>}>\n"
        "    EOF\n"
        "Verify with `cat /app/response.txt` before completing. An empty or missing file scores 0.",
        "An empty response scores 0.",
    ),
)

MCQA_SEPARATOR = "\n---\n\n"
MCQA_FORMAT_PREFIX = (
    "Answer the following multiple choice question. The last line of your response should be in the following format: "
)
MCQA_FORMAT_WRAPPERS = ("{}", "\\boxed{{{}}}")
OPTION_LINE = re.compile(r"^[ \t]*([A-Z])[.):][ \t]", re.MULTILINE)
LISTED_OPTIONS = re.compile(r"'Answer: (?:\\boxed\{)?([A-Z](?:/[A-Z])+)")
MCQA_REGEX = r"Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])\s*"
MCQA_BOXED_REGEX = r"\\boxed\{\s*([A-Za-z0-9])\s*\}"
MCQA_EXTRACTIONS = frozenset(
    pattern for regex in (MCQA_REGEX, MCQA_BOXED_REGEX) for pattern in (regex, regex.replace("\\", "\\\\"))
)
"""Source answer-extraction regexes the letter grader reproduces, as written or with doubled escapes."""

OPENQA_RUBRIC = """
Require a complete, understandable question with all referenced passages, diagrams, choices, and prior turns supplied.
Do not invent omitted source context or infer an intended question from a topic fragment.

Check each reference against the question, including assumptions, scope, units, dates, and requested level of
explanation. Flag a reference that is unsupported, contradictory, or answers a different question.

Distinguish alternate valid factual answers from ambiguity that prevents a reasonable response. A semantic reference
judge may accept paraphrases, but it cannot repair a wrong or incomplete reference.

Science questions may need detailed reasoning or domain expertise. Difficulty alone is not a defect. Flag missing
experimental conditions, misleading scientific premises, or references that omit key findings.

Quoted answers, units, notation, and multilingual content can be coherent. Check whether the required boxed format can
express the substantive answer without changing its meaning.

Compare the wrapper's boxed-answer requirement with the question's own final-answer delimiters. Record conflicting
formatting requirements rather than silently choosing one or rewriting the question.

The grader applies the source's normalized exact gate and then a semantic judge. Model quality review does not itself
implement or certify that grading fallback.
"""

KNOWLEDGE_MCQA_RUBRIC = """
Require a complete question and all labeled options. Check whether exactly one option is defensible from the stated
context, and whether the reference selects it. Overlapping answers or unstated assumptions behind strongest/best claims
are concrete defects.

Specialized medical or scientific knowledge is allowed. Unsupported specificity, contradictory premises, and fabricated
distinctions between near-identical options are defects; unfamiliarity alone is not.
"""


def openqa_references(data: dict) -> tuple[str, ...]:
    """The accepted answers, from ``expected_answers`` or a single ``reference_answer``, without emphasis marks."""
    expected = data.get("expected_answers")
    if isinstance(expected, list):
        answers = tuple(answer for answer in expected if isinstance(answer, str))
    else:
        answer = data.get("reference_answer")
        answers = (answer,) if isinstance(answer, str) else ()
    return tuple(stripped for answer in answers if (stripped := answer.strip().strip("*").strip()))


def convert_openqa(row: RawRow) -> TaskSpec | NormalizedTask | ImportRejection:
    instruction, data = row.data["instruction"], row.data.get("verifier_data")
    if not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and verifier_data are required")
    references, question = openqa_references(data), data.get("instruction")
    if not isinstance(question, str) or not question.strip():
        return source_defect("missing_question", "The semantic judge requires its source question")
    if not references:
        return source_defect("invalid_references", "At least one reference is required")
    if any(not normalize_reference(reference) for reference in references):
        return source_defect("invalid_references", "A reference becomes empty under normalization")
    package = verifyit_package(
        JudgeSpec(references=references, question=question),
        archive_resources(row.data).verifier,
        environment=REWARDKIT_IMAGE.requirements(),
    )
    prompt = TextMessage(role="user", content=replace_phrases(instruction, OPENQA_DELIVERY))
    task = conversation_task(row, events=(prompt,), package=package)
    return rewritten_task(task, original=instruction, reason=OPENQA_REWRITE_REASON)


def mcqa_question(instruction: str, options: int) -> str:
    """The question and its options without the file-delivery wrapper, asking for one option letter."""
    _, separator, problem = instruction.partition(MCQA_SEPARATOR)
    if not separator:
        raise ValueError("Unrecognized MCQA delivery wrapper")
    if not problem.strip():
        raise ValueError("Missing MCQA question/options")
    first, separator, rest = problem.partition("\n\n")
    if first.startswith(MCQA_FORMAT_PREFIX):
        letters = tuple(chr(65 + index) for index in range(options))
        option_lists = {"/".join(letters)}
        listed = LISTED_OPTIONS.search(first)
        # A generated wrapper can list the labels out of order or repeat one; accept any listing of exactly
        # the actual labels.
        if listed is not None and set(listed.group(1).split("/")) == set(letters):
            option_lists.add(listed.group(1))
        valid_formats = {
            f"{MCQA_FORMAT_PREFIX}'Answer: {wrapper.format(listed_options)}' "
            f"(e.g. 'Answer: {wrapper.format(example)}')."
            for wrapper in MCQA_FORMAT_WRAPPERS
            for listed_options in option_lists
            for example in letters
        }
        if first not in valid_formats or not separator or not rest.strip():
            raise ValueError("Unsupported MCQA format wrapper")
        problem = rest
    return f"{problem.strip()}\n\nReturn one option letter from A through {chr(64 + options)}."


def convert_knowledge_mcqa(row: RawRow) -> TaskSpec | NormalizedTask | ImportRejection:
    instruction, data = row.data["instruction"], row.data.get("verifier_data")
    if not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and verifier_data are required")
    if data.get("output_regex") not in MCQA_EXTRACTIONS:
        return unsupported("unsupported_answer_contract", "Unsupported source MCQA extraction regex")
    letters = {match.group(1) for match in OPTION_LINE.finditer(instruction)}
    options = max((ord(letter) - 64 for letter in letters), default=0)
    if not letters or letters != {chr(65 + index) for index in range(options)}:
        return unsupported("unsupported_answer_contract", "Options must be a contiguous labeled sequence beginning at A")
    try:
        question = mcqa_question(instruction, options)
    except ValueError as error:
        return unsupported("unsupported_answer_contract", str(error))
    task = mcq_task(row, prompt=question, answer=data.get("expected_answer"), options=options)
    if isinstance(task, ImportRejection):
        return task
    return rewritten_task(task, original=instruction, reason=MCQA_REWRITE_REASON)


def pipelines() -> list[RlDataPipeline]:
    openqa = (
        ("knowledge-openqa", "laion__nemotron-gym-knowledge-openqa-v4"),
        ("science-openqa", "laion__nemotron-gym-science-so-openq-v3"),
    )
    return [
        *(
            RlDataPipeline(
                name=name,
                source=tasktrove_source(config),
                convert=convert_openqa,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=OPENQA_RUBRIC,
                atlas_id=f"Task Trove:{config}",
            )
            for name, config in openqa
        ),
        RlDataPipeline(
            name="tasktrove-knowledge_mcqa",
            source=tasktrove_source("laion__nemotron-gym-knowledge-mcqa-v2"),
            convert=convert_knowledge_mcqa,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=KNOWLEDGE_MCQA_RUBRIC,
            controls=Controls(golden=reference_reply, negative=wrong_reply),
            atlas_id="Task Trove:laion__nemotron-gym-knowledge-mcqa-v2",
        ),
    ]
