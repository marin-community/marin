# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove question-answering sources: open-ended answers judged against references, and MCQA.

The open-ended sources grade with the Nemotron harness's exact gate followed by a semantic judge;
verifyit's ``reference`` judge rubric is a port of that harness. Verification cannot reach a judge
endpoint, so those sources have no controls and their kept rows are admitted without them. The
multiple-choice source grades the option letter in process.
"""

import re

from taskcompendium.convert.answers import mcq_task, source_defect, unsupported
from taskcompendium.convert.conversation import conversation_task
from taskcompendium.convert.delivery import replace_phrases, rewritten_task
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, NormalizedTask, RawRow
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter, tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import archive_resources
from experiments.post_training.task_curation.pipeline import CurationRecipe, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

MCQA_CONFIG = "laion__nemotron-gym-knowledge-mcqa-v2"

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
LISTED_OPTIONS = re.compile(r"'Answer: (?:\\boxed\{)?([A-Z](?:/[A-Z])*)")
ESCAPED_OPTION_NEWLINE = re.compile(r"\\n(?=[ \t]*[A-Z][.):][ \t])")
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


def convert_openqa(row: RawRow, context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
    instruction, data = row.data["instruction"], row.data.get("verifier_data")
    if not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and verifier_data are required")
    references, question = openqa_references(data), data.get("instruction")
    if not isinstance(question, str) or not question.strip():
        return source_defect("missing_question", "The semantic judge requires its source question")
    if not references:
        return source_defect("invalid_references", "At least one reference is required")
    # JudgeSpec carries the grading inputs; archived scripts remain private provenance only.
    package = verifyit_package(
        JudgeSpec(references=references, question=question),
        tuple(
            resource.model_copy(update={"path": "source/" + resource.path})
            for resource in archive_resources(row.data).verifier
        ),
        environment=required_grader_environment(context),
    )
    prompt = TextMessage(role="user", content=replace_phrases(instruction, OPENQA_DELIVERY))
    task = conversation_task(row, events=(prompt,), package=package)
    subject = "knowledge" if isinstance(data.get("expected_answers"), list) else "science"
    task = task.model_copy(update={"tags": ("qa", "openqa", "judge", "reference", "nemotron", subject)})
    return rewritten_task(task, original=instruction, reason=OPENQA_REWRITE_REASON)


def mcqa_option_labels(problem: str) -> tuple[str, ...]:
    """Choice labels beginning with A, excluding preceding labeled premises."""
    matches = list(OPTION_LINE.finditer(problem))
    start = next((index for index, match in enumerate(matches) if match.group(1) == "A"), len(matches))
    # Source questions can repeat a complete choice list, then append a revised
    # list. Use the final list, without hiding duplicate labels within one list.
    for index in range(start + 1, len(matches)):
        if matches[index].group(1) != "A":
            continue
        preceding = {match.group(1) for match in matches[start:index]}
        maximum = max(preceding)
        complete = preceding == {chr(letter) for letter in range(65, ord(maximum) + 1)} and maximum != "A"
        following = {match.group(1) for match in matches[index:]}
        complete = complete and "B" in following
        gap = problem[matches[index - 1].end() : matches[index].start()]
        premise = maximum == "A" and ("\n\n" in gap or len(gap.strip().splitlines()) > 1 or "?" in gap)
        if complete or premise:
            start = index
    end = len(matches)
    for index in range(start + 1, len(matches)):
        gap = problem[matches[index - 1].end() : matches[index].start()]
        if gap.partition("\n\n")[2].lstrip().startswith("(Note:"):
            end = index
            break
    choices = dict.fromkeys(
        (
            matches[index].group(1),
            problem[matches[index].end() : matches[index + 1].start() if index + 1 < end else len(problem)]
            .partition("\n\n(Note:")[0]
            .strip(),
        )
        for index in range(start, end)
    )
    return tuple(label for label, _ in choices)


def mcqa_question(instruction: str) -> tuple[str, tuple[str, ...]]:
    """The question and its choice labels, with source delivery wrappers removed."""
    _, separator, problem = instruction.partition(MCQA_SEPARATOR)
    if not separator:
        raise ValueError("Unrecognized MCQA delivery wrapper")
    if not problem.strip():
        raise ValueError("Missing MCQA question/options")
    # Some archives contain literal backslash-n separators. Decode only option boundaries,
    # preserving mathematical escapes and any literal backslashes in the question.
    problem = ESCAPED_OPTION_NEWLINE.sub("\n", problem)
    labels = mcqa_option_labels(problem)
    options = max((ord(label) - 64 for label in labels), default=0)
    letters = tuple(chr(65 + index) for index in range(options))
    if not labels:
        raise ValueError("Missing labeled options beginning at A")
    first, separator, rest = problem.partition("\n\n")
    if first.startswith(MCQA_FORMAT_PREFIX):
        listed = LISTED_OPTIONS.search(first)
        # The generated delivery wrapper can contain stale labels. Its syntax identifies
        # the wrapper; the actual question determines which answers are available.
        option_lists = (listed.group(1),) if listed is not None else ()
        valid_formats = {
            f"{MCQA_FORMAT_PREFIX}'Answer: {wrapper.format(listed_options)}' "
            f"(e.g. 'Answer: {wrapper.format(example)}')."
            for wrapper in MCQA_FORMAT_WRAPPERS
            for listed_options in option_lists
            for example in listed_options.split("/")
        }
        if first not in valid_formats or not separator or not rest.strip():
            raise ValueError("Unsupported MCQA format wrapper")
        problem = rest
    unique_labels = tuple(dict.fromkeys(labels))
    choices = f"A through {chr(64 + options)}" if unique_labels == letters else ", ".join(unique_labels)
    return f"{problem.strip()}\n\nReturn one option letter from {choices}.", labels


def convert_knowledge_mcqa(row: RawRow, _context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
    instruction, data = row.data["instruction"], row.data.get("verifier_data")
    if not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and verifier_data are required")
    if data.get("output_regex") not in MCQA_EXTRACTIONS:
        return unsupported("unsupported_answer_contract", "Unsupported source MCQA extraction regex")
    try:
        question, labels = mcqa_question(instruction)
    except ValueError as error:
        return unsupported("unsupported_answer_contract", str(error))
    answer = data.get("expected_answer")
    if not isinstance(answer, str) or answer.strip().upper() not in labels:
        return source_defect("invalid_reference", f"The key must name a listed option: {answer!r}")
    if labels.count(answer.strip().upper()) > 1:
        return unsupported("unsupported_answer_contract", "The reference names multiple listed options")
    task = mcq_task(row, prompt=question, answer=answer, options=max(ord(label) - 64 for label in labels))
    if isinstance(task, ImportRejection):
        return task
    task = task.model_copy(update={"tags": ("qa", "mcq", "nemotron")})
    return rewritten_task(task, original=instruction, reason=MCQA_REWRITE_REASON)


def sources() -> list[RlDataSource[CurationRecipe]]:
    openqa = (
        (
            "knowledge-openqa",
            "laion__nemotron-gym-knowledge-openqa-v4",
            SourceInfo(
                id="Task Trove:laion__nemotron-gym-knowledge-openqa-v4",
                title="laion/nemotron-gym-knowledge-openqa-v4",
                origin="Task Trove",
                family="qa-short-answer",
                tags=("agentic", "multi-turn"),
                count=122357,
                notes=(
                    "Exact gate then reference-based judge. Reference answers are real; needs the judge "
                    "in the new grader."
                ),
            ),
        ),
        (
            "science-openqa",
            "laion__nemotron-gym-science-so-openq-v3",
            SourceInfo(
                id="Task Trove:laion__nemotron-gym-science-so-openq-v3",
                title="laion/nemotron-gym-science-so-openq-v3",
                origin="Task Trove",
                family="llm-judge-freeform",
                tags=("agentic", "multi-turn"),
                count=150644,
                notes=(
                    "Reference answer plus judge. Move the reference out of the criterion text into a "
                    "data field at conversion."
                ),
            ),
        ),
    )
    return [
        *(
            RlDataSource(
                pipeline=process_rows,
                info=info,
                config=CurationRecipe(
                    name=name,
                    source=tasktrove_source(config),
                    convert=TaskTroveConverter(config, convert_openqa),
                    version="3",
                    intended_use=IntendedUse.TRAIN,
                    rubric=OPENQA_RUBRIC,
                    grader=GRADER_PACKAGES,
                ),
            )
            for name, config, info in openqa
        ),
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id=f"Task Trove:{MCQA_CONFIG}",
                title="laion/nemotron-gym-knowledge-mcqa-v2",
                origin="Task Trove",
                family="qa-short-answer",
                tags=("agentic", "multi-turn"),
                count=616888,
                notes=(
                    "Held-out MCQA with regex extraction. Drop the trailing-letter fallback at conversion "
                    "and subsample hard: 617k rows is a third of the corpus."
                ),
            ),
            config=CurationRecipe(
                name="tasktrove-knowledge_mcqa",
                source=tasktrove_source(MCQA_CONFIG),
                convert=TaskTroveConverter(MCQA_CONFIG, convert_knowledge_mcqa),
                version="5",
                intended_use=IntendedUse.TRAIN,
                rubric=KNOWLEDGE_MCQA_RUBRIC,
                controls=Controls(golden=reference_reply),
            ),
        ),
    ]
