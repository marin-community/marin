# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove instruction-following sources, graded in process by verifyit.

The IFEval source's constraints map directly to verifyit's IFEval checks, which score the fraction
satisfied. Structured tasks accept any schema-valid instance. Conversion excludes malformed,
vacuous or contradictory schemas and conflicting public language requirements.
"""

import json
import re

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from taskcompendium.convert.answers import ifeval_task, json_schema_task, source_defect, unsupported
from taskcompendium.convert.delivery import rewritten_task
from taskcompendium.convert.json_schema import required_object_conflicts
from taskcompendium.convert.tasktrove_json_schemas import is_trivial
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, NormalizedTask, RawRow
from verifyit.grade import InvalidTask
from verifyit.modes.grade_ifeval import resolve_checks
from verifyit.spec import Constraint, SchemaFormat

from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter, tasktrove_source
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim, recipe_source
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

IFEVAL_CONFIG = "laion__nemotron-gym-instruction-following-v3"
STRUCTURED_CONFIG = "laion__nemotron-gym-instruction-following-structured-v3"

SHELL_PREAMBLE = "You are running in a shell-based sandbox."
PREAMBLE_SEPARATOR = "\n---\n"
IFEVAL_REWRITE_REASON = "Remove the source's shell answer-file preamble"
EXCLUSIVE_LANGUAGE = ("Your ENTIRE response should be in ", "no other language is allowed")
NON_LATIN_LANGUAGES = frozenset({"ar", "bg", "bn", "he", "hi", "ja", "ko", "ne", "ru", "ta", "te", "th", "zh"})
LATIN_WORD = re.compile(r"[A-Za-z]+")
POSITIONAL_WORDS = {
    "first_word:first_word_answer": "first_word",
    "first_word:first_word_sent": "first_word",
    "last_word:last_word_answer": "last_word",
    "last_word:last_word_sent": "last_word",
    "length_constraints:nth_paragraph_first_word": "first_word",
}
"""Constraints that require a specific word, mapped to the parameter naming it."""

JSON_DELIVERY = ("Write your final JSON to `/app/answer.txt`.", "Return your final JSON in the assistant response.")
STRUCTURED_REWRITE_REASON = "Replace the source's answer-file delivery with an answer in the assistant response"

IFEVAL_RUBRIC = """
Identify the underlying content request independently of the formatting constraints. A topic fragment, random text, or
missing question is not an answerable task; do not invent what the user meant.

Check that the content request and ALL constraints can be satisfied together. Reject contradictory counts and answer
formats that cannot express the requested answer, even if the formal checker can pass.

Reject requests depending on absent documents, profiles, prior turns, or unavailable tools. Asking for clarification is
not a complete answer to a missing-input task.

Distinguish an actual missing-input deliverable from a capability/setup question such as 'could you review an email for
me?'; that question can be answered by requesting the email. A promised but absent article needed for an actual summary
or analysis remains missing context.

Apply language requirements literally. ENTIRELY or ONLY in one language conflicts with mandatory words from another
language unless an explicit exception is supplied. Do not invent an exception. A question written in Chinese does not
by itself require an exclusively Chinese answer.

Structural markers (P.S., markdown delimiters), mathematical notation, and code syntax are not foreign lexical words.
Treat unclear marker case or delimiter precedence as uncertainty; do not invent an exception to an explicit prohibition
on capitals, punctuation, or trailing characters.

Multilingual requests, spelling errors, and hard but feasible constraints alone are not defects. Require a meaningful,
comprehensible answer; flag constraints that destroy that answer rather than merely making it difficult or unusual.

Check the hidden constraint identifiers and parameters against the public wording. The checker measures formal
compliance only; it does not certify factual correctness or semantic usefulness.

For a constraint-only verifier, reference_status concerns agreement between public constraints and checker
configuration. There is no canonical answer key; its absence alone is not missing context, a reason for
reference_status=unknown, or a reason to downgrade a clearly answerable task.

Distinguish a contradiction in the written instructions from impossibility under the checker. Some shared checks
approximate language, word counts, or sentence structure; passing them does not repair a bad prompt.

Shared checker details: copy:repeat_phrase requires at least N unchanged case-insensitive substring occurrences, so
transformed variants do not satisfy it. letters:letter_counting counts ASCII word matches rather than letters.
count:count_unique requires at least five unique ASCII words and 50% uniqueness, not all words unique.
startend:end_checker uses literal endswith, including closing delimiters. Use these facts to identify concrete
prompt/checker mismatches; under-enforcement alone does not make an otherwise meaningful content request unanswerable.

Assess whether a short answer can still convey the requested content. Do not infer impossibility just because a story
must be compressed or a constraint makes the task difficult. Use quality=good and high confidence when there is no
material defect; reserve uncertainty for a specific unresolved issue.
"""

STRUCTURED_RUBRIC = """
The opening Evaluation contract is authoritative: any schema-valid instance is acceptable, and unstated values may be
chosen. Do not misclassify this as extraction of one hidden reference document.

Flag contradictory lower-priority extraction or grounding instructions as ambiguity when they can confuse a solver.
Remove them in a separate rewrite, preserving the authoritative contract and all supplied facts.

The public JSON Schema must match the grader's schema. Never relax a required field, type, enum, or bound.

Compare complete object structure, including root type, required, additionalProperties, and definitions. Matching
nested field definitions is insufficient when the public schema is only a property map and the grader's schema adds
hidden root requirements. Cite a specific omitted constraint rather than claiming the schemas are identical without
checking them.

Check satisfiability: required properties forbidden by additionalProperties=false make a mandatory object impossible.
Check mandatory nested objects too. Meta-schema validity does not prove that an instance exists. Misplaced keywords may
be ignored rather than making the schema unsatisfiable.

Require meaningful fields, not only formally valid strings: a contact phone pattern excluding all digits cannot express
the described phone number. The any-valid-instance contract does not make that defect disappear. Do not invent units
or facts when a numeric field cannot represent stated content.

Flag quote-all-values boilerplate when it contradicts required numeric or boolean types. This can need a wording repair
while preserving the authoritative contract and the exact schema. Ordinary restructuring language subordinate to a
clear generation contract is not by itself a defect.

A schema-valid instance demonstrates structural feasibility only. The grader does not enforce grounding or format
annotations; report a mismatch if the prompt requires checks the grader does not implement.
"""


def language_conflicts(prompt: str, constraints: tuple[Constraint, ...]) -> list[str]:
    """Mandatory Latin words that an explicitly exclusive non-Latin response language forbids.

    This bounded rule does not infer a language from the question or classify foreign words across
    languages sharing a script.
    """
    if not all(phrase in prompt for phrase in EXCLUSIVE_LANGUAGE):
        return []
    languages = {
        constraint.params.get("language")
        for constraint in constraints
        if constraint.name == "language:response_language"
    }
    if not languages & NON_LATIN_LANGUAGES:
        return []
    words = (
        constraint.params.get(parameter)
        for constraint in constraints
        if (parameter := POSITIONAL_WORDS.get(constraint.name)) is not None
    )
    return [word for word in words if isinstance(word, str) and LATIN_WORD.fullmatch(word)]


def convert_ifeval(row: RawRow, _context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
    instruction, data = row.data["instruction"], row.data.get("verifier_data")
    if not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and verifier_data are required")
    prompt = instruction
    if prompt.startswith(SHELL_PREAMBLE):
        _, separator, prompt = prompt.partition(PREAMBLE_SEPARATOR)
        if not separator:
            return unsupported("unknown_wrapper", "The shell preamble has no task separator")
    names, parameters = data.get("instruction_id_list"), data.get("kwargs")
    if not isinstance(names, list) or not isinstance(parameters, list) or len(names) != len(parameters):
        return unsupported("invalid_constraints", "Constraint names and parameters must align")
    if not all(
        isinstance(name, str) and isinstance(params, dict) for name, params in zip(names, parameters, strict=True)
    ):
        return unsupported("invalid_constraints", "Constraint names and parameters must be objects")
    constraints = tuple(Constraint(name=name, params=params) for name, params in zip(names, parameters, strict=True))
    try:
        resolve_checks(constraints)
    except (InvalidTask, ValueError) as error:
        return unsupported("invalid_constraints", str(error))
    conflicts = language_conflicts(prompt, constraints)
    if conflicts:
        return source_defect(
            "exclusive_language_positional_word",
            f"Exclusive non-Latin response language conflicts with mandatory Latin words {conflicts!r}",
        )
    task = ifeval_task(row, prompt=prompt.strip(), constraints=constraints)
    if isinstance(task, ImportRejection):
        return task
    task = task.model_copy(update={"tags": ("instruction-following", "ifeval", "nemotron")})
    return rewritten_task(task, original=instruction, reason=IFEVAL_REWRITE_REASON)


def convert_structured(row: RawRow, _context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
    instruction, data = row.data["instruction"], row.data.get("verifier_data")
    if not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and verifier_data are required")
    schema = data.get("schema")
    if data.get("schema_type") != "json" or not isinstance(schema, dict):
        return unsupported("unsupported_schema", "Expected an explicit JSON Schema object")
    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as error:
        return source_defect("invalid_schema", str(error))
    if is_trivial(schema):
        return source_defect("null_grader", "Schema has no properties or required fields to check")
    conflicts = required_object_conflicts(schema)
    if conflicts:
        return source_defect("unsatisfiable_schema", "; ".join(conflicts))
    task = json_schema_task(
        row,
        prompt=instruction.replace(*JSON_DELIVERY),
        schema=json.dumps(schema),
        schema_format=SchemaFormat.JSON,
    )
    if isinstance(task, ImportRejection):
        return task
    task = task.model_copy(update={"tags": ("instruction-following", "structured-output", "json-schema", "nemotron")})
    return rewritten_task(task, original=instruction, reason=STRUCTURED_REWRITE_REASON)


def sources() -> list[RlDataSource[RlDataPipeline]]:
    return [
        recipe_source(
            info=SourceInfo(
                id=f"Task Trove:{IFEVAL_CONFIG}",
                title="laion/nemotron-gym-instruction-following-v3",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn"),
                count=46391,
                notes=(
                    "IFEval-style deterministic checkers. Filter rows with empty constraint lists "
                    "(vacuous pass) at conversion."
                ),
            ),
            pipeline=RlDataPipeline(
                name="tasktrove-ifeval",
                source=tasktrove_source(IFEVAL_CONFIG),
                convert=TaskTroveConverter(IFEVAL_CONFIG, convert_ifeval),
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=IFEVAL_RUBRIC,
            ),
        ),
        recipe_source(
            info=SourceInfo(
                id=f"Task Trove:{STRUCTURED_CONFIG}",
                title="laion/nemotron-gym-instruction-following-structured-v3",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn"),
                count=9437,
                notes="Any schema-valid instance is accepted, and jsonschema does the grading.",
            ),
            pipeline=RlDataPipeline(
                name="tasktrove-structured",
                source=tasktrove_source(STRUCTURED_CONFIG),
                convert=TaskTroveConverter(STRUCTURED_CONFIG, convert_structured),
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=STRUCTURED_RUBRIC,
                controls=Controls(),
            ),
        ),
    ]
