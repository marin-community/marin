# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converters for Nemotron Ultra components, each fixing the component's grader.

A graded component's tasks ship a ``*_grade.py`` script from beside this module as ``/tests/grade.py``,
with the vendored NeMo Gym scorer it calls from ``scorers/`` (calendar, format, instruction-following,
multiple-choice, structured-output, competitive-code, RDKit chemistry or single-step tool-action) and
the row's grading contract as ``/tests/config.json``. Reasoning Gym rows are scored by the puzzle task's
own scorer in the grader image's ``reasoning_gym``. Every script runs in the grader image
(``images.recipes.GRADER``). Components whose NeMo Gym agent needs a model judge, a live environment or
a Lean toolchain keep their row as a ``NoGrader`` contract.
"""

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.code import (
    CODE_GRADER_MEMORY_MB,
    FAILING_PROGRAM,
    THREAD_ENVIRONMENT,
    python_reply,
    validate_code_cases,
)
from taskcompendium.convert.nemotron_ultra import (
    PLACEHOLDER_FIELD,
    BlendRequest,
    agent_request,
    blend_request,
    blend_task,
    text_request,
)
from taskcompendium.grader import grader_config
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationToolCall,
    EnvironmentRequirements,
    FinalAction,
    NoGrader,
    PlainText,
    ProviderRequirement,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import (
    Controls,
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    Reply,
)
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.grade_scripts import (
    ANSWER_PATH,
    grade_script,
    grader_package,
    vendored_files,
)

VERIFIER_REVISION = "d8b6e8c163def3660e9d3072c1c174226a1709fa"
"""The NeMo Gym revision whose agents grade the pinned Ultra blends."""
PLACEHOLDER_SOURCE_FIELD = "placeholder_source"
AGENT_CAPABILITY_PREFIX = "nemotron-agent:"
SCORER_TIMEOUT = 60.0
CODE_SCORER_TIMEOUT = 330.0

HERE = Path(__file__).parent
SCORERS = HERE / "scorers"
SKYRL_SCORERS = HERE.parent / "skyrl" / "scorers"
SHIPS = (SCORERS,)
CODE_SHIPS = (SCORERS, SKYRL_SCORERS)
"""The vendored directories a component's tasks ship files from; code tasks also ship LiveCodeBench."""
ULTRA_ENVS = "skyrl_gym/envs/nemotron_ultra"
ULTRA_BASE = vendored_files(
    SCORERS,
    "skyrl_gym/__init__.py",
    "skyrl_gym/envs/__init__.py",
    "skyrl_gym/envs/aime/utils.py",
    f"{ULTRA_ENVS}/__init__.py",
    f"{ULTRA_ENVS}/answer_extraction.py",
)
"""The final-answer extractor every Ultra grade script imports, with the packages around it."""
LIVECODEBENCH = inline_resource("skyrl_gym/envs/lcb/livecodebench.py", (SKYRL_SCORERS / "livecodebench.py").read_bytes())
"""The one vendored LiveCodeBench evaluator, at the path ``code_gen`` imports it from."""


def ultra_grade_script(script: str, *modules: str) -> tuple[TaskResource, ...]:
    """A grade script beside this module with the Ultra base and the named ``nemotron_ultra`` modules."""
    return grade_script(HERE / script, *ULTRA_BASE, *vendored_files(SCORERS, *(f"{ULTRA_ENVS}/{m}.py" for m in modules)))


CALENDAR_GRADE = ultra_grade_script("calendar_grade.py", "calendar")
FORMAT_GRADE = ultra_grade_script("format_grade.py", "format_verification")
INSTRUCTION_FOLLOWING_GRADE = ultra_grade_script("instruction_following_grade.py", "instruction_following")
MCQA_GRADE = ultra_grade_script("mcqa_grade.py", "mcqa")
CODE_GRADE = (*ultra_grade_script("code_grade.py", "code_gen"), LIVECODEBENCH)
STRUCTURED_OUTPUT_GRADE = ultra_grade_script("structured_output_grade.py", "structured_outputs")
RDKIT_GRADE = ultra_grade_script("rdkit_chemistry_grade.py", "rdkit_chemistry")
TOOL_ACTION_GRADE = ultra_grade_script("expected_action_grade.py", "tool_call")
REASONING_GYM_GRADE = ultra_grade_script("reasoning_gym_grade.py")

TOOL_ACTION_AGENTS = (
    "single_step_tool_use_with_argument_comparison_agent",
    "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
)
RDKIT_PROPERTIES = frozenset({"count", "bool", "presence", "fragment"})
STRUCTURED_SCHEMA_TYPES = frozenset({"json", "yaml", "toml", "xml", "csv"})
FORMAT_VERIFIER_TYPES = frozenset({"regex", "inline_prose", "string_match"})
WRONG_TOOL = "__wrong_tool__"

DAPO = "BytedTsinghua-SIA/DAPO-Math-17k"
DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem."
)
DAPO_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'


def _scored(
    row: RawRow,
    request: BlendRequest,
    files: tuple[TaskResource, ...],
    context: ConversionContext,
    *,
    answer_type: AnswerType = AnswerType.TEXT,
    timeout: float = SCORER_TIMEOUT,
    env: Mapping[str, str] | None = None,
) -> NormalizedTask:
    package = grader_package(
        files,
        {"contract": request.contract},
        environment=required_grader_environment(context),
        timeout=timeout,
        answer_path=ANSWER_PATH,
        env=env,
    )
    return blend_task(row, request, package, answer_type)


def _text_scored(
    row: RawRow, context: ConversionContext, agents: tuple[str, ...], files: tuple[TaskResource, ...]
) -> NormalizedTask | ImportRejection:
    request = text_request(row.data, agents)
    if isinstance(request, ImportRejection):
        return request
    return _scored(row, request, files, context)


def convert_calendar(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    return _text_scored(row, context, ("calendar_simple_agent",), CALENDAR_GRADE)


def convert_format(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    verifier = row.data.get("verifier")
    kind = verifier.get("type") if isinstance(verifier, dict) else None
    if kind not in FORMAT_VERIFIER_TYPES:
        return unsupported("unsupported_format_verifier", str(kind))
    agents = ("citation_format_simple_agent", "freeform_formatting_simple_agent")
    return _text_scored(row, context, agents, FORMAT_GRADE)


def convert_instruction_following(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    return _text_scored(row, context, ("instruction_following_simple_agent",), INSTRUCTION_FOLLOWING_GRADE)


def convert_mcqa(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    return _text_scored(row, context, ("mcqa_simple_agent",), MCQA_GRADE)


def convert_reasoning_gym(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Score the reply with the Reasoning Gym task its metadata names."""
    return _text_scored(row, context, ("reasoning_gym_simple_agent",), REASONING_GYM_GRADE)


def convert_code(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Score the last fenced program against the row's hidden unit tests, one thread per test."""
    request = text_request(row.data, ("code_gen_simple_agent",))
    if isinstance(request, ImportRejection):
        return request
    try:
        validate_code_cases(row.data["verifier_metadata"]["unit_tests"])
    except (KeyError, TypeError, ValueError) as error:
        return source_defect("invalid_code_tests", str(error))
    return _scored(row, request, CODE_GRADE, context, timeout=CODE_SCORER_TIMEOUT, env=THREAD_ENVIRONMENT)


def convert_structured_output(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Validate a text document, or the payload of one typed tool call, against the row's schema."""
    mode = row.data.get("response_mode", "text")
    if mode not in {"text", "tool_call"}:
        return unsupported("unsupported_structured_response_mode", str(mode))
    schema_type = str(row.data.get("schema_type", "json")).lower()
    if mode == "text" and schema_type not in STRUCTURED_SCHEMA_TYPES:
        return unsupported("unsupported_structured_schema_type", schema_type)
    agents = ("structured_outputs_simple_agent", "structured_outputs_v3_simple_agent")
    request = agent_request(row.data, agents) if mode == "tool_call" else text_request(row.data, agents)
    if isinstance(request, ImportRejection):
        return request
    if mode == "tool_call" and not request.tools:
        return unsupported(
            "missing_structured_tool_schema", "Typed terminal tool grading requires the public tool declaration"
        )
    try:
        # The scorer's OpenAPI validator checks the fixed 2020-12 metaschema even when the row
        # declares another draft.
        Draft202012Validator.check_schema(json.loads(row.data["schema_str"]))
    except (json.JSONDecodeError, SchemaError) as error:
        return source_defect("invalid_structured_schema", str(error))
    answer_type = AnswerType.NATIVE_ACTION if mode == "tool_call" else AnswerType.TEXT
    return _scored(row, request, STRUCTURED_OUTPUT_GRADE, context, answer_type=answer_type)


def convert_rdkit(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Compare the rounded wrapped answer with the stored molecular property target."""
    request = agent_request(row.data, ("rdkit_chemistry_agent",))
    if isinstance(request, ImportRejection):
        return request
    if row.data["property_type"] not in RDKIT_PROPERTIES:
        return unsupported("unsupported_chemistry_property", str(row.data["property_type"]))
    if not math.isfinite(float(row.data["expected_answer"])):
        return source_defect("nonfinite_chemistry_target", "The rounded comparator requires a finite target")
    return _scored(row, request, RDKIT_GRADE, context)


def _tool_action(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    request = agent_request(row.data, TOOL_ACTION_AGENTS)
    if isinstance(request, ImportRejection):
        return request
    kind = row.data["expected_action"]["type"]
    if kind not in {"message", "function_call"}:
        return unsupported("unsupported_action_type", str(kind))
    return _scored(row, request, TOOL_ACTION_GRADE, context, answer_type=AnswerType.NATIVE_ACTION)


def convert_toolcall_schema(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Compare one predicted action with the expected call, without executing the tool."""
    return _tool_action(row, context)


def convert_next_action(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Compare a predicted SWE agent action with the expected call, without executing the tool."""
    return _tool_action(row, context)


def _agent_provider(request: BlendRequest) -> dict[str, ProviderRequirement]:
    """The NeMo Gym agent that serves the request's tools and initial environment."""
    interface = AGENT_CAPABILITY_PREFIX + request.agent
    return {"nemotron_agent": ProviderRequirement(action_interface=interface, initial_state=request.state)}


def _ungraded(
    row: RawRow,
    request: BlendRequest,
    providers: dict[str, ProviderRequirement],
    changes: tuple[NormalizationChange, ...] = (),
) -> NormalizedTask:
    """Keep the conversation and the agent's grading contract; no grader runs here."""
    expected_action = request.contract.get("expected_action", {})
    answers_with_call = isinstance(expected_action, dict) and expected_action.get("type") == "function_call"
    action = answers_with_call and bool(request.tools)
    grader = NoGrader(
        reason=f"The NeMo Gym agent {request.agent} at revision {VERIFIER_REVISION} has no runnable grader here",
        contract={"evaluator": request.agent, "source_revision": VERIFIER_REVISION, "contract": request.contract},
    )
    task = TaskSpec(
        id=row.id,
        source=row.source,
        context=request.context,
        environment_requirements=EnvironmentRequirements(
            capabilities=(AGENT_CAPABILITY_PREFIX + request.agent,), tool_providers=providers
        ),
        final_tools=request.tools,
        interaction_tools=request.tools,
        answer_type=AnswerType.NATIVE_ACTION if action else AnswerType.TEXT,
        answer_format=FinalAction() if action else PlainText(),
        grader=grader,
    )
    return NormalizedTask(task, (*changes, *request.changes))


def convert_ungraded(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    """A conversation component whose agent cannot run here."""
    request = blend_request(row.data)
    if isinstance(request, ImportRejection):
        return request
    return _ungraded(row, request, _agent_provider(request) if request.tools else {})


def convert_ungraded_agent(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    """An agent-environment component, whose initial state the NeMo Gym agent also serves."""
    request = blend_request(row.data)
    if isinstance(request, ImportRejection):
        return request
    return _ungraded(row, request, _agent_provider(request) if request.tools or request.state else {})


def restore_placeholder(data: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[NormalizationChange, ...]]:
    """Apply the Ultra release's ``fill_placeholders.py`` to a row with its attached upstream record."""
    placeholder = data[PLACEHOLDER_FIELD]
    source = data[PLACEHOLDER_SOURCE_FIELD]
    if (source["dataset"], source["split"], source["row_index"]) != (
        placeholder["dataset"],
        placeholder["split"],
        int(placeholder["row"]),
    ):
        raise ValueError("Placeholder source identity does not match its declared recipe")
    record = source["record"]
    bare = record["prompt"][0]["content"]
    if placeholder["dataset"] == DAPO:
        if DAPO_PREFIX in bare:
            bare = bare.split(DAPO_PREFIX, 1)[1]
        if DAPO_SUFFIX in bare:
            bare = bare.rsplit(DAPO_SUFFIX, 1)[0]
    bare = bare.strip()
    if placeholder.get("mode") == "canonical":
        question = placeholder.get("lead", "") + bare + placeholder.get("trail", "")
    else:
        question = placeholder.get("prefix", "") + bare + placeholder.get("suffix", "")
    raw = record["reward_model"]["ground_truth"]
    if isinstance(raw, list) and raw:
        answer = str(raw[0])
    elif isinstance(raw, str):
        answer = raw.strip()
        if answer.startswith(("[", "{")) and answer.endswith(("]", "}")):
            try:
                parsed = json.loads(answer)
            except json.JSONDecodeError:
                # The release's unwrap_answer keeps free-form math such as {1, 2}.
                parsed = answer
            answer = str(parsed[0]) if isinstance(parsed, list) and parsed else str(parsed)
    else:
        raise ValueError("Placeholder source lacks a supported ground_truth")
    if not bare or not answer:
        raise ValueError("Placeholder source question and answer must be nonempty")
    restored = copy.deepcopy(dict(data))
    restored.pop(PLACEHOLDER_FIELD)
    restored.pop(PLACEHOLDER_SOURCE_FIELD)
    restored["placeholder_provenance"] = {key: value for key, value in source.items() if key != "record"}
    restored["question"] = question
    restored["expected_answer"] = answer
    restored["responses_create_params"]["input"][0]["content"] = question
    for matched in restored.get("matched_sources", []):
        if "expected_answer" in matched:
            matched["expected_answer"] = answer
    changes = (
        NormalizationChange(
            field="question",
            reason="Apply the pinned release's explicit external-source placeholder recipe",
            original=json.dumps(placeholder, ensure_ascii=False),
            replacement=question,
        ),
        NormalizationChange(
            field="expected_answer",
            reason="Restore the placeholder source reward_model.ground_truth as directed by the release",
            original=str(data.get("expected_answer", "")),
            replacement=answer,
        ),
    )
    return restored, changes


def convert_math(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    """A math component, with questions held by DAPO or Skywork placeholders restored first."""
    data: Mapping[str, Any] = row.data
    changes: tuple[NormalizationChange, ...] = ()
    if data.get(PLACEHOLDER_FIELD) and PLACEHOLDER_SOURCE_FIELD in data:
        try:
            data, changes = restore_placeholder(data)
        except (ValueError, KeyError, TypeError) as error:
            return unsupported("invalid_placeholder_source", str(error))
    request = blend_request(data)
    if isinstance(request, ImportRejection):
        return request
    return _ungraded(row, request, _agent_provider(request) if request.tools else {}, changes)


def mcqa_reference(contract: Mapping[str, Any]) -> str | None:
    """The gold letter in the row's answer format, for the standard letter grading modes only."""
    if "output_regex" in (contract.get("template_metadata") or {}):
        return None
    gold = str(contract.get("expected_answer") or "").strip().upper()
    allowed = {
        key.upper()
        for option in contract.get("options") or []
        for key, value in option.items()
        if isinstance(key, str) and len(key) == 1 and key.isalpha() and value is not None
    }
    if gold not in allowed:
        return None
    mode = contract.get("grading_mode", "strict_single_letter_boxed")
    if mode in {"strict_single_letter_boxed", "lenient_boxed"}:
        return "\\boxed{" + gold + "}"
    if mode == "lenient_answer_colon":
        return "Answer: " + gold
    if mode == "lenient_answer_colon_md":
        return "**Answer**: " + gold
    return None


def mcqa_golden(task: TaskSpec) -> Reply | None:
    reference = mcqa_reference(grader_config(task)["contract"])
    return answer_reply(task, reference) if reference is not None else None


def code_golden(task: TaskSpec) -> Reply | None:
    """Only a source-provided program can be a known-correct submission."""
    reply = python_reply(grader_config(task)["contract"].get("gold_standard_solution"))
    return answer_reply(task, reply) if reply is not None else None


def code_negative(task: TaskSpec) -> Reply:
    return answer_reply(task, FAILING_PROGRAM)


def reasoning_gym_golden(task: TaskSpec) -> Reply | None:
    answer = grader_config(task)["contract"].get("answer")
    return answer_reply(task, answer) if isinstance(answer, str) else None


def _rdkit_reply(task: TaskSpec, offset: int) -> Reply:
    contract = grader_config(task)["contract"]
    value = round(float(contract["expected_answer"])) + offset
    return answer_reply(task, rf"\boxed{{{value}}}" if contract.get("use_box_format", False) else f"(({value}))")


def rdkit_golden(task: TaskSpec) -> Reply:
    """The rounded target in the row's answer wrapper."""
    return _rdkit_reply(task, 0)


def rdkit_negative(task: TaskSpec) -> Reply:
    return _rdkit_reply(task, 1)


def _expected_call(expected: Mapping[str, Any], name: str) -> AssistantToolCalls:
    arguments = json.loads(expected["arguments"]) if expected["type"] == "function_call" else {}
    return AssistantToolCalls(calls=(ConversationToolCall(call_id="control", name=name, arguments=arguments),))


def tool_action_golden(task: TaskSpec) -> Reply:
    """The expected call; an expected message accepts any text reply without calls."""
    expected = grader_config(task)["contract"]["expected_action"]
    if expected["type"] == "function_call":
        return Reply(_expected_call(expected, expected["name"]))
    return Reply(TextMessage(role="assistant", content="A nonliteral response."))


def tool_action_negative(task: TaskSpec) -> Reply:
    """A call to a tool the task never advertised."""
    return Reply(_expected_call(grader_config(task)["contract"]["expected_action"], WRONG_TOOL))


# Components whose rows carry no known answer check only that an empty reply scores zero.
REPLY_CONTROLS = Controls()
MCQA_CONTROLS = Controls(golden=mcqa_golden)
CODE_CONTROLS = Controls(golden=code_golden, negative=code_negative, memory_mb=CODE_GRADER_MEMORY_MB)
RDKIT_CONTROLS = Controls(golden=rdkit_golden, negative=rdkit_negative)
TOOL_ACTION_CONTROLS = Controls(golden=tool_action_golden, negative=tool_action_negative)
# Reasoning Gym scorers give partial credit, so a fixed wrong answer has no single expected reward;
# only the row's answer is checked.
REASONING_GYM_CONTROLS = Controls(golden=reasoning_gym_golden)
