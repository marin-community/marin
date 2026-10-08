# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converters for Nemotron Ultra components, each fixing the component's grader.

Graded components call the NeMo Gym scorer that the grader image installs under
``skyrl_gym.envs.nemotron_ultra``, unchanged, through ``source_callable.py``: calendar, format,
instruction-following, structured-output, competitive-code and single-step tool-action scorers run
in ``NEMOTRON_ULTRA_IMAGE``, multiple-choice in ``ULTRA_MCQA_IMAGE``, and RDKit chemistry and
tool-call-schema actions in ``EXECUTABLE_MATH_IMAGE``. Components whose NeMo Gym agent needs a
model judge, a live environment or a Lean toolchain keep their row as a ``NoGrader`` contract.
"""

import copy
import json
import math
from collections.abc import Mapping
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
from taskcompendium.convert.source_scorer import source_scorer_package
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
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.models import (
    Controls,
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    Reply,
)

from experiments.post_training.task_curation.images import (
    EXECUTABLE_MATH_IMAGE,
    NEMOTRON_ULTRA_IMAGE,
    ULTRA_MCQA_IMAGE,
)
from experiments.post_training.task_curation.pipeline import Image

VERIFIER_REVISION = "d8b6e8c163def3660e9d3072c1c174226a1709fa"
"""The NeMo Gym revision whose agents grade the pinned Ultra blends."""
PLACEHOLDER_SOURCE_FIELD = "placeholder_source"
AGENT_CAPABILITY_PREFIX = "nemotron-agent:"
SCORER_TIMEOUT = 60.0
CODE_SCORER_TIMEOUT = 330.0
STATE_PATH = "/app/state.json"
"""Where the runtime writes the captured terminal message, which the code scorer inspects."""

ANSWER_EXTRACTOR = "skyrl_gym.envs.nemotron_ultra.answer_extraction:final_answer_text"
CALENDAR_SCORER = {
    "function": "skyrl_gym.envs.nemotron_ultra.calendar:grade_calendar",
    "args": ["answer", "contract.exp_cal_state"],
    "answer_extractor": ANSWER_EXTRACTOR,
}
FORMAT_SCORER = {
    "function": "skyrl_gym.envs.nemotron_ultra.format_verification:grade_format",
    "args": ["answer", "contract.verifier"],
    "answer_extractor": ANSWER_EXTRACTOR,
}
INSTRUCTION_FOLLOWING_SCORER = {
    "function": "skyrl_gym.envs.nemotron_ultra.instruction_following:grade_instruction_following",
    "args": ["answer", "contract"],
    "answer_extractor": ANSWER_EXTRACTOR,
}
MCQA_SCORER = {
    "function": "skyrl_gym.envs.nemotron_ultra.mcqa:grade_mcqa",
    "args": ["answer", "contract"],
    "answer_extractor": ANSWER_EXTRACTOR,
}
CODE_SCORER = {
    "function": "skyrl_gym.envs.nemotron_ultra.code_gen:grade_code",
    "args": ["answer", "contract"],
    "answer_extractor": ANSWER_EXTRACTOR,
    "kwargs": {"assistant_message": "terminal_message"},
}
STRUCTURED_OUTPUT_SCORER = {
    "function": "skyrl_gym.envs.nemotron_ultra.structured_outputs:grade_structured_output",
    "args": ["answer", "contract", "terminal_message"],
    "answer_extractor": ANSWER_EXTRACTOR,
}
RDKIT_SCORER = {
    "function": "skyrl_gym.envs.nemotron_ultra.rdkit_chemistry:grade_rdkit_chemistry",
    "args": ["answer", "contract"],
    "answer_extractor": ANSWER_EXTRACTOR,
}
TOOL_ACTION_SCORER = {
    "function": "skyrl_gym.envs.nemotron_ultra.tool_call:grade_expected_action",
    "args": ["contract.expected_action", "terminal_message"],
    "answer_extractor": ANSWER_EXTRACTOR,
    "input_format": "event",
}

TOOL_ACTION_AGENTS = (
    "single_step_tool_use_with_argument_comparison_agent",
    "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
)
RDKIT_PROPERTIES = frozenset({"count", "bool", "presence", "fragment"})
STRUCTURED_SCHEMA_TYPES = frozenset({"json", "yaml", "toml", "xml", "csv"})
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
    scorer: Mapping[str, Any],
    image: Image,
    *,
    answer_type: AnswerType = AnswerType.TEXT,
    timeout: float = SCORER_TIMEOUT,
    state_path: str | None = None,
    env: Mapping[str, str] | None = None,
) -> NormalizedTask:
    package = source_scorer_package(
        invocation=scorer,
        config={"contract": request.contract},
        environment=image.requirements(),
        timeout=timeout,
        state_path=state_path,
        env=env,
    )
    return blend_task(row, request, package, answer_type)


def _text_scored(
    row: RawRow, agents: tuple[str, ...], scorer: Mapping[str, Any], image: Image
) -> NormalizedTask | ImportRejection:
    request = text_request(row.data, agents)
    if isinstance(request, ImportRejection):
        return request
    return _scored(row, request, scorer, image)


def convert_calendar(row: RawRow) -> NormalizedTask | ImportRejection:
    return _text_scored(row, ("calendar_simple_agent",), CALENDAR_SCORER, NEMOTRON_ULTRA_IMAGE)


def convert_format(row: RawRow) -> NormalizedTask | ImportRejection:
    agents = ("citation_format_simple_agent", "freeform_formatting_simple_agent")
    return _text_scored(row, agents, FORMAT_SCORER, NEMOTRON_ULTRA_IMAGE)


def convert_instruction_following(row: RawRow) -> NormalizedTask | ImportRejection:
    agents = ("instruction_following_simple_agent",)
    return _text_scored(row, agents, INSTRUCTION_FOLLOWING_SCORER, NEMOTRON_ULTRA_IMAGE)


def convert_mcqa(row: RawRow) -> NormalizedTask | ImportRejection:
    return _text_scored(row, ("mcqa_simple_agent",), MCQA_SCORER, ULTRA_MCQA_IMAGE)


def convert_code(row: RawRow) -> NormalizedTask | ImportRejection:
    """Score the last fenced program against the row's hidden unit tests, one thread per test."""
    request = text_request(row.data, ("code_gen_simple_agent",))
    if isinstance(request, ImportRejection):
        return request
    try:
        validate_code_cases(row.data["verifier_metadata"]["unit_tests"])
    except (KeyError, TypeError, ValueError) as error:
        return source_defect("invalid_code_tests", str(error))
    return _scored(
        row,
        request,
        CODE_SCORER,
        NEMOTRON_ULTRA_IMAGE,
        timeout=CODE_SCORER_TIMEOUT,
        state_path=STATE_PATH,
        env=THREAD_ENVIRONMENT,
    )


def convert_structured_output(row: RawRow) -> NormalizedTask | ImportRejection:
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
    scorer = {**STRUCTURED_OUTPUT_SCORER, "input_format": "event" if mode == "tool_call" else "text"}
    answer_type = AnswerType.NATIVE_ACTION if mode == "tool_call" else AnswerType.TEXT
    return _scored(row, request, scorer, NEMOTRON_ULTRA_IMAGE, answer_type=answer_type)


def convert_rdkit(row: RawRow) -> NormalizedTask | ImportRejection:
    """Compare the rounded wrapped answer with the stored molecular property target."""
    request = agent_request(row.data, ("rdkit_chemistry_agent",))
    if isinstance(request, ImportRejection):
        return request
    if row.data["property_type"] not in RDKIT_PROPERTIES:
        return unsupported("unsupported_chemistry_property", str(row.data["property_type"]))
    if not math.isfinite(float(row.data["expected_answer"])):
        return source_defect("nonfinite_chemistry_target", "The rounded comparator requires a finite target")
    return _scored(row, request, RDKIT_SCORER, EXECUTABLE_MATH_IMAGE)


def _tool_action(row: RawRow, image: Image) -> NormalizedTask | ImportRejection:
    request = agent_request(row.data, TOOL_ACTION_AGENTS)
    if isinstance(request, ImportRejection):
        return request
    kind = row.data["expected_action"]["type"]
    if kind not in {"message", "function_call"}:
        return unsupported("unsupported_action_type", str(kind))
    return _scored(row, request, TOOL_ACTION_SCORER, image, answer_type=AnswerType.NATIVE_ACTION)


def convert_toolcall_schema(row: RawRow) -> NormalizedTask | ImportRejection:
    """Compare one predicted action with the expected call, without executing the tool."""
    return _tool_action(row, EXECUTABLE_MATH_IMAGE)


def convert_next_action(row: RawRow) -> NormalizedTask | ImportRejection:
    """Compare a predicted SWE agent action with the expected call, without executing the tool."""
    return _tool_action(row, NEMOTRON_ULTRA_IMAGE)


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


def convert_ungraded(row: RawRow) -> NormalizedTask | ImportRejection:
    """A conversation component whose agent cannot run here."""
    request = blend_request(row.data)
    if isinstance(request, ImportRejection):
        return request
    return _ungraded(row, request, _agent_provider(request) if request.tools else {})


def convert_ungraded_agent(row: RawRow) -> NormalizedTask | ImportRejection:
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


def convert_math(row: RawRow) -> NormalizedTask | ImportRejection:
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


MCQA_CONTROLS = Controls(golden=mcqa_golden)
CODE_CONTROLS = Controls(golden=code_golden, negative=code_negative, memory_mb=CODE_GRADER_MEMORY_MB)
RDKIT_CONTROLS = Controls(golden=rdkit_golden, negative=rdkit_negative)
TOOL_ACTION_CONTROLS = Controls(golden=tool_action_golden, negative=tool_action_negative)
