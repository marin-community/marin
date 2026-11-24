# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The verifier contract: ``tests/verifier.toml`` contains ``mode`` plus that mode's fields.

Structured references and predicted-action arguments use JSON strings in TOML to preserve JSON null and nested values.

Every mode is a frozen dataclass here. ``parse_spec`` builds one from TOML text and rejects unknown
or missing fields; ``render_spec`` writes it back. Paths in a spec (``schema``, ``cases``,
``special_judge``, ``restore``, ``path``) are relative to the directory holding ``verifier.toml``.
"""

import dataclasses
import json
from dataclasses import dataclass, field, fields
from typing import Any, Literal, get_origin

from verifyit.closure_scoring import ClosureFormat
from verifyit.compat import StrEnum, tomllib
from verifyit.json_comparison import JsonValue, NumericTypePolicy
from verifyit.json_objects import unique_object

DEFAULT_REWARD_KEY = "reward"
DEFAULT_OUTPUT = "/app/answer.txt"
DEFAULT_WORKSPACE = "/app"


class Mode(StrEnum):
    UNSOLVABLE = "unsolvable"
    STRUCTURED_EXACT = "structured_exact"
    PREDICTED_ACTION = "predicted_action"
    MCQ = "mcq"
    MATH = "math"
    NUMERIC = "numeric"
    EXACT = "exact"
    STRUCTURE_CLOSURE = "structure_closure"
    REASONING_MARKUP = "reasoning_markup"
    JSON_SCHEMA = "json-schema"
    XML_ELEMENTS = "xml-elements"
    CSV_COLUMNS = "csv-columns"
    IFEVAL = "ifeval"
    REASONING_GYM = "reasoning-gym"
    STDIO = "stdio"
    PYTEST = "pytest"
    JUNIT = "junit"
    GOTEST = "gotest"
    JUDGE = "judge"
    SCRIPT = "script"


class MathType(StrEnum):
    SCALAR = "scalar"
    INTERVAL = "interval"
    SET = "set"
    TUPLE = "tuple"
    LIST = "list"
    EQUATION = "equation"


class MathProfile(StrEnum):
    ANCHORED = "anchored"
    BOXED = "boxed"
    RAW = "raw"


class SchemaFormat(StrEnum):
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


class Compare(StrEnum):
    EXACT = "exact"
    TOKENS = "tokens"
    FLOAT = "float"
    DECIMAL_LINES = "decimal_lines"


class TestIdMatching(StrEnum):
    EXACT = "exact"
    UNIQUE_PREFIX = "unique_prefix"


class EmptyOutputPolicy(StrEnum):
    """Whether a present empty answer is scored zero or passed to its grader."""

    ZERO = "zero"
    GRADE = "grade"


class McqExtraction(StrEnum):
    """How an MCQ grader finds the chosen option in the output.

    ``ANSWER_LINE`` reads the last ``Answer: X`` line or a bare letter. ``SUPERGPQA_REFERENCE``
    reproduces the SuperGPQA runner's zero-shot extraction, including its option-text fallback.
    """

    ANSWER_LINE = "answer_line"
    SUPERGPQA_REFERENCE = "supergpqa_reference"


@dataclass(frozen=True)
class FunctionCall:
    """A function name and decoded JSON argument object, without a harness call ID."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class UnsolvableSpec:
    """A private task label requiring a validated terminal ejection for credit."""

    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)


@dataclass(frozen=True)
class ReasoningMarkupSpec:
    """Grade decoded reply markers, assuming the prompt opened no span. Length stops excuse open spans."""

    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)


@dataclass(frozen=True)
class PredictedActionSpec:
    expected_calls: tuple[FunctionCall, ...]
    numeric_tolerance: float | None = None
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)


@dataclass(frozen=True)
class StructuredExactSpec:
    expected: JsonValue
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)
    numeric_types: NumericTypePolicy = field(default=NumericTypePolicy.VALUE, kw_only=True)


@dataclass(frozen=True)
class McqSpec:
    expected: str
    options: int = 4
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)
    extraction: McqExtraction = field(default=McqExtraction.ANSWER_LINE, kw_only=True)
    # The text of each option, in letter order; the SuperGPQA extraction maps matched option text to its letter.
    option_texts: tuple[str, ...] = field(default=(), kw_only=True)


@dataclass(frozen=True)
class MathSpec:
    expected: str
    math_type: MathType = MathType.SCALAR
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)
    profile: MathProfile = MathProfile.ANCHORED
    allow_additive_constant: bool = False
    positive_symbols: tuple[str, ...] = field(default=(), kw_only=True)


@dataclass(frozen=True)
class NumericSpec:
    expected: str
    tolerance_abs: float
    tolerance_rel: float
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)


@dataclass(frozen=True)
class ExactSpec:
    expected: tuple[str, ...]
    ignore_case: bool = True
    ignore_whitespace: bool = True
    ordered: bool = True
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)
    strip_outer_whitespace: bool = True
    substring: bool = False


@dataclass(frozen=True)
class StructureClosureSpec:
    """Complete a pinned assistant prefix while preserving its reference content."""

    prefix: str
    expected: str
    format: ClosureFormat
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.GRADE, kw_only=True)


@dataclass(frozen=True)
class JsonSchemaSpec:
    schema: str = "schema.json"
    format: SchemaFormat = SchemaFormat.JSON
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)


@dataclass(frozen=True)
class XmlElementsSpec:
    """Names a well-formed XML answer must carry as element tags or attribute names.

    Every name in ``required`` must appear. ``any_of`` is the weaker alternative for a schema that
    marks no field required: one of its names present is enough. A spec with neither is invalid.
    """

    required: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)


@dataclass(frozen=True)
class CsvColumnsSpec:
    """Column headers a CSV answer must carry, in the same ``required`` / ``any_of`` form."""

    required: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)


@dataclass(frozen=True)
class Constraint:
    name: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IfevalSpec:
    constraints: tuple[Constraint, ...]
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)


@dataclass(frozen=True)
class ReasoningGymSpec:
    dataset: str
    entry: str = "entry.json"
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)
    params: str | None = None


@dataclass(frozen=True)
class StdioSpec:
    command: str
    build: str = ""
    """Shell command run once in the workspace before the cases, e.g. ``g++ -O2 -o main main.cpp``."""
    cases: str = "cases"
    compare: Compare = Compare.TOKENS
    special_judge: str | None = None
    per_case_timeout: float = 10.0
    min_cases: int = 1
    float_tolerance: float = 1e-6
    workspace: str = DEFAULT_WORKSPACE


@dataclass(frozen=True)
class PytestSpec:
    paths: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    candidate_modules: tuple[str, ...] = ()
    """Python modules the task explicitly asks the candidate to implement."""
    must_pass: tuple[str, ...] = ()
    must_not_break: tuple[str, ...] = ()
    setup: str = ""
    """Shell command run in the workspace after restore and before the tests, with
    ``VERIFYIT_TESTS_DIR`` and ``VERIFYIT_WORKSPACE`` in its environment."""
    restore: tuple[str, ...] = ()
    protected_paths_files: tuple[str, ...] = ()
    """Tests-directory manifests of workspace paths restored by setup."""
    python: str = "python3"
    timeout: float = 600.0
    workspace: str = DEFAULT_WORKSPACE
    setup_failure_is_infra: bool = False
    batch_size: int = 0
    id_matching: TestIdMatching = TestIdMatching.EXACT


@dataclass(frozen=True)
class JunitSpec:
    command: str
    report: str = "**/TEST-*.xml"
    must_pass: tuple[str, ...] = ()
    must_not_break: tuple[str, ...] = ()
    setup: str = ""
    """Shell command run in the workspace after restore and before the tests, with
    ``VERIFYIT_TESTS_DIR`` and ``VERIFYIT_WORKSPACE`` in its environment."""
    restore: tuple[str, ...] = ()
    timeout: float = 600.0
    workspace: str = DEFAULT_WORKSPACE


@dataclass(frozen=True)
class GotestSpec:
    packages: tuple[str, ...] = ("./...",)
    args: tuple[str, ...] = ()
    must_pass: tuple[str, ...] = ()
    must_not_break: tuple[str, ...] = ()
    setup: str = ""
    """Shell command run in the workspace after restore and before the tests, with
    ``VERIFYIT_TESTS_DIR`` and ``VERIFYIT_WORKSPACE`` in its environment."""
    restore: tuple[str, ...] = ()
    timeout: float = 600.0
    workspace: str = DEFAULT_WORKSPACE


RUBRIC_REFERENCE = "reference"
RUBRIC_CHECKLIST = "checklist"
RUBRIC_LABELS = "labels"
RUBRIC_FRONTIERSCIENCE = "frontierscience_research"
RUBRIC_ANSWERS = "answers"
RUBRIC_PAIRWISE = "pairwise"
RUBRICS = frozenset(
    {RUBRIC_REFERENCE, RUBRIC_CHECKLIST, RUBRIC_LABELS, RUBRIC_ANSWERS, RUBRIC_PAIRWISE, RUBRIC_FRONTIERSCIENCE}
)


class JudgeCompletionPolicy(StrEnum):
    COMPLETED_TEXT = "completed_text"
    CONTENT = "content"


@dataclass(frozen=True)
class StripReasoningBlocks:
    """Drop reasoning blocks, text before the last closing delimiter, outer whitespace and a final ``<|eot_id|>``.

    An answer with an unclosed opening delimiter becomes empty.
    """

    kind: Literal["strip_reasoning_blocks"] = "strip_reasoning_blocks"


@dataclass(frozen=True)
class StripOuterWhitespace:
    kind: Literal["strip_outer_whitespace"] = "strip_outer_whitespace"


@dataclass(frozen=True)
class ExtractGroup:
    """Replace the answer with the first group of the first ``re.search`` match; leave it unchanged without one."""

    kind: Literal["extract_group"] = field(default="extract_group", kw_only=True)
    pattern: str


CandidateStep = StripReasoningBlocks | StripOuterWhitespace | ExtractGroup
CANDIDATE_STEP_TYPES: dict[str, type[CandidateStep]] = {
    "strip_reasoning_blocks": StripReasoningBlocks,
    "strip_outer_whitespace": StripOuterWhitespace,
    "extract_group": ExtractGroup,
}


@dataclass(frozen=True)
class JudgeSamplingParams:
    """Sampling options for judge requests. A ``None`` member is omitted."""

    temperature: float | None = None


@dataclass(frozen=True)
class JudgeSpec:
    """An LLM judge over the answer file.

    ``candidate_steps`` rewrite the answer, in order, before the constraints, the exact gates and
    the prompt, for every rubric. ``constraints`` are IFEval checks that must all pass before the
    judge is consulted.

    Rubric ``reference`` asks whether the answer matches any of ``references``, after an exact gate
    that scores verbatim matches 1 without a model call. Rubric ``checklist`` asks one yes/no
    question per entry of ``criteria`` and scores the fraction answered yes; ``context`` names a
    file under the tests directory (a conversation transcript, say) shown to the judge alongside
    the answer. Rubric ``labels`` reads a verdict label from the reply, as ``label_scan`` selects,
    and scores it with ``label_scores``. Rubric ``answers`` asks for one JSON verdict per entry of
    ``references`` in a single call and scores 1 when every reference is matched. Its detail
    records the matched fraction as ``partial_credit``.

    Rubric ``pairwise`` compares the answer with the single baseline in ``references`` twice, once
    in each position, as Arena-Hard does. ``prompt_template`` places the two answers with
    ``{answer_a}`` and ``{answer_b}``. The last match of the first ``verdict_patterns`` entry that
    matches the reply names a verdict, ``label_scores`` gives assistant A's score for it, and
    ``label_weights`` repeats it as several games (Arena-Hard counts a decisive verdict three times).

    ``system_role`` is the role of the ``system_prompt`` message, ``system`` or ``developer``.
    ``response_format`` is sent with each Chat Completions label request. An empty ``sampling`` table
    uses the endpoint defaults. The ``answers`` rubric always uses the endpoint sampling defaults.
    """

    references: tuple[str, ...] = ()
    criteria: tuple[str, ...] = ()
    question: str = ""
    context: str = ""
    constraints: tuple[Constraint, ...] = ()
    rubric: str = RUBRIC_REFERENCE
    model: str = ""
    exact_gate: bool = True
    request_timeout: float = 120.0
    transport_attempts: int = field(default=3, kw_only=True)
    transport_budget: float = field(default=180.0, kw_only=True)
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)
    system_prompt: str = ""
    prompt_template: str = ""
    label_scores: dict[str, float] = field(default_factory=dict)
    label_metrics: dict[str, dict[str, float]] = field(default_factory=dict, kw_only=True)
    completion_policy: JudgeCompletionPolicy = field(default=JudgeCompletionPolicy.COMPLETED_TEXT, kw_only=True)
    exact_gate_answers: tuple[str, ...] = ()
    exact_gate_label: str = ""
    strip_reasoning_blocks: bool = False
    candidate_steps: tuple[CandidateStep, ...] = field(default=(), kw_only=True)
    max_completion_tokens: int = 8192
    incomplete_retry_tokens: int = 0
    reasoning_effort: str = ""
    label_scan: str = "literal"
    label_field: str = ""
    label_case: str = "sensitive"
    api: str = "chat_completions"
    context_char_limit: int = field(default=60_000, kw_only=True)
    max_turns: int = field(default=1, kw_only=True)
    parse_attempts: int = field(default=2, kw_only=True)
    chat_template_kwargs: dict[str, str] = field(default_factory=dict, kw_only=True)
    system_role: Literal["system", "developer"] = field(default="system", kw_only=True)
    response_format: dict[str, Any] = field(default_factory=dict, kw_only=True)
    sampling: JudgeSamplingParams = field(default_factory=lambda: JudgeSamplingParams(temperature=0.0), kw_only=True)
    verdict_patterns: tuple[str, ...] = field(default=(), kw_only=True)
    label_weights: dict[str, float] = field(default_factory=dict, kw_only=True)


class JudgeRuntimeSource(StrEnum):
    """Select the environment-configured judge endpoint for remote grading."""

    ENVIRONMENT = "environment"


@dataclass(frozen=True)
class ScriptSpec:
    path: str
    args: tuple[str, ...] = ()
    timeout: float = 600.0
    workspace: str = DEFAULT_WORKSPACE
    reward_key: str = DEFAULT_REWARD_KEY
    verdict_file: str | None = None


Spec = (
    UnsolvableSpec
    | StructuredExactSpec
    | PredictedActionSpec
    | McqSpec
    | MathSpec
    | NumericSpec
    | ExactSpec
    | StructureClosureSpec
    | ReasoningMarkupSpec
    | JsonSchemaSpec
    | XmlElementsSpec
    | CsvColumnsSpec
    | IfevalSpec
    | ReasoningGymSpec
    | StdioSpec
    | PytestSpec
    | JunitSpec
    | GotestSpec
    | JudgeSpec
    | ScriptSpec
)


@dataclass(frozen=True)
class CandidateEntrypoints:
    """Names in the mode module; modules stay lazy to preserve optional dependencies."""

    validate: str
    grade: str


@dataclass(frozen=True)
class ModeDescriptor:
    spec_type: type[Spec]
    module: str
    candidate: CandidateEntrypoints | None = None


MODE_DESCRIPTORS: dict[Mode, ModeDescriptor] = {
    Mode.UNSOLVABLE: ModeDescriptor(
        UnsolvableSpec, "grade_unsolvable", CandidateEntrypoints("validate_unsolvable", "grade_unsolvable_candidate")
    ),
    Mode.STRUCTURED_EXACT: ModeDescriptor(
        StructuredExactSpec,
        "grade_structured_exact",
        CandidateEntrypoints("validate_structured_exact", "grade_structured_exact_candidate"),
    ),
    Mode.PREDICTED_ACTION: ModeDescriptor(
        PredictedActionSpec,
        "grade_predicted_action",
        CandidateEntrypoints("validate_predicted_action", "grade_predicted_action_candidate"),
    ),
    Mode.MCQ: ModeDescriptor(McqSpec, "grade_mcq", CandidateEntrypoints("validate_mcq", "grade_mcq_candidate")),
    Mode.MATH: ModeDescriptor(
        MathSpec, "grade_math", CandidateEntrypoints("validate_math", "grade_math_text_candidate")
    ),
    Mode.NUMERIC: ModeDescriptor(
        NumericSpec, "grade_math", CandidateEntrypoints("validate_numeric", "grade_numeric_text_candidate")
    ),
    Mode.EXACT: ModeDescriptor(
        ExactSpec, "grade_exact", CandidateEntrypoints("validate_exact", "grade_exact_candidate")
    ),
    Mode.STRUCTURE_CLOSURE: ModeDescriptor(
        StructureClosureSpec,
        "grade_structure_closure",
        CandidateEntrypoints("validate_structure_closure", "grade_structure_closure_candidate"),
    ),
    Mode.REASONING_MARKUP: ModeDescriptor(
        ReasoningMarkupSpec,
        "grade_reasoning_markup",
        CandidateEntrypoints("validate_reasoning_markup", "grade_reasoning_markup_candidate"),
    ),
    Mode.JSON_SCHEMA: ModeDescriptor(
        JsonSchemaSpec,
        "grade_json_schema",
        CandidateEntrypoints("validate_json_schema", "grade_json_resource_candidate"),
    ),
    Mode.XML_ELEMENTS: ModeDescriptor(
        XmlElementsSpec, "grade_xml", CandidateEntrypoints("validate_xml_elements", "grade_xml_candidate")
    ),
    Mode.CSV_COLUMNS: ModeDescriptor(
        CsvColumnsSpec, "grade_csv", CandidateEntrypoints("validate_csv_columns", "grade_csv_candidate")
    ),
    Mode.IFEVAL: ModeDescriptor(
        IfevalSpec, "grade_ifeval", CandidateEntrypoints("validate_ifeval", "grade_ifeval_candidate")
    ),
    Mode.REASONING_GYM: ModeDescriptor(ReasoningGymSpec, "grade_reasoning_gym"),
    Mode.STDIO: ModeDescriptor(StdioSpec, "grade_stdio"),
    Mode.PYTEST: ModeDescriptor(PytestSpec, "grade_pytest"),
    Mode.JUNIT: ModeDescriptor(JunitSpec, "grade_junit"),
    Mode.GOTEST: ModeDescriptor(GotestSpec, "grade_gotest"),
    Mode.JUDGE: ModeDescriptor(JudgeSpec, "grade_judge"),
    Mode.SCRIPT: ModeDescriptor(ScriptSpec, "grade_script"),
}
MODES: dict[type[Spec], Mode] = {descriptor.spec_type: mode for mode, descriptor in MODE_DESCRIPTORS.items()}


def mode_of(spec: Spec) -> Mode:
    return MODES[type(spec)]


def _coerce(name: str, annotation: Any, value: Any) -> Any:
    if annotation == tuple[FunctionCall, ...]:
        if not isinstance(value, list):
            raise ValueError(f"field {name!r} expects a list of function calls")
        calls = []
        for call in value:
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise ValueError("function calls require name and arguments")
            calls.append(FunctionCall(name=call["name"], arguments=call["arguments"]))
        return tuple(calls)
    if annotation in (tuple[str, ...],):
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) for v in value):
            raise ValueError(f"field {name!r} expects strings")
        return tuple(value)
    if annotation == tuple[Constraint, ...]:
        return tuple(Constraint(name=c["name"], params=dict(c.get("params", {}))) for c in value)
    if annotation == tuple[CandidateStep, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"field {name!r} expects a list of steps")
        return tuple(_step_from_table(name, step) for step in value)
    if annotation is JudgeSamplingParams:
        if not isinstance(value, dict):
            raise ValueError(f"field {name!r} expects a table")
        declared = {f.name: f for f in fields(JudgeSamplingParams)}
        if unknown := sorted(set(value) - set(declared)):
            raise ValueError(f"field {name!r} does not accept {unknown}")
        return JudgeSamplingParams(**{key: _coerce(key, declared[key].type, member) for key, member in value.items()})
    if annotation == dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(f"field {name!r} expects a table of numbers")
        return {key: _coerced_float(name, item) for key, item in value.items()}
    if annotation == dict[str, dict[str, float]]:
        if not isinstance(value, dict) or any(not isinstance(inner, dict) for inner in value.values()):
            raise ValueError(f"field {name!r} expects a table of number tables")
        return {
            key: {inner_key: _coerced_float(name, item) for inner_key, item in inner.items()}
            for key, inner in value.items()
        }
    # typing.get_origin excludes parameterized generics: on 3.10 they still report as ``type``.
    if get_origin(annotation) is None and isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation(value)
    if annotation is float and type(value) is int:
        try:
            return float(value)
        except OverflowError as error:
            raise ValueError(f"field {name!r} is outside the float range") from error
    if annotation == (str | None):
        return value
    if annotation == (float | None):
        if value is not None and type(value) not in (int, float):
            raise ValueError(f"field {name!r} expects a number or null")
        try:
            return float(value) if value is not None else None
        except OverflowError as error:
            raise ValueError(f"field {name!r} is outside the float range") from error
    if annotation is int and type(value) is not int:
        raise ValueError(f"field {name!r} expects an integer")
    if get_origin(annotation) is None and isinstance(annotation, type) and not isinstance(value, annotation):
        raise ValueError(f"field {name!r} expects {annotation.__name__}, got {type(value).__name__}")
    return value


def _step_from_table(name: str, table: Any) -> CandidateStep:
    if not isinstance(table, dict) or table.get("kind") not in CANDIDATE_STEP_TYPES:
        raise ValueError(f"field {name!r} expects steps of kind {sorted(CANDIDATE_STEP_TYPES)}")
    step_type = CANDIDATE_STEP_TYPES[table["kind"]]
    declared = {f.name for f in fields(step_type)}
    if set(table) != declared or any(not isinstance(value, str) for value in table.values()):
        raise ValueError(f"step {table['kind']!r} takes exactly the string fields {sorted(declared)}")
    return step_type(**table)


def upgrade_judge_table(table: dict[str, Any]) -> dict[str, Any]:
    """Upgrade candidate preprocessing and sampling in previously serialized judges."""
    if "strip_candidate_reasoning_blocks" not in table and "temperature" not in table:
        return table
    table = dict(table)
    if "strip_candidate_reasoning_blocks" in table:
        if "candidate_steps" in table:
            raise ValueError("judge spec sets both strip_candidate_reasoning_blocks and candidate_steps")
        strip = table.pop("strip_candidate_reasoning_blocks")
        if type(strip) is not bool:
            raise ValueError("field 'strip_candidate_reasoning_blocks' expects bool")
        if strip:
            table["candidate_steps"] = [{"kind": "strip_reasoning_blocks"}]
    if "temperature" in table:
        if "sampling" in table:
            raise ValueError("judge spec sets both temperature and sampling")
        temperature = table.pop("temperature")
        table["sampling"] = {} if temperature is None or temperature == "API default" else {"temperature": temperature}
    return table


def _coerced_float(name: str, value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"field {name!r} expects a number")
    try:
        return float(value)
    except OverflowError as error:
        raise ValueError(f"field {name!r} is outside the float range") from error


def spec_from_table(table: dict[str, Any]) -> Spec:
    """Build the typed spec for ``table["mode"]``; unknown or missing fields raise ``ValueError``."""
    if "mode" not in table:
        raise ValueError("verifier spec has no mode")
    mode = Mode(table["mode"])
    if mode is Mode.JUDGE:
        table = upgrade_judge_table(table)
    spec_type = MODE_DESCRIPTORS[mode].spec_type
    declared = {f.name: f for f in fields(spec_type)}
    unknown = sorted(set(table) - set(declared) - {"mode"})
    if unknown:
        raise ValueError(f"mode {mode} does not accept {unknown}")
    values = {name: _coerce(name, f.type, table[name]) for name, f in declared.items() if name in table}
    missing = [
        n
        for n, f in declared.items()
        if n not in values and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    if missing:
        raise ValueError(f"mode {mode} requires {missing}")
    return spec_type(**values)


def parse_spec(text: str) -> Spec:
    table = tomllib.loads(text)
    if table.get("mode") == Mode.PREDICTED_ACTION and isinstance(table.get("expected_calls"), list):
        for call in table.get("expected_calls", []):
            if isinstance(call, dict) and isinstance(call.get("arguments"), str):
                call["arguments"] = json.loads(call["arguments"], object_pairs_hook=unique_object)
    if table.get("mode") == Mode.STRUCTURED_EXACT and "expected" in table:
        if not isinstance(table["expected"], str):
            raise ValueError("Structured reference in TOML must be a JSON string")
        table["expected"] = json.loads(table["expected"], object_pairs_hook=unique_object)
    return spec_from_table(table)


def spec_to_table(spec: Spec) -> dict[str, Any]:
    """The TOML table for a spec: ``mode`` first, then every field except ``None`` at a ``None`` default.

    Judge ``sampling`` is a nested table without its ``None`` members.
    """
    table: dict[str, Any] = {"mode": mode_of(spec).value}
    for f in fields(spec):
        value = getattr(spec, f.name)
        if value is None and f.default is None:
            continue
        if isinstance(value, tuple):
            value = [dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v for v in value]
        elif isinstance(value, JudgeSamplingParams):
            value = {key: member for key, member in dataclasses.asdict(value).items() if member is not None}
        elif isinstance(value, StrEnum):
            value = value.value
        table[f.name] = value
    return table


def render_spec(spec: Spec) -> str:
    table = spec_to_table(spec)
    if isinstance(spec, PredictedActionSpec):
        table["expected_calls"] = [
            {"name": call.name, "arguments": json.dumps(call.arguments, allow_nan=False)} for call in spec.expected_calls
        ]
    if isinstance(spec, StructuredExactSpec):
        table["expected"] = json.dumps(spec.expected, allow_nan=False)
    # Rendering is host-side staging; the grader machine never imports this, so the
    # writer dependency stays off the machine's interpreter.
    import tomli_w  # noqa: PLC0415

    return tomli_w.dumps(table)
