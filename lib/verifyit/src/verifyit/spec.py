# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The verifier contract: ``tests/verifier.toml`` is one flat table, ``mode`` plus that mode's fields.

Predicted-action call arguments use JSON strings in TOML to preserve JSON null and nested values.

Every mode is a frozen dataclass here. ``parse_spec`` builds one from TOML text and rejects unknown
or missing fields; ``render_spec`` writes it back. Paths in a spec (``schema``, ``cases``,
``special_judge``, ``restore``, ``path``) are relative to the directory holding ``verifier.toml``.
"""

import dataclasses
import json
import tomllib
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any

import tomlkit

DEFAULT_REWARD_KEY = "reward"
DEFAULT_OUTPUT = "/app/answer.txt"
DEFAULT_WORKSPACE = "/app"


class Mode(StrEnum):
    PREDICTED_ACTION = "predicted_action"
    MCQ = "mcq"
    MATH = "math"
    NUMERIC = "numeric"
    EXACT = "exact"
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


@dataclass(frozen=True)
class FunctionCall:
    """A function name and decoded JSON argument object, without a harness call ID."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PredictedActionSpec:
    expected_calls: tuple[FunctionCall, ...]
    numeric_tolerance: float | None = None
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)


@dataclass(frozen=True)
class McqSpec:
    expected: str
    options: int = 4
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)


@dataclass(frozen=True)
class MathSpec:
    expected: str
    math_type: MathType = MathType.SCALAR
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)
    profile: MathProfile = MathProfile.ANCHORED
    allow_additive_constant: bool = False


@dataclass(frozen=True)
class NumericSpec:
    expected: float
    tolerance_abs: float = 1e-6
    tolerance_rel: float = 1e-6
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
    must_pass: tuple[str, ...] = ()
    must_not_break: tuple[str, ...] = ()
    setup: str = ""
    """Shell command run in the workspace after restore and before the tests, with
    ``VERIFYIT_TESTS_DIR`` and ``VERIFYIT_WORKSPACE`` in its environment."""
    restore: tuple[str, ...] = ()
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
RUBRICS = frozenset({RUBRIC_REFERENCE, RUBRIC_CHECKLIST, RUBRIC_LABELS})


@dataclass(frozen=True)
class JudgeSpec:
    """An LLM judge over the answer file.

    Rubric ``reference`` asks whether the answer matches any of ``references``, after an exact
    gate that answers verbatim matches without a model call. Rubric ``checklist`` asks one yes/no
    question per entry of ``criteria`` and scores the fraction answered yes. ``context`` names a
    file under the tests directory (a conversation transcript, say) shown to the judge alongside the
    answer. ``constraints`` are IFEval checks that must all pass before the judge is consulted.

    Every rubric sends ``max_completion_tokens`` and, when set, ``reasoning_effort`` with each
    request. A reply truncated at that budget is requested again once with
    ``incomplete_retry_tokens`` when it is nonzero; otherwise the grade is an infrastructure error.
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
    output: str = DEFAULT_OUTPUT
    empty_output: EmptyOutputPolicy = field(default=EmptyOutputPolicy.ZERO, kw_only=True)
    system_prompt: str = ""
    prompt_template: str = ""
    label_scores: dict[str, float] = field(default_factory=dict)
    exact_gate_answers: tuple[str, ...] = ()
    exact_gate_label: str = ""
    strip_reasoning_blocks: bool = False
    max_completion_tokens: int = 8192
    incomplete_retry_tokens: int = 0
    reasoning_effort: str = ""
    label_scan: str = "literal"
    label_case: str = "sensitive"
    api: str = "chat_completions"


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
    PredictedActionSpec
    | McqSpec
    | MathSpec
    | NumericSpec
    | ExactSpec
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

SPEC_TYPES: dict[Mode, type] = {
    Mode.PREDICTED_ACTION: PredictedActionSpec,
    Mode.MCQ: McqSpec,
    Mode.MATH: MathSpec,
    Mode.NUMERIC: NumericSpec,
    Mode.EXACT: ExactSpec,
    Mode.JSON_SCHEMA: JsonSchemaSpec,
    Mode.XML_ELEMENTS: XmlElementsSpec,
    Mode.CSV_COLUMNS: CsvColumnsSpec,
    Mode.IFEVAL: IfevalSpec,
    Mode.REASONING_GYM: ReasoningGymSpec,
    Mode.STDIO: StdioSpec,
    Mode.PYTEST: PytestSpec,
    Mode.JUNIT: JunitSpec,
    Mode.GOTEST: GotestSpec,
    Mode.JUDGE: JudgeSpec,
    Mode.SCRIPT: ScriptSpec,
}
MODES: dict[type, Mode] = {t: m for m, t in SPEC_TYPES.items()}


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
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
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
    if isinstance(annotation, type) and not isinstance(value, annotation):
        raise ValueError(f"field {name!r} expects {annotation.__name__}, got {type(value).__name__}")
    return value


def spec_from_table(table: dict[str, Any]) -> Spec:
    """Build the typed spec for ``table["mode"]``; unknown or missing fields raise ``ValueError``."""
    if "mode" not in table:
        raise ValueError("verifier spec has no mode")
    mode = Mode(table["mode"])
    spec_type = SPEC_TYPES[mode]
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
                call["arguments"] = json.loads(call["arguments"])
    return spec_from_table(table)


def spec_to_table(spec: Spec) -> dict[str, Any]:
    """The flat TOML table for a spec: ``mode`` first, then every field, ``None`` fields omitted."""
    table: dict[str, Any] = {"mode": mode_of(spec).value}
    for f in fields(spec):
        value = getattr(spec, f.name)
        if value is None:
            continue
        if isinstance(value, tuple):
            value = [dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v for v in value]
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
    return tomlkit.dumps(table)
