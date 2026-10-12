# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.spec import (
    Compare,
    Constraint,
    EmptyOutputPolicy,
    ExactSpec,
    FunctionCall,
    IfevalSpec,
    MathProfile,
    MathSpec,
    MathType,
    McqSpec,
    NumericSpec,
    PredictedActionSpec,
    PytestSpec,
    ReasoningGymSpec,
    StdioSpec,
    StructuredExactSpec,
    parse_spec,
    render_spec,
    spec_from_table,
)
from verifyit.spec import (
    TestIdMatching as IdMatching,
)


def test_round_trip_every_field_kind():
    specs = [
        McqSpec(expected="C", options=5),
        StructuredExactSpec(expected={"values": [None, True, 1, 1.0, {"text": "value"}]}),
        StructuredExactSpec(expected=None, empty_output=EmptyOutputPolicy.GRADE),
        PredictedActionSpec(
            expected_calls=(FunctionCall("lookup", {"values": [None, True, 1, 1.0, {"text": "value"}]}),),
            numeric_tolerance=0.01,
        ),
        MathSpec(expected="(1, 2)", math_type=MathType.TUPLE),
        MathSpec(expected="0.5", profile=MathProfile.BOXED),
        NumericSpec(expected="42", tolerance_abs=0.1, tolerance_rel=0.01),
        ExactSpec(expected=("a", "b"), ordered=False, strip_outer_whitespace=False),
        IfevalSpec(constraints=(Constraint("last_word:last_word_answer", {"last_word": "contest"}),)),
        StdioSpec(command="python3 /app/main.py", compare=Compare.FLOAT, special_judge="judge.py", min_cases=3),
        PytestSpec(setup="uv init", setup_failure_is_infra=True),
        PytestSpec(batch_size=40, id_matching=IdMatching.UNIQUE_PREFIX),
    ]
    for spec in specs:
        assert parse_spec(render_spec(spec)) == spec


def test_single_string_becomes_tuple_and_none_is_omitted():
    spec = parse_spec('mode = "exact"\nexpected = "42"\n')
    assert spec == ExactSpec(expected=("42",))
    assert "special_judge" not in render_spec(StdioSpec(command="./a.out"))


def test_large_repository_test_id_lists_round_trip():
    # SWE-smith repositories can carry thousands of parametrized PASS_TO_PASS IDs.
    ids = tuple(f'tests/test_schema.py::test_schema[case-{i}-"quoted"-\\-\n-λ]' for i in range(10_000))
    spec = PytestSpec(must_pass=ids[:2], must_not_break=ids)
    assert parse_spec(render_spec(spec)) == spec


@pytest.mark.parametrize(
    "text, message",
    [
        ('expected = "C"\n', "no mode"),
        ('mode = "mcq"\n', r"requires \['expected'\]"),
        ('mode = "mcq"\nexpected = "C"\nbogus = 1\n', r"does not accept \['bogus'\]"),
        ('mode = "mcq"\nexpected = 3\n', "expects str"),
        ('mode = "nope"\n', "nope"),
        ('mode = "pytest"\nsetup_failure_is_infra = "true"\n', "expects bool"),
        ('mode = "exact"\nexpected = [1]\n', "expects strings"),
        ('mode = "mcq"\nexpected = "A"\noptions = true\n', "expects an integer"),
        ('mode = "predicted_action"\nexpected_calls = 1\n', "expects a list of function calls"),
    ],
)
def test_malformed_specs_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        parse_spec(text)


def test_exact_substring_roundtrip_and_default_compatibility():
    spec = ExactSpec(("Paris",), substring=True)
    assert parse_spec(render_spec(spec)) == spec
    assert parse_spec('mode = "exact"\nexpected = "Paris"\n').substring is False


def test_exact_substring_rejects_non_boolean_toml():
    with pytest.raises(ValueError, match="expects bool"):
        parse_spec('mode = "exact"\nexpected = "Paris"\nsubstring = 1\n')


def test_reasoning_gym_params_roundtrip_and_legacy_default():
    configured = ReasoningGymSpec(dataset="decimal_arithmetic", params="params.json")
    assert parse_spec(render_spec(configured)) == configured
    legacy = parse_spec('mode = "reasoning-gym"\ndataset = "decimal_arithmetic"\n')
    assert legacy.params is None
    assert "params" not in render_spec(legacy)


@pytest.mark.parametrize(
    "mode,fields",
    [
        ("predicted_action", {"expected_calls": [{"name": "lookup", "arguments": {"id": 1}}]}),
        ("mcq", {"expected": "A"}),
        ("math", {"expected": "1"}),
        ("numeric", {"expected": "1", "tolerance_abs": 0.0, "tolerance_rel": 0.0}),
        ("exact", {"expected": [""]}),
        ("json-schema", {}),
        ("xml-elements", {"required": ["answer"]}),
        ("csv-columns", {"required": ["answer"]}),
        ("ifeval", {"constraints": [{"name": "punctuation:no_comma"}]}),
        ("reasoning-gym", {"dataset": "decimal_arithmetic"}),
        ("judge", {"references": ["reference"]}),
    ],
)
def test_output_modes_materialize_and_render_explicit_empty_policy(mode, fields):
    default = spec_from_table({"mode": mode, **fields})
    assert default.empty_output is EmptyOutputPolicy.ZERO
    assert 'empty_output = "zero"' in render_spec(default)
    configured = spec_from_table({"mode": mode, **fields, "empty_output": "grade"})
    assert configured.empty_output is EmptyOutputPolicy.GRADE
    assert parse_spec(render_spec(configured)) == configured
    with pytest.raises(ValueError):
        spec_from_table({"mode": mode, **fields, "empty_output": "reward_half"})


@pytest.mark.parametrize(
    "text",
    [
        "mode = 'structured_exact'\nexpected = '{\"value\":1,\"value\":2}'\n",
        "mode = 'predicted_action'\n[[expected_calls]]\nname = 'submit'\narguments = '{\"value\":1,\"value\":2}'\n",
    ],
)
def test_embedded_private_json_rejects_duplicate_keys(text):
    with pytest.raises(ValueError):
        parse_spec(text)
