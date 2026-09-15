# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tasktrove_verify.spec import (
    Compare,
    Constraint,
    ExactSpec,
    IfevalSpec,
    McqSpec,
    StdioSpec,
    parse_spec,
    render_spec,
)


def test_round_trip_every_field_kind():
    specs = [
        McqSpec(expected="C", options=5),
        ExactSpec(expected=("a", "b"), ordered=False),
        IfevalSpec(constraints=(Constraint("last_word:last_word_answer", {"last_word": "contest"}),)),
        StdioSpec(command="python3 /app/main.py", compare=Compare.FLOAT, special_judge="judge.py", min_cases=3),
    ]
    for spec in specs:
        assert parse_spec(render_spec(spec)) == spec


def test_single_string_becomes_tuple_and_none_is_omitted():
    spec = parse_spec('mode = "exact"\nexpected = "42"\n')
    assert spec == ExactSpec(expected=("42",))
    assert "special_judge" not in render_spec(StdioSpec(command="./a.out"))


@pytest.mark.parametrize(
    "text, message",
    [
        ('expected = "C"\n', "no mode"),
        ('mode = "mcq"\n', r"requires \['expected'\]"),
        ('mode = "mcq"\nexpected = "C"\nbogus = 1\n', r"does not accept \['bogus'\]"),
        ('mode = "mcq"\nexpected = 3\n', "expects str"),
        ('mode = "nope"\n', "nope"),
    ],
)
def test_malformed_specs_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        parse_spec(text)
