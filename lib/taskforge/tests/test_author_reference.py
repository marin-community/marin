# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import re
from typing import Any

import pytest
from taskcompendium.models import TaskSpec

from taskforge.model.controls import CONTROL
from taskforge.policy import Policy, load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.stages.author import AUTHOR_REFERENCE, DECISION_MODEL_GRADER, author_reference, system_prompt

# sha256 of the reference d8c26e50ba rendered live from TaskSpec and CONTROL: what recorded author requests embed.
V1_DIGEST = "a91b393984f9d4d9887ea534a22c35a3252bebd76244b8ad51a1c335eda87a06"
# The same with a decision model configured, as bded40edcc rendered it live before the kind was pinned.
DECISION_MODEL_DIGEST = "3c73be30771a048cf3f2f1a559f5c581b9b748f38eea0a6a95cff1d316dc4bf4"
WORDING = frozenset({"description", "title"})
# Keywords whose values are schemas; every other keyword's value is data (default, examples, const, enum, mapping, ...).
SCHEMA = frozenset({"additionalProperties", "contains", "else", "if", "items", "not", "propertyNames", "then"})
SCHEMA_LISTS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
SCHEMA_MAPS = frozenset({"$defs", "dependentSchemas", "patternProperties", "properties"})


class Absent:
    """A keyword one side lacks, distinct from an explicit null."""


def shape(schema: object) -> object:
    """The schema node without its wording; data values stay verbatim."""
    if not isinstance(schema, dict):
        return schema
    shaped: dict[str, object] = {}
    for key, value in schema.items():
        if key in SCHEMA_MAPS:
            value = {name: shape(child) for name, child in value.items()}
        elif key in SCHEMA_LISTS or (key in SCHEMA and isinstance(value, list)):
            value = [shape(child) for child in value]
        elif key in SCHEMA:
            value = shape(value)
        elif key in WORDING:
            continue
        shaped[key] = value
    return shaped


def shapes(reference: dict[str, object]) -> dict[str, object]:
    return {name: shape(schema) for name, schema in reference.items()}


def as_json(value: object) -> str:
    return "absent" if isinstance(value, Absent) else json.dumps(value, sort_keys=True)


def differences(pinned: object, installed: object, path: str = "$") -> list[str]:
    """Paths where the two differ, telling absence from null and booleans, integers and floats apart."""
    if isinstance(pinned, dict) and isinstance(installed, dict):
        return [
            line
            for key in sorted(pinned.keys() | installed.keys())
            for line in differences(pinned.get(key, Absent()), installed.get(key, Absent()), f"{path}/{key}")
        ]
    if isinstance(pinned, list) and isinstance(installed, list) and len(pinned) == len(installed):
        return [
            line
            for i, pair in enumerate(zip(pinned, installed, strict=True))
            for line in differences(*pair, f"{path}[{i}]")
        ]
    if as_json(pinned) == as_json(installed):
        return []
    return [f"{path}: pinned {as_json(pinned)[:160]}, installed {as_json(installed)[:160]}"]


def node(reference: Any, path: str) -> Any:
    for step in path.split("/"):
        reference = reference[int(step)] if isinstance(reference, list) else reference[step]
    return reference


MODE = "Control/$defs/TaskResource/properties/mode/anyOf/0"
DELETE = Absent()
# Each changes what the schema accepts and must be reported at its path; none is wording.
MUTATIONS = [
    ("TaskSpec/properties/grader", "const", None),
    ("Control/$defs/RewardBand/properties/reward_min", "default", DELETE),
    ("TaskSpec/$defs/JudgeSpec/properties/exact_gate", "default", 1),
    ("TaskSpec/$defs/JudgeSpec/properties/request_timeout", "default", 120),
    ("TaskSpec/$defs/ScriptGrader/properties/reward/default", "description", "x"),
    ("TaskSpec/properties/grader/discriminator/mapping", "title", "#/$defs/NoGrader"),
    ("TaskSpec/$defs/JudgeSpec/properties/exact_gate", "examples", [{"title": "x"}]),
    ("Control/$defs/ControlKind", "enum", ["positive", "negative", "malformed", "other"]),
    (MODE, "pattern", "^[0-7]{3}$"),
    (MODE, "format", "regex"),
    ("Control", "required", ["candidate", "expected"]),
    ("Control/$defs/Candidate/properties", "extra", {"type": "string"}),
    ("Control/$defs/Candidate/properties", "reply", DELETE),
]


@pytest.mark.parametrize("where, key, value", MUTATIONS, ids=[f"{where}/{key}" for where, key, _ in MUTATIONS])
def test_the_structural_check_reports_a_shape_change_at_its_path(where: str, key: str, value: object) -> None:
    pinned = json.loads(author_reference(False))
    changed = copy.deepcopy(pinned)
    target = node(changed, where)
    if isinstance(value, Absent):
        del target[key]
    else:
        target[key] = value
    reported = [line.split(": ", 1)[0] for line in differences(shapes(pinned), shapes(changed))]
    assert reported == ["$/" + re.sub(r"/(\d+)", r"[\1]", f"{where}/{key}")]


def test_the_structural_check_ignores_schema_wording() -> None:
    pinned = json.loads(author_reference(False))
    changed = copy.deepcopy(pinned)
    node(changed, "TaskSpec/$defs/JsonAnswer")["description"] = "Reworded."
    node(changed, "TaskSpec/properties/grader")["title"] = "Reworded"
    node(changed, "Control/$defs/RewardBand/properties/reward_min")["title"] = "Reworded"
    assert differences(shapes(pinned), shapes(changed)) == []


def policy(decision_model: bool) -> Policy:
    return load_policy(DEFAULTS_DIR).overlaid({"decision_model.model": "decision-model" if decision_model else ""})


def installed_shapes(decision_model: bool) -> dict[str, object]:
    installed = json.loads(json.dumps({"TaskSpec": TaskSpec.model_json_schema(), "Control": CONTROL.json_schema()}))
    if not decision_model:  # the author is offered this kind only when a decision model is configured
        grader = installed["TaskSpec"]["properties"]["grader"]
        del installed["TaskSpec"]["$defs"]["DecisionModelGrader"], grader["discriminator"]["mapping"]["decision_model"]
        grader["oneOf"].remove({"$ref": "#/$defs/DecisionModelGrader"})
    return shapes(installed)


@pytest.mark.parametrize("decision_model, digest", [(False, V1_DIGEST), (True, DECISION_MODEL_DIGEST)])
def test_author_is_shown_the_pinned_reference_bytes(decision_model: bool, digest: str) -> None:
    reference = author_reference(decision_model)
    assert hashlib.sha256(reference.encode()).hexdigest() == digest
    assert reference in system_prompt(policy(decision_model))


@pytest.mark.parametrize("decision_model", [False, True])
def test_a_reworded_field_description_leaves_the_author_prompt_unchanged(
    monkeypatch: pytest.MonkeyPatch, decision_model: bool
) -> None:
    before = system_prompt(policy(decision_model))
    reworded = TaskSpec.model_json_schema()
    reworded["$defs"]["JsonAnswer"]["description"] = "The answer field, reworded."
    monkeypatch.setattr(TaskSpec, "model_json_schema", lambda *_args, **_kwargs: reworded)
    assert system_prompt(policy(decision_model)) == before


@pytest.mark.parametrize("decision_model", [False, True])
def test_pinned_reference_has_the_installed_schema_shapes(decision_model: bool) -> None:
    found = differences(shapes(json.loads(author_reference(decision_model))), installed_shapes(decision_model))
    assert not found, (
        f"The installed TaskSpec/Control shapes no longer match the pinned {AUTHOR_REFERENCE.name} (with "
        f"{DECISION_MODEL_GRADER.name} when a decision model is configured: {decision_model}), so the author would be "
        "shown a schema its submissions are not validated against. Do not edit the pinned files: recorded author "
        "requests replay against their bytes. Add a new reference version and select it deliberately.\n"
        + "\n".join(found[:20])
    )
