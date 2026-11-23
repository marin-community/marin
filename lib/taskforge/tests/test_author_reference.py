# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

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
    pinned = json.dumps(shapes(json.loads(author_reference(decision_model))), sort_keys=True, indent=2)
    installed = json.dumps(installed_shapes(decision_model), sort_keys=True, indent=2)
    assert pinned == installed, (
        f"The installed TaskSpec/Control shapes differ from {AUTHOR_REFERENCE.name} (with "
        f"{DECISION_MODEL_GRADER.name} when a decision model is configured: {decision_model}). "
        "Add a new reference version and select it deliberately."
    )
