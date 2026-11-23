# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from taskcompendium.models import TaskSpec

from taskforge.model.controls import CONTROL
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.stages.author import AUTHOR_REFERENCE, author_reference, system_prompt

# sha256 of the v2 reference as generated from TaskSpec and CONTROL: what recorded author requests embed.
V2_DIGEST = "28bef7104d48e9e3752479bc0c98a9c87dae1af4a1704e0e22994e6d15b29a40"
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


def test_author_is_shown_the_v2_reference_bytes() -> None:
    reference = author_reference()
    assert hashlib.sha256(reference.encode()).hexdigest() == V2_DIGEST
    assert reference in system_prompt(load_policy(DEFAULTS_DIR))


def test_a_reworded_field_description_leaves_the_author_prompt_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    before = system_prompt(load_policy(DEFAULTS_DIR))
    reworded = TaskSpec.model_json_schema()
    reworded["$defs"]["JsonAnswer"]["description"] = "The answer field, reworded."
    monkeypatch.setattr(TaskSpec, "model_json_schema", lambda *_args, **_kwargs: reworded)
    assert system_prompt(load_policy(DEFAULTS_DIR)) == before


def test_pinned_reference_has_the_installed_schema_shapes() -> None:
    pinned = json.dumps(shapes(json.loads(author_reference())), sort_keys=True, indent=2)
    installed = json.dumps(
        shapes({"TaskSpec": TaskSpec.model_json_schema(), "Control": CONTROL.json_schema()}), sort_keys=True, indent=2
    )
    assert pinned == installed, (
        f"The installed TaskSpec/Control shapes differ from {AUTHOR_REFERENCE.name}. "
        "Add a new reference version and select it deliberately."
    )
