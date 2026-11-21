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


def shape(node: object) -> object:
    """``node`` without wording keywords; keys under ``properties`` and ``$defs`` are names and stay."""
    if isinstance(node, list):
        return [shape(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        key: {name: shape(value) for name, value in child.items()} if key in ("properties", "$defs") else shape(child)
        for key, child in node.items()
        if key not in WORDING
    }


def differences(pinned: object, installed: object, path: str = "$") -> list[str]:
    if isinstance(pinned, dict) and isinstance(installed, dict):
        keys = sorted(pinned.keys() | installed.keys())
        return [line for key in keys for line in differences(pinned.get(key), installed.get(key), f"{path}/{key}")]
    if pinned == installed:
        return []
    return [f"{path}: pinned {json.dumps(pinned)[:160]}, installed {json.dumps(installed)[:160]}"]


def policy(decision_model: bool) -> Policy:
    return load_policy(DEFAULTS_DIR).overlaid({"decision_model.model": "decision-model" if decision_model else ""})


def installed_shapes(decision_model: bool) -> object:
    installed = json.loads(json.dumps({"TaskSpec": TaskSpec.model_json_schema(), "Control": CONTROL.json_schema()}))
    if not decision_model:  # the author is offered this kind only when a decision model is configured
        grader = installed["TaskSpec"]["properties"]["grader"]
        del installed["TaskSpec"]["$defs"]["DecisionModelGrader"], grader["discriminator"]["mapping"]["decision_model"]
        grader["oneOf"].remove({"$ref": "#/$defs/DecisionModelGrader"})
    return shape(installed)


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
    found = differences(shape(json.loads(author_reference(decision_model))), installed_shapes(decision_model))
    assert not found, (
        f"The installed TaskSpec/Control shapes no longer match the pinned {AUTHOR_REFERENCE.name} (with "
        f"{DECISION_MODEL_GRADER.name} when a decision model is configured: {decision_model}), so the author would be "
        "shown a schema its submissions are not validated against. Do not edit the pinned files: recorded author "
        "requests replay against their bytes. Add a new reference version and select it deliberately.\n"
        + "\n".join(found[:20])
    )
