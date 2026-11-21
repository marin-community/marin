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

# sha256 of the reference d8c26e50ba rendered live from TaskSpec and CONTROL: what recorded author requests embed.
V1_DIGEST = "a91b393984f9d4d9887ea534a22c35a3252bebd76244b8ad51a1c335eda87a06"
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


def test_author_is_shown_the_v1_reference_bytes() -> None:
    reference = author_reference()
    assert hashlib.sha256(reference.encode()).hexdigest() == V1_DIGEST
    assert reference in system_prompt(load_policy(DEFAULTS_DIR))


def test_a_reworded_field_description_leaves_the_author_prompt_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    before = system_prompt(load_policy(DEFAULTS_DIR))
    reworded = TaskSpec.model_json_schema()
    reworded["$defs"]["JsonAnswer"]["description"] = "The answer field, reworded."
    monkeypatch.setattr(TaskSpec, "model_json_schema", lambda *_args, **_kwargs: reworded)
    assert system_prompt(load_policy(DEFAULTS_DIR)) == before


def test_pinned_reference_has_the_installed_schema_shapes() -> None:
    installed = json.loads(json.dumps({"TaskSpec": TaskSpec.model_json_schema(), "Control": CONTROL.json_schema()}))
    found = differences(shape(json.loads(author_reference())), shape(installed))
    assert not found, (
        f"The installed TaskSpec/Control shapes no longer match the pinned {AUTHOR_REFERENCE.name}, so the author "
        "would be shown a schema its submissions are not validated against. Do not edit the pinned file: recorded "
        "author requests replay against its bytes. Add a new reference version and select it deliberately.\n"
        + "\n".join(found[:20])
    )
