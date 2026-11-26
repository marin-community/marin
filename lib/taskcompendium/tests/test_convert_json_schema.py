# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskcompendium.convert.json_schema import required_object_conflicts

CLOSED = {"type": "object", "additionalProperties": False, "properties": {"name": {"type": "string"}}}


@pytest.mark.parametrize(
    "schema,conflicts",
    [
        ({**CLOSED, "required": ["name"]}, []),
        ({**CLOSED, "required": ["name", "id"]}, ["$.id: required but forbidden by additionalProperties=false"]),
        ({**CLOSED, "required": ["id"], "patternProperties": {"^i": {"type": "string"}}}, []),
        ({"type": "object", "required": ["id"]}, []),
        (
            {"type": "object", "properties": {"owner": {**CLOSED, "required": ["email"]}}, "required": ["owner"]},
            ["$.owner.email: required but forbidden by additionalProperties=false"],
        ),
        # An optional child can be omitted, so its own conflicts do not make the schema unsatisfiable.
        ({"type": "object", "properties": {"owner": {**CLOSED, "required": ["email"]}}}, []),
    ],
)
def test_required_object_conflicts_finds_mandatory_properties_the_schema_forbids(schema, conflicts):
    assert required_object_conflicts(schema) == conflicts
