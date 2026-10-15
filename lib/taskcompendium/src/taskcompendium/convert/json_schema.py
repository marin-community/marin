# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static checks on JSON Schemas that a schema-valid answer must satisfy."""

import re
from typing import Any


def required_object_conflicts(schema: dict[str, Any], path: str = "$") -> list[str]:
    """Mandatory object properties that their own schema forbids, which no instance can satisfy.

    A property is forbidden when the object sets ``additionalProperties: false`` and neither
    ``properties`` nor a ``patternProperties`` pattern admits it. Required child objects are
    checked recursively; optional children are not, because an instance can omit them.
    """
    if schema.get("type") != "object":
        return []
    properties = schema.get("properties", {})
    patterns = schema.get("patternProperties", {})
    conflicts = []
    for name in schema.get("required", []):
        if (
            schema.get("additionalProperties") is False
            and name not in properties
            and not any(re.search(pattern, name) for pattern in patterns)
        ):
            conflicts.append(f"{path}.{name}: required but forbidden by additionalProperties=false")
        child = properties.get(name)
        if isinstance(child, dict):
            conflicts.extend(required_object_conflicts(child, f"{path}.{name}"))
    return conflicts
