# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve task environment templates at the machine boundary."""

import re
from collections.abc import Mapping

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_vars(environment: Mapping[str, str], host_environment: Mapping[str, str]) -> dict[str, str]:
    """Resolve ``${NAME}`` and ``${NAME:-default}`` in environment values."""
    resolved = {}
    for key, value in environment.items():
        match = _TEMPLATE_PATTERN.fullmatch(value)
        if match is None:
            resolved[key] = value
            continue
        name, default = match.groups()
        if name in host_environment:
            resolved[key] = host_environment[name]
        elif default is not None:
            resolved[key] = default
        else:
            raise ValueError(f"Environment variable '{name}' not found in host environment")
    return resolved
