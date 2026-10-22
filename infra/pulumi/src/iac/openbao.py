# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public OpenBao policy declarations for Iris credential delegation."""

import json
import re

from pydantic import BaseModel, ConfigDict, Field

CONTROLLER_TTL = 1200
TASK_TTL = 600


class CredentialProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    # Operator-owned KV v2 object paths, relative to the namespace's secret/ mount.
    objects: tuple[str, ...] = Field(min_length=1)


def credential_catalog(cluster: str, profiles: tuple[CredentialProfile, ...]) -> dict:
    """Render policy and role API bodies; never create aliases or populate secrets."""
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", cluster):
        raise ValueError("Cluster names must not introduce alias wildcards or namespace paths")
    if len({profile.name for profile in profiles}) != len(profiles):
        raise ValueError("Credential profile names must be unique")
    policies = {}
    roles = {}
    aliases = []
    controller_paths: dict[str, dict[str, object]] = {
        "auth/token/lookup-self": {"capabilities": ["read"]},
        "auth/token/revoke-accessor": {"capabilities": ["update"]},
    }
    for profile in profiles:
        paths = {}
        for path in profile.objects:
            if (
                not path
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or any(marker in path for marker in ("*", "+", "{{", "}}"))
            ):
                raise ValueError("Profiles must name exact KV objects, without wildcards or templates")
            paths[f"secret/data/{path}"] = {"capabilities": ["read"]}
        policy_name = f"workload-{profile.name}"
        alias = f"workload:{cluster}:{profile.name}"
        policies[policy_name] = json.dumps({"path": paths}, indent=2)
        roles[profile.name] = {
            "allowed_policies": [policy_name],
            "allowed_entity_aliases": [alias],
            "orphan": False,
            "renewable": False,
            "token_type": "service",
            "token_explicit_max_ttl": TASK_TTL,
            "token_no_default_policy": True,
        }
        controller_paths[f"auth/token/create/{profile.name}"] = {
            "capabilities": ["update"],
            "required_parameters": ["entity_alias"],
            "allowed_parameters": {"entity_alias": [alias], "*": []},
        }
        aliases.append({"name": alias, "entity_name": alias, "auth_mount": "token/"})
    policies["iris-controller"] = json.dumps({"path": controller_paths}, indent=2)
    return {
        "namespace": cluster,
        "kv_mount": {"path": "secret", "type": "kv", "options": {"version": "2"}},
        "policies": policies,
        "token_roles": roles,
        "required_precreated_aliases": aliases,
        "controller_jwt_role_parameters": {
            "role_type": "jwt",
            "user_claim": "sub",
            "token_policies": ["iris-controller"],
            "token_type": "service",
            "token_ttl": CONTROLLER_TTL,
            "token_max_ttl": CONTROLLER_TTL,
            "token_explicit_max_ttl": CONTROLLER_TTL,
            "token_no_default_policy": True,
            "clock_skew_leeway": 10,
            "expiration_leeway": 10,
            "not_before_leeway": 10,
        },
    }
