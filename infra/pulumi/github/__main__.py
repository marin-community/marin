# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marin's GitHub resources."""

import os
import sys
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pulumi
from iac.github.credentials import credential_manifest
from iac.github.resources import credential_resource_plans, register_credentials


def main() -> None:
    config = pulumi.Config()
    manifest = credential_manifest(
        schema_version=config.require_int("schemaVersion"),
        organization=config.require("organization"),
        repositories=cast(list[str], config.require_object("repositories")),
        credentials=cast(list[dict], config.require_object("credentials")),
    )
    provider_owner = pulumi.Config("github").require("owner")
    if provider_owner != manifest.organization:
        raise ValueError(
            f"github:owner {provider_owner!r} does not match marin-github:organization {manifest.organization!r}"
        )

    plans = credential_resource_plans(manifest)
    register_credentials(manifest)
    pulumi.export("credential_count", len(plans))


main()
