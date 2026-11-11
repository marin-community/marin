# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate pinned Micro210 environment evidence."""

import click


def validate_environment_preflight(manifest: dict, data_uri: str) -> None:
    """Reject incomplete or mismatched environment checks before constructing a GPU artifact."""
    if manifest.get("scope") == "diagnostic":
        raise click.ClickException("A diagnostic preflight does not cover an evaluation dataset.")
    required = set(manifest["required_source_keys"])
    missing = required - set(manifest["validated_source_keys"])
    if manifest["failures"] or missing or len(required) != manifest["expected_source_keys"]:
        raise click.ClickException(f"Environment manifest is incomplete ({len(missing)} missing keys).")
    if data_uri != manifest["data_uri"]:
        raise click.ClickException("Environment manifest belongs to a different dataset scope.")
