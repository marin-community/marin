#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# ///
"""Verify the external GitHub App and operator-host image build prerequisites."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from types import MappingProxyType
from typing import Any

EXPECTED_PERMISSIONS = MappingProxyType(
    {
        "contents": "write",
        "issues": "write",
        "metadata": "read",
        "pull_requests": "write",
    }
)
EXPECTED_EVENTS = frozenset({"issue_comment", "pull_request_review_comment"})
PULUMI_DIR = Path(__file__).resolve().parent


def gh_json(path: str) -> Any:
    result = subprocess.run(
        ["gh", "api", path],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    return json.loads(result.stdout)


def validate_github_app(
    app: dict[str, Any], installations: dict[str, Any], organization: str, slug: str
) -> dict[str, Any]:
    owner = app.get("owner", {}).get("login")
    if owner != organization:
        raise ValueError(f"GitHub App {slug!r} is owned by {owner!r}, expected {organization!r}")

    matches = [
        installation for installation in installations.get("installations", []) if installation.get("app_slug") == slug
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {slug!r} installation on {organization}, found {len(matches)}")
    installation = matches[0]
    if installation.get("target_type") != "Organization":
        raise ValueError(f"GitHub App {slug!r} is not installed on an organization")
    if installation.get("repository_selection") != "all":
        raise ValueError(f"GitHub App {slug!r} must cover all {organization} repositories")

    permissions = installation.get("permissions", {})
    missing_permissions = {
        name: access for name, access in EXPECTED_PERMISSIONS.items() if permissions.get(name) != access
    }
    if missing_permissions:
        raise ValueError(f"GitHub App {slug!r} has incorrect permissions: {missing_permissions}")
    missing_events = EXPECTED_EVENTS - set(installation.get("events", []))
    if missing_events:
        raise ValueError(f"GitHub App {slug!r} is missing events: {sorted(missing_events)}")
    return installation


def validate_buildx(builder: str | None) -> None:
    command = ["docker", "buildx", "inspect", "--bootstrap"]
    if builder:
        command.insert(3, builder)
    result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise ValueError(f"Docker buildx is unavailable: {result.stderr.strip()}")
    platform_line = next(
        (
            line.removeprefix("Platforms:").strip()
            for line in result.stdout.splitlines()
            if line.startswith("Platforms:")
        ),
        "",
    )
    platforms = {platform.strip().removesuffix("*") for platform in platform_line.split(",")}
    if "linux/amd64" not in platforms:
        raise ValueError(f"Docker buildx builder does not support linux/amd64: {platform_line}")


def validate_registry_credentials(config: dict[str, Any], registry: str) -> str:
    helpers = config.get("credHelpers", {})
    helper = helpers.get(registry) if isinstance(helpers, dict) else None
    auths = config.get("auths", {})
    registry_auth = auths.get(registry) if isinstance(auths, dict) else None
    if isinstance(registry_auth, dict) and registry_auth.get("auth"):
        return "inline"
    helper = helper or config.get("credsStore")
    if not isinstance(helper, str) or not helper:
        raise ValueError(f"Docker has no credential helper for {registry}; run gcloud auth configure-docker {registry}")
    result = subprocess.run(
        [f"docker-credential-{helper}", "get"],
        input=f"{registry}\n",
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"Docker credential helper {helper!r} cannot authenticate {registry}")
    return helper


def stack_build_config(stack: str) -> tuple[str | None, str]:
    result = subprocess.run(
        ["pulumi", "config", "--json", "--cwd", str(PULUMI_DIR), "--stack", stack],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    config = json.loads(result.stdout)
    value = config.get("marin-loom:buildxBuilder", {}).get("value")
    if value is not None and not isinstance(value, str):
        raise ValueError("buildxBuilder must be a string")
    region = config.get("marin-loom:region", {}).get("value")
    if not isinstance(region, str) or not region:
        raise ValueError("the stack must configure a region")
    return value, f"{region}-docker.pkg.dev"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organization", default="marin-community")
    parser.add_argument("--app", default="loom-oa-dev")
    parser.add_argument("--stack", default="marin-loom")
    args = parser.parse_args()

    validate_github_app(
        gh_json(f"/apps/{args.app}"),
        gh_json(f"/orgs/{args.organization}/installations"),
        args.organization,
        args.app,
    )
    builder, registry = stack_build_config(args.stack)
    validate_buildx(builder)
    docker_config_dir = Path(os.environ.get("DOCKER_CONFIG", Path.home() / ".docker"))
    try:
        docker_config = json.loads((docker_config_dir / "config.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read Docker credential configuration: {error}") from error
    validate_registry_credentials(docker_config, registry)
    print(f"GitHub App {args.app} is owned by and installed across {args.organization}.")
    print(f"Docker buildx supports linux/amd64 and {registry} authentication is configured.")


if __name__ == "__main__":
    main()
