# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate Loom stack outputs consumed by GitHub resources."""


def federation_profile(value: object, name: str) -> str:
    """Return one named GitHub federation's Loom profile."""
    if not isinstance(value, dict):
        raise ValueError("Loom githubFederationProfiles output must be an object")
    profile = value.get(name)
    if not isinstance(profile, str) or not profile:
        raise ValueError(f"Loom stack must export a profile for GitHub federation {name!r}")
    return profile
