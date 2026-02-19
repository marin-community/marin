# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for image tag parsing in cluster.py."""

import pytest

from iris.cli.cluster import _parse_artifact_registry_tag, _parse_ghcr_tag


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("ghcr.io/marin-community/iris-controller:latest", ("marin-community", "iris-controller", "latest")),
        ("ghcr.io/marin-community/iris-worker:v1.2.3", ("marin-community", "iris-worker", "v1.2.3")),
        ("ghcr.io/myorg/myimage", ("myorg", "myimage", "latest")),
    ],
)
def test_parse_ghcr_tag(tag, expected):
    assert _parse_ghcr_tag(tag) == expected


@pytest.mark.parametrize(
    "tag",
    [
        "us-central1-docker.pkg.dev/proj/repo/img:latest",
        "docker.io/library/ubuntu:latest",
        "ghcr.io/incomplete",
        "",
    ],
)
def test_parse_ghcr_tag_returns_none_for_non_ghcr(tag):
    assert _parse_ghcr_tag(tag) is None


@pytest.mark.parametrize(
    "tag, expected",
    [
        (
            "us-central1-docker.pkg.dev/hai-gcp-models/marin/iris-worker:latest",
            ("us-central1", "hai-gcp-models", "iris-worker", "latest"),
        ),
        (
            "us-central2-docker.pkg.dev/myproj/repo/myimg:v2",
            ("us-central2", "myproj", "myimg", "v2"),
        ),
        (
            "us-central1-docker.pkg.dev/proj/repo/img",
            ("us-central1", "proj", "img", "latest"),
        ),
    ],
)
def test_parse_artifact_registry_tag(tag, expected):
    assert _parse_artifact_registry_tag(tag) == expected


@pytest.mark.parametrize(
    "tag",
    [
        "ghcr.io/marin-community/iris-controller:latest",
        "docker.io/library/ubuntu:latest",
        "",
    ],
)
def test_parse_artifact_registry_tag_returns_none_for_non_gcp(tag):
    assert _parse_artifact_registry_tag(tag) is None
