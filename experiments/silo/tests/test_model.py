# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from silo.errors import SiloRecipeError
from silo.model import (
    CANDIDATE_DEFAULT,
    VERIFIER_DEFAULT,
    ResourceProfile,
    build_dockerfile,
    parse_recipe,
    snapshot_name,
)

DIGEST = "docker.io/library/python@sha256:" + "2" * 64
FROM_ONLY = f"FROM {DIGEST}\n"


def test_snapshot_names_match_the_pipeline_byte_for_byte():
    # Values produced by capability_pipeline.daytona_resources.snapshot_name on
    # 2026-09-22 for the same inputs. If these drift, every snapshot the
    # controller already recorded stops resolving.
    assert snapshot_name("cap-harbor", FROM_ONLY, CANDIDATE_DEFAULT) == "cap-harbor-c8240700ed97847f44ea"
    assert snapshot_name("cap-harbor", FROM_ONLY, VERIFIER_DEFAULT) == "cap-harbor-8a7f69a902e01156a2fb"


def test_cache_bytes_matches_the_pipeline_encoding():
    assert CANDIDATE_DEFAULT.cache_bytes() == b'{"cpu":4,"disk_gb":10,"memory_gb":8}'


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_profile_rejects_non_positive_ints(bad):
    with pytest.raises(ValueError):
        ResourceProfile(cpu=bad, memory_gb=1, disk_gb=1)


def test_from_only_recipe_needs_no_build():
    plan = parse_recipe(FROM_ONLY)
    assert plan.base_ref == DIGEST
    assert plan.base_pinned
    assert not plan.requires_build


def test_tag_base_is_accepted_but_recorded_as_unpinned():
    # Builder Dockerfiles use tags (Daytona accepted them); the record says so.
    plan = parse_recipe("FROM ubuntu:22.04\nRUN apt-get update\n")
    assert not plan.base_pinned
    assert plan.receipt()["base_pinned"] is False


def test_derive_daytona_recipe_shape_is_understood():
    # The shape image_runtime_metadata.derive_daytona_recipe emits.
    recipe = f'FROM {DIGEST}\nENTRYPOINT ["/usr/bin/tini","--"]\nCMD ["python3"]\n'
    plan = parse_recipe(recipe)
    assert plan.entrypoint == ("/usr/bin/tini", "--")
    assert plan.cmd == ("python3",)
    assert not plan.requires_build


def test_verifier_recipe_shape_requires_a_build():
    # The shape daytona_policy.verifier_snapshot_recipe emits.
    recipe = f"FROM {DIGEST}\nUSER root\nRUN python3 -m pip install --no-cache-dir --no-deps a==1 b==2\n"
    plan = parse_recipe(recipe)
    assert plan.user == "root"
    assert plan.requires_build
    assert plan.run_steps == ("python3 -m pip install --no-cache-dir --no-deps a==1 b==2",)


def test_line_continuations_join():
    plan = parse_recipe(f"FROM {DIGEST}\nRUN echo a \\\n    && echo b\n")
    assert plan.run_steps == ("echo a      && echo b",)


@pytest.mark.parametrize(
    "recipe",
    [
        f"FROM {DIGEST} AS build\n",
        f"FROM {DIGEST}\nCOPY . /app\n",
        f"FROM {DIGEST}\nFROM {DIGEST}\n",
        "RUN true\n",
        "",
        f"FROM {DIGEST}\nENTRYPOINT [not json\n",
    ],
)
def test_parse_recipe_outside_the_subset_is_refused_not_guessed(recipe):
    with pytest.raises(SiloRecipeError):
        parse_recipe(recipe)


def test_rendered_dockerfile_roundtrips():
    recipe = f'FROM {DIGEST}\nUSER root\nRUN echo hi\nENTRYPOINT ["/bin/sh"]\nCMD ["-c","x"]\n'
    assert parse_recipe(build_dockerfile(parse_recipe(recipe))) == parse_recipe(recipe)
