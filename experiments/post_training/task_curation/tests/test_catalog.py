# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from experiments.post_training.task_curation.images.recipes import RECIPES
from experiments.post_training.task_curation.sources import all_pipelines
from experiments.post_training.task_curation.tests import (
    test_arc,
    test_nemotron_ultra,
    test_reasoning_gym,
    test_skyrl,
    test_tasktrove_code,
    test_tasktrove_text,
)

ATLAS_CATALOG = Path(__file__).parents[1] / "atlas_catalog.json"
FAMILY_TESTS = (test_arc, test_nemotron_ultra, test_reasoning_gym, test_skyrl, test_tasktrove_code, test_tasktrove_text)


def test_every_declaration_has_a_fixture_row():
    covered = [name for module in FAMILY_TESTS for name in module.ROWS]
    assert len(covered) == len(set(covered))
    assert set(covered) == set(all_pipelines())


def test_declarations_join_available_atlas_listings_once():
    """Catches two declarations claiming one listing, or a declaration naming an atlas id with no available listing."""
    listings = json.loads(ATLAS_CATALOG.read_text())["listings"]
    available = {listing["id"] for listing in listings if listing["atlas_status"] == "Available"}
    declared = [pipeline.atlas_id for pipeline in all_pipelines().values()]
    assert len(declared) == len(set(declared))
    assert set(declared) <= available


def test_declared_grader_images_are_buildable_recipes():
    """The build CLI builds only registered recipes, so a declaration naming another could never resolve."""
    declared = {pipeline.grader_image for pipeline in all_pipelines().values()} - {None}
    assert declared <= set(RECIPES.values())
