# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from experiments.post_training.task_curation.images import IMAGES
from experiments.post_training.task_curation.pipeline import Image
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
# Available Atlas listings without a declaration. They read open-athena/task-trove, the output of
# the retired cleanup, and have no archive in open-thoughts/TaskTrove.
UNDECLARED_ATLAS_IDS = {
    "Task Trove:AweAI-Team__CalibForge",
    "Task Trove:GAIR__OpenSWE__openswe_oss",
    "Task Trove:GAIR__OpenSWE__openswe_other",
    "Task Trove:R2E-Gym__R2E-Gym-V1",
    "Task Trove:SWE-Gym__SWE-Gym",
    "Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__code",
    "Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__music",
}
FAMILY_TESTS = (test_arc, test_nemotron_ultra, test_reasoning_gym, test_skyrl, test_tasktrove_code, test_tasktrove_text)


def test_every_declaration_has_a_fixture_row():
    covered = [name for module in FAMILY_TESTS for name in module.ROWS]
    assert len(covered) == len(set(covered))
    assert set(covered) == set(all_pipelines())


def test_declarations_join_available_atlas_listings_once():
    listings = json.loads(ATLAS_CATALOG.read_text())["listings"]
    available = {listing["id"] for listing in listings if listing["atlas_status"] == "Available"}
    declared = [pipeline.atlas_id for pipeline in all_pipelines().values()]
    assert len(declared) == len(set(declared))
    assert set(declared) <= available
    assert available - set(declared) == UNDECLARED_ATLAS_IDS


def test_declared_images_are_committed():
    declared = {pipeline.environment for pipeline in all_pipelines().values() if isinstance(pipeline.environment, Image)}
    assert declared <= set(IMAGES)
