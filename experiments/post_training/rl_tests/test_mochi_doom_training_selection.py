# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

from finestore.rl.strata import TaskStrata, read_selection_manifest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow

from experiments.post_training.mochi_doom_ftpo.training_selection import (
    publish_selection,
    selected_training,
    training_groups,
)
from experiments.post_training.task_curation.datasets.skyrl.math import convert_dapo_math


def test_training_batch_approximates_two_to_one_weights_and_removes_demoted_domains():
    statuses = dict.fromkeys(("math16k", "math32k", "if32k", "code32k", "rgym32k", "struct32k", "mcq32k"), "target")
    assert training_groups(statuses) == {
        "math": 19,
        "if32k": 9,
        "code32k": 9,
        "rgym32k": 9,
        "struct32k": 9,
        "mcq32k": 9,
    }
    statuses.update(code32k="non-target", mcq32k="non-target", rgym32k="non-target")
    assert training_groups(statuses) == {"math": 32, "if32k": 16, "struct32k": 16}


def test_selected_pool_publication_preserves_balanced_structure_and_native_family_identity(tmp_path):
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = convert_dapo_math(
        RawRow(
            "a",
            source,
            {
                "prompt": [{"role": "user", "content": "What is 1 + 1?"}],
                "reward_model": {"ground_truth": "2"},
            },
        ),
        ConversionContext({}, None),
    )
    rows = [
        {
            "task_spec": task.model_copy(update={"id": f"{family}-{index}"}).model_dump_json(),
            **TaskStrata(family=family, source="fixture").model_dump(),
        }
        for family, count in (("math", 64), ("if32k", 20), ("calendar", 10), ("closure", 40), ("rgym32k", 20))
        for index in range(count)
    ]
    groups = {"math": 32, "if32k": 16, "struct32k": 16}
    selected = selected_training(rows, groups)
    assert Counter(row["family"] for row in selected) == {"math": 64, "if32k": 20, "struct32k": 20}
    heldout = [
        {
            "task_spec": task.model_copy(update={"id": "heldout"}).model_dump_json(),
            **TaskStrata(family="math", source="fixture").model_dump(),
        }
    ]
    publish_selection(
        StoragePath(str(tmp_path)),
        "fixture-targets",
        "2026.11.25.1",
        selected,
        heldout,
        heldout,
        {"heldout": [], "exclusions": [], "groups_per_family": groups},
    )
    manifest = read_selection_manifest(str(tmp_path))
    assert manifest is not None
    assert manifest.identity("calendar-0").panel_name == "fixture-targets"
    structured_ids = {uid for uid, row in manifest.tasks.items() if row.family == "struct32k"}
    assert Counter(uid.split("-")[0] for uid in structured_ids) == {"calendar": 10, "closure": 10}
    assert not any(uid.startswith("rgym32k-") for uid in manifest.tasks)
    assert "heldout" in manifest.tasks
