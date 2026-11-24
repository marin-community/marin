# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from finestore.rl.strata import read_selection_manifest
from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow

from experiments.post_training.mochi_doom_ftpo.cohorts import Candidate, _publish, freeze, target_status
from experiments.post_training.task_curation.datasets.skyrl.math import convert_dapo_math


def test_freeze_excludes_entire_reserve_and_aliases_from_training(tmp_path):
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = convert_dapo_math(
        RawRow(
            "a",
            source,
            {"prompt": [{"role": "user", "content": "What is 1 + 1?"}], "reward_model": {"ground_truth": "2"}},
        ),
        ConversionContext({}, None),
    )
    rows = []
    for domain, count in (
        ("math", 520),
        ("if32k", 260),
        ("code32k", 260),
        ("rgym32k", 260),
        ("mcq32k", 260),
        ("calendar", 132),
        ("closure", 132),
    ):
        rows.extend(
            Candidate(
                task.model_copy(update={"id": f"{domain}-{i}"}),
                domain,
                f"{domain}-{i:04d}",
                domain == "math" and i < 512,
            )
            for i in range(count)
        )
    # Both aliases are beyond the selected range, and must still be removed by identity or prompt.
    rows.append(replace(rows[0], task=task.model_copy(update={"id": "hash-alias"}), original_validation=False))
    rows.append(replace(rows[0], prompt_hash="math-z-alias", original_validation=False))
    cohorts, training = freeze(rows)
    held = [row for values in cohorts.values() for row in values]
    assert not {row.prompt_hash for row in held} & {row.prompt_hash for row in training}
    assert not {row.task.id for row in held} & {row.task.id for row in training}
    _publish(str(tmp_path), cohorts, training, {"heldout": [], "exclusions": []})
    published = read_selection_manifest(str(tmp_path))
    assert published is not None
    assert set(published.tasks) == {row.task.id for row in [*training, *held]}
    assert len(training) == 32
    assert [row.prompt_hash for row in cohorts["math32k"]] == [row.prompt_hash for row in cohorts["math16k"][:256]]
    assert sum(row.domain == "calendar" for row in cohorts["struct32k"][:128]) == 64


@pytest.mark.parametrize("confirmed,count,decision", [(9, 128, "extend"), (9, 256, "non-target"), (10, 128, "target")])
def test_sparse_domain_requires_extension_before_demotion(confirmed, count, decision):
    assert target_status(confirmed, count) == decision
