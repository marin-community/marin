# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from rolloutengine.models import TASK_ROLLOUT_COLUMN, TaskRolloutSpec
from taskcompendium.importers.harbor.package import image_references
from zephyr.writers import write_parquet_file

from experiments.post_training.mochi_loop_guards.audit import resolved_manifest
from experiments.post_training.mochi_loop_guards.fixture_tasks import fixture_rollout
from experiments.post_training.mochi_loop_guards.selection import _harbor_rows
from experiments.post_training.task_curation.harbor_packages import package_archive


def test_frozen_source_indices_bind_named_packages(tmp_path: Path):
    archives = tmp_path / "tasks"
    archives.mkdir()
    digests = {}
    for name in ("zebra", "alpha"):
        package = tmp_path / name
        (package / "tests").mkdir(parents=True)
        (package / "task.toml").write_text('version = "1.0"\n[environment]\ndocker_image = "python:3.12-slim"\n')
        (package / "instruction.md").write_text(f"Solve {name}.")
        (package / "tests/test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
        (archives / f"{name}.tar.gz").write_bytes(package_archive(package))
        digests.update({reference: f"sha256:{'a' * 64}" for reference in image_references(package)})
    frozen = json.loads(Path("cohorts/manifest.json").read_text())["cohorts"]["tb2"]
    cohort = {**frozen, "rows": [frozen["rows"][1], frozen["rows"][0]]}

    rows = _harbor_rows("tb2", {"cohorts": {"tb2": cohort}}, str(tmp_path), digests)

    specs = [TaskRolloutSpec.model_validate_json(row[TASK_ROLLOUT_COLUMN]) for row in rows]
    assert [spec.task.id for spec in specs] == ["tb2/zebra", "tb2/alpha"]
    assert [spec.task.source.row for spec in specs] == ["tasks/zebra.tar.gz:0", "tasks/alpha.tar.gz:0"]
    assert [spec.task.context.events[-1].content for spec in specs] == ["Solve zebra.", "Solve alpha."]


def test_audit_resolves_source_indices_and_hash_order_against_published_tasks(tmp_path: Path):
    cohorts = {}
    for cohort in ("tb2", "tblite", "swe100", "swesmith_ho"):
        published_ids = [f"{cohort}/alpha", f"{cohort}/zebra"]
        frozen_ids = list(reversed(published_ids)) if cohort == "swesmith_ho" else ["source:0", "source:1"]
        key = "swesmith-ho" if cohort == "swesmith_ho" else cohort
        cohorts[key] = {"rows": [{"task_id": task_id} for task_id in frozen_ids], "trials_per_task": 2}
        write_parquet_file(
            [{TASK_ROLLOUT_COLUMN: fixture_rollout(task_id).model_dump_json()} for task_id in published_ids],
            str(tmp_path / f"{cohort}.parquet"),
        )
    manifest = {"manifest_sha256": "frozen", "cohorts": cohorts}

    resolved = resolved_manifest(manifest, str(tmp_path))

    assert [row["task_id"] for row in resolved["cohorts"]["tb2"]["rows"]] == ["tb2/alpha", "tb2/zebra"]
    assert [row["task_id"] for row in resolved["cohorts"]["swesmith-ho"]["rows"]] == [
        "swesmith_ho/zebra",
        "swesmith_ho/alpha",
    ]
    assert manifest["cohorts"]["tb2"]["rows"] == [{"task_id": "source:0"}, {"task_id": "source:1"}]
