# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

from click.testing import CliRunner
from marin.execution.lazy import materialized_config

from experiments.post_training.dspark_sft import CompletedSFT, OfflineDSpark, TeacherFamily, TrainingMix, artifacts, main


def test_dspark_plan_accepts_pinned_mochi_without_resolving_nominal_export(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    config = tmp_path / "recipe.json"
    config.write_text(
        json.dumps(
            {
                "teacher": {
                    "family": "mochi",
                    "identity": "grug/hero-finished-hf@2026.11.22.1",
                    "directory": str(tmp_path / "weights"),
                    "manifest_sha256": "1" * 64,
                },
                "data": {"directory": str(tmp_path / "mix"), "manifest_sha256": "2" * 64},
                "max_steps": 1,
                "teacher_replicas": 4,
                "trainer_ranks": 2,
            }
        )
    )
    result = CliRunner().invoke(main, ["--config", str(config)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "artifacts").exists()


def test_draft_artifact_binds_exact_teacher_and_mix(tmp_path):
    config = OfflineDSpark(
        CompletedSFT(TeacherFamily.SNOWBALL, "checkpoints/snowball-hf@1", str(tmp_path / "model"), "1" * 64),
        TrainingMix(str(tmp_path / "mix"), "2" * 64),
        max_steps=1,
        teacher_replicas=2,
        trainer_ranks=1,
    )
    prepared, draft = artifacts(config)
    new_prepared, new_draft = artifacts(replace(config, data=replace(config.data, manifest_sha256="3" * 64)))
    assert prepared.name != new_prepared.name
    assert draft.name != new_draft.name
    bound = materialized_config(draft, str(tmp_path))
    assert bound["teacher"] == config.teacher.directory
    assert bound["recipe"].extraction_taps == (2, 12, 22, 26)
    assert bound["prepared"] == prepared.path(str(tmp_path))
