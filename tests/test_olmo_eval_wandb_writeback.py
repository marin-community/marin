# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv
import json

import pytest

from experiments.domain_phase_mix.write_olmo_eval_wandb import (
    build_payload,
    flatten_metrics,
    load_target_metadata,
    logged_artifact_uri,
    task_metric_key,
    write_payload_files,
)


def test_flatten_metrics_preserves_nested_scalar_paths():
    metrics = {
        "accuracy": {"logprob": 0.25, "ignored": "not-scalar"},
        "bits_per_byte": 1.75,
        "nan": float("nan"),
    }

    assert flatten_metrics(metrics) == [
        ("accuracy/logprob", 0.25),
        ("bits_per_byte", 1.75),
    ]


def test_task_metric_key_keeps_exact_metric_hierarchy_with_sanitized_task():
    key = task_metric_key(
        key_prefix="olmo_base_eval/easy_bpb",
        task="arc_easy:bpb:olmo3base",
        metric_path="accuracy/logprob",
    )

    assert key == "olmo_base_eval/easy_bpb/arc_easy_bpb_olmo3base/accuracy/logprob"


def test_load_target_metadata_uses_single_matching_matrix_row(tmp_path):
    metadata_csv = tmp_path / "matrix.csv"
    with metadata_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "wandb_run_id", "checkpoint_root"])
        writer.writeheader()
        writer.writerow(
            {
                "run_name": "baseline_proportional",
                "wandb_run_id": "baseline_proportional-982696",
                "checkpoint_root": "gs://checkpoint",
            }
        )

    target = load_target_metadata(
        metadata_csv=metadata_csv,
        target_run_name="baseline_proportional",
        target_wandb_run_id=None,
        checkpoint_root=None,
        hf_repo="Calvin-Xu/example",
        fieldbook_run_id="run_abc",
        fieldbook_job_id="job_abc",
        slurm_job_id="15955836",
        olmo_eval_git_sha="deadbeef",
    )

    assert target.wandb_run_id == "baseline_proportional-982696"
    assert target.checkpoint_root == "gs://checkpoint"
    assert target.hf_repo == "Calvin-Xu/example"
    assert target.fieldbook_run_id == "run_abc"
    assert target.slurm_job_id == "15955836"


def test_load_target_metadata_rejects_ambiguous_matrix_rows(tmp_path):
    metadata_csv = tmp_path / "matrix.csv"
    with metadata_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "wandb_run_id"])
        writer.writeheader()
        writer.writerow({"run_name": "baseline_proportional", "wandb_run_id": "a"})
        writer.writerow({"run_name": "baseline_proportional", "wandb_run_id": "b"})

    with pytest.raises(ValueError, match="Expected exactly one metadata row"):
        load_target_metadata(
            metadata_csv=metadata_csv,
            target_run_name="baseline_proportional",
            target_wandb_run_id=None,
            checkpoint_root=None,
            hf_repo=None,
            fieldbook_run_id=None,
            fieldbook_job_id=None,
            slurm_job_id=None,
            olmo_eval_git_sha=None,
        )


def test_build_payload_flattens_task_metrics_and_summary_values(tmp_path):
    metrics_json = tmp_path / "metrics.json"
    metrics_json.write_text(
        json.dumps(
            {
                "experiment_id": "abc123",
                "experiment_name": "olmo_eval_canary",
                "experiment_group": "olmo_base_eval",
                "experiment_duration_seconds": 12.5,
                "tasks": [
                    {
                        "task": "arc_easy:bpb:olmo3base",
                        "metrics": {"accuracy": {"logprob": 0.5}},
                        "num_instances": 8,
                        "primary_metric": "accuracy:logprob",
                    },
                    {
                        "task": "codex_humaneval:bpb:olmo3base",
                        "metrics": {"bits_per_byte": 2.0},
                        "num_instances": 164,
                        "primary_metric": "bits_per_byte",
                    },
                ],
            }
        )
    )
    target = load_target_metadata(
        metadata_csv=None,
        target_run_name="baseline_proportional",
        target_wandb_run_id="baseline_proportional-982696",
        checkpoint_root="gs://checkpoint",
        hf_repo=None,
        fieldbook_run_id="run_abc",
        fieldbook_job_id="job_abc",
        slurm_job_id="15955836",
        olmo_eval_git_sha=None,
    )

    payload = build_payload(metrics_json, target, "olmo_base_eval/easy_bpb")

    assert len(payload.metric_records) == 2
    assert payload.summary_updates["olmo_base_eval/easy_bpb/_summary/task_count"] == 2
    assert payload.summary_updates["olmo_base_eval/easy_bpb/_summary/scalar_metric_count"] == 2
    assert payload.summary_updates["olmo_base_eval/easy_bpb/_summary/primary_metric_mean"] == pytest.approx(1.25)
    assert payload.summary_updates["olmo_base_eval/easy_bpb/_summary/experiment_duration_seconds"] == 12.5
    assert payload.summary_updates["olmo_base_eval/easy_bpb/_provenance/fieldbook_run_id"] == "run_abc"
    assert payload.summary_updates["olmo_base_eval/easy_bpb/arc_easy_bpb_olmo3base/accuracy/logprob"] == 0.5
    assert payload.summary_updates["olmo_base_eval/easy_bpb/codex_humaneval_bpb_olmo3base/bits_per_byte"] == 2.0


def test_write_payload_files_exports_long_table_and_manifest(tmp_path):
    metrics_json = tmp_path / "metrics.json"
    metrics_json.write_text(
        json.dumps(
            {
                "experiment_name": "olmo_eval_canary",
                "tasks": [
                    {
                        "task": "arc_easy:bpb:olmo3base",
                        "metrics": {"accuracy": {"logprob": 0.5}},
                        "num_instances": 8,
                        "primary_metric": "accuracy:logprob",
                    }
                ],
            }
        )
    )
    target = load_target_metadata(
        metadata_csv=None,
        target_run_name="baseline_proportional",
        target_wandb_run_id="baseline_proportional-982696",
        checkpoint_root=None,
        hf_repo=None,
        fieldbook_run_id=None,
        fieldbook_job_id=None,
        slurm_job_id=None,
        olmo_eval_git_sha=None,
    )
    payload = build_payload(metrics_json, target, "olmo_base_eval/easy_bpb")

    table_path, manifest_path = write_payload_files(payload, tmp_path / "writeback")

    with table_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    manifest = json.loads(manifest_path.read_text())
    assert rows == [
        {
            "task": "arc_easy:bpb:olmo3base",
            "metric_path": "accuracy/logprob",
            "value": "0.5",
            "is_primary": "True",
            "primary_metric": "accuracy:logprob",
            "num_instances": "8",
            "wandb_summary_key": "olmo_base_eval/easy_bpb/arc_easy_bpb_olmo3base/accuracy/logprob",
        }
    ]
    assert manifest["target"]["wandb_run_id"] == "baseline_proportional-982696"
    assert manifest["metric_record_count"] == 1


def test_logged_artifact_uri_does_not_duplicate_existing_version():
    class LoggedArtifact:
        entity = "marin-community"
        project = "marin"
        name = "artifact-name:v0"
        version = "v0"

    assert logged_artifact_uri(LoggedArtifact()) == "marin-community/marin/artifact-name:v0"
