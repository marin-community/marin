# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import shutil
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.domain_phase_mix.apply_olmo_eval_sc_fanout_patch as olmo_patch
import experiments.domain_phase_mix.olmo_base_eval_sc_batch as olmo_sc
from experiments.domain_phase_mix.olmo_base_eval_sc_batch import (
    build_eval_manifest,
    clean_incomplete_dataset_cache,
    model_dir_for_row,
    offline_cache_report,
    prewarm_datasets,
    resolve_checkpoint,
    rows_from_upload_manifest,
    safe_segment,
    stage_checkpoints,
    validate_eval_manifest,
    worker_environment,
    write_csv,
    write_sbatch,
)


def upload_manifest_text(rows: list[dict[str, object]]) -> str:
    fieldnames = [
        "panel",
        "scale",
        "run_name",
        "source_experiment",
        "checkpoint_root",
        "checkpoint_uri",
        "expected_checkpoint_step",
        "path_in_repo",
        "metadata_json",
    ]
    parts: list[str] = []
    writer = csv.DictWriter(_ListWriter(parts), fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return "".join(parts)


class _ListWriter:
    def __init__(self, parts: list[str]):
        self.parts = parts

    def write(self, value: str) -> None:
        self.parts.append(value)


def sample_upload_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "panel": "parity",
        "scale": "60m_1p2b",
        "run_name": "baseline_proportional",
        "source_experiment": "pinlin_calvin_xu/data_mixture/ngd3dm2_hybrid_canary",
        "checkpoint_root": (
            "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
            "ngd3dm2_hybrid_canary/baseline_proportional-bac80b"
        ),
        "checkpoint_uri": (
            "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
            "ngd3dm2_hybrid_canary/baseline_proportional-bac80b/hf/step-4576"
        ),
        "expected_checkpoint_step": 4576,
        "path_in_repo": "checkpoints/60m_1p2b/parity/baseline_proportional/hf/step-4576",
        "metadata_json": json.dumps({"run_id": 0, "wandb_run_id": "baseline_proportional-bac80b"}),
    }
    row.update(overrides)
    return row


def write_basic_skills_snapshot(work_dir: Path) -> Path:
    snapshot = work_dir / "hf_home" / "hub" / "datasets--allenai--basic-skills" / "snapshots" / "abc123"
    for subset in olmo_sc.BASIC_SKILLS_SUBTASKS:
        path = snapshot / subset / "validation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    ref = work_dir / "hf_home" / "hub" / "datasets--allenai--basic-skills" / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text("abc123")
    return snapshot


def test_rows_from_upload_manifest_prefers_metadata_wandb_run_id():
    rows = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=3,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.index == 3
    assert row.wandb_run_id == "baseline_proportional-bac80b"
    assert row.hf_checkpoint_path == "checkpoints/60m_1p2b/parity/baseline_proportional/hf/step-4576"
    assert row.output_name == "003_60m_1p2b_parity_baseline_proportional"


def test_rows_from_upload_manifest_falls_back_to_checkpoint_root_leaf():
    rows = rows_from_upload_manifest(
        upload_manifest_text(
            [
                sample_upload_row(
                    panel="proportional_controllability",
                    run_name="p60_del_00",
                    checkpoint_root=(
                        "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
                        "ngd3dm2_pctrl60/p60_del_00-1035b3"
                    ),
                    metadata_json=json.dumps({"intervention_type": "domain_deletion"}),
                    path_in_repo="checkpoints/60m_1p2b/proportional_controllability/p60_del_00/hf/step-4576",
                )
            ]
        ),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )

    assert rows[0].wandb_run_id == "p60_del_00-1035b3"


def test_validate_eval_manifest_rejects_duplicate_wandb_ids():
    rows = rows_from_upload_manifest(
        upload_manifest_text(
            [
                sample_upload_row(run_name="a", path_in_repo="checkpoints/60m_1p2b/parity/a/hf/step-4576"),
                sample_upload_row(run_name="b", path_in_repo="checkpoints/60m_1p2b/parity/b/hf/step-4576"),
            ]
        ),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )

    with pytest.raises(ValueError, match="Duplicate wandb_run_id"):
        validate_eval_manifest(rows)


def test_build_eval_manifest_concatenates_local_manifests(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(upload_manifest_text([sample_upload_row()]))
    second.write_text(
        upload_manifest_text(
            [
                sample_upload_row(
                    run_name="run_00002",
                    checkpoint_root=(
                        "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
                        "ngd3dm2_qsplit240/run_00002-cde4ce"
                    ),
                    path_in_repo="checkpoints/60m_1p2b/parity/run_00002/hf/step-4576",
                    metadata_json=json.dumps({"run_id": 2, "wandb_run_id": "run_00002-cde4ce"}),
                )
            ]
        )
    )

    rows = build_eval_manifest([str(first), str(second)], hf_repo_id="Calvin-Xu/checkpoints")

    assert [row.index for row in rows] == [0, 1]
    assert [row.run_name for row in rows] == ["baseline_proportional", "run_00002"]


def test_write_sbatch_sets_array_bounds_and_worker_args(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = rows_from_upload_manifest(
        upload_manifest_text(
            [
                sample_upload_row(run_name="a", path_in_repo="checkpoints/60m_1p2b/parity/a/hf/step-4576"),
                sample_upload_row(
                    run_name="b",
                    checkpoint_root=(
                        "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/" "ngd3dm2_qsplit240/b-run"
                    ),
                    path_in_repo="checkpoints/60m_1p2b/parity/b/hf/step-4576",
                    metadata_json=json.dumps({"wandb_run_id": "b-run"}),
                ),
            ]
        ),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )
    write_csv(rows, manifest)
    output = tmp_path / "array.sbatch"

    class Args:
        pass

    args = Args()
    args.manifest = manifest
    args.output = output
    args.worker_script = tmp_path / "worker.py"
    args.writeback_script = tmp_path / "writeback.py"
    args.array_start_index = None
    args.array_end_index = None
    args.work_dir = tmp_path / "work"
    args.olmo_eval_dir = tmp_path / "OLMo-Eval"
    args.array_concurrency = 16
    args.job_name = "olmo-test"
    args.account = "nlp"
    args.partition = "sc-loprio"
    args.cpus_per_task = 8
    args.mem = "64G"
    args.time = "12:00:00"
    args.suite = "smoke"
    args.limit = 8
    args.key_prefix = "olmo_base_eval/easy_bpb_canary"
    args.writeback_apply = True
    args.cleanup_model = True
    args.skip_existing = True
    args.checkpoint_mode = "download"
    args.hf_revision = None
    args.offline = False

    write_sbatch(args)
    text = output.read_text()

    assert "#SBATCH --array=0-1%16" in text
    assert "--suite smoke" in text
    assert "--limit 8" in text
    assert "--writeback-apply" in text
    assert "--cleanup-model" in text
    assert "--skip-existing" in text


def test_write_sbatch_can_target_canary_range_from_full_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = rows_from_upload_manifest(
        upload_manifest_text(
            [
                sample_upload_row(run_name="a", path_in_repo="checkpoints/60m_1p2b/parity/a/hf/step-4576"),
                sample_upload_row(
                    run_name="b",
                    checkpoint_root=(
                        "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/" "ngd3dm2_qsplit240/b-run"
                    ),
                    path_in_repo="checkpoints/60m_1p2b/parity/b/hf/step-4576",
                    metadata_json=json.dumps({"wandb_run_id": "b-run"}),
                ),
                sample_upload_row(
                    run_name="c",
                    checkpoint_root=(
                        "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/" "ngd3dm2_qsplit240/c-run"
                    ),
                    path_in_repo="checkpoints/60m_1p2b/parity/c/hf/step-4576",
                    metadata_json=json.dumps({"wandb_run_id": "c-run"}),
                ),
            ]
        ),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )
    write_csv(rows, manifest)
    output = tmp_path / "canary.sbatch"

    args = Namespace(
        manifest=manifest,
        runtime_manifest=None,
        output=output,
        worker_script=tmp_path / "worker.py",
        writeback_script=tmp_path / "writeback.py",
        array_start_index=0,
        array_end_index=1,
        work_dir=tmp_path / "work",
        olmo_eval_dir=tmp_path / "OLMo-Eval",
        array_concurrency=2,
        job_name="olmo-canary",
        account="nlp",
        partition="sc-loprio",
        cpus_per_task=8,
        mem="64G",
        time="12:00:00",
        suite="full",
        limit=None,
        key_prefix="olmo_base_eval/easy_bpb",
        writeback_apply=False,
        cleanup_model=False,
        skip_existing=True,
        checkpoint_mode="local-only",
        hf_revision=None,
        offline=True,
    )

    write_sbatch(args)

    assert "#SBATCH --array=0-1%2" in output.read_text()


def test_write_sbatch_local_only_offline_disables_hub_access_by_contract(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )
    write_csv(rows, manifest)
    output = tmp_path / "array.sbatch"

    class Args:
        pass

    args = Args()
    args.manifest = manifest
    args.output = output
    args.worker_script = tmp_path / "worker.py"
    args.writeback_script = tmp_path / "writeback.py"
    args.array_start_index = None
    args.array_end_index = None
    args.work_dir = tmp_path / "work"
    args.olmo_eval_dir = tmp_path / "OLMo-Eval"
    args.array_concurrency = 16
    args.job_name = "olmo-test"
    args.account = "nlp"
    args.partition = "sc-loprio"
    args.cpus_per_task = 8
    args.mem = "64G"
    args.time = "12:00:00"
    args.suite = "full"
    args.limit = None
    args.key_prefix = "olmo_base_eval/easy_bpb"
    args.writeback_apply = True
    args.cleanup_model = False
    args.skip_existing = True
    args.checkpoint_mode = "local-only"
    args.hf_revision = "abc123"
    args.offline = True

    write_sbatch(args)
    text = output.read_text()

    assert "export HF_HUB_OFFLINE=1" in text
    assert "export HF_DATASETS_OFFLINE=1" in text
    assert "export TRANSFORMERS_OFFLINE=1" in text
    assert "export UV_OFFLINE=1" in text
    assert "export HUGGINGFACE_HUB_CACHE=" in text
    assert "export HF_ALLOW_CODE_EVAL=1" in text
    assert "export HF_MODULES_CACHE=" in text
    assert "--checkpoint-mode local-only" in text
    assert "--hf-revision abc123" in text
    assert "--cleanup-model" not in text


def test_write_sbatch_datasets_offline_download_keeps_checkpoint_hub_online(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )
    write_csv(rows, manifest)
    output = tmp_path / "array.sbatch"

    args = Namespace(
        manifest=manifest,
        runtime_manifest=None,
        output=output,
        worker_script=tmp_path / "worker.py",
        writeback_script=tmp_path / "writeback.py",
        array_start_index=0,
        array_end_index=0,
        work_dir=tmp_path / "work",
        olmo_eval_dir=tmp_path / "OLMo-Eval",
        array_concurrency=1,
        job_name="olmo-datasets-offline",
        account="nlp",
        partition="sc-loprio",
        cpus_per_task=8,
        mem="64G",
        time="12:00:00",
        suite="full",
        limit=None,
        key_prefix="olmo_base_eval/easy_bpb",
        writeback_apply=True,
        cleanup_model=True,
        skip_existing=True,
        checkpoint_mode="download",
        hf_revision=None,
        offline=False,
        datasets_offline=True,
    )

    write_sbatch(args)
    text = output.read_text()

    assert "export HF_HUB_OFFLINE=1" not in text
    assert "unset HF_HUB_OFFLINE" in text
    assert "export TRANSFORMERS_OFFLINE=1" not in text
    assert "unset TRANSFORMERS_OFFLINE" in text
    assert "export HF_DATASETS_OFFLINE=1" in text
    assert "export UV_OFFLINE=1" in text
    assert "export HF_HUB_DISABLE_PROGRESS_BARS=1" in text
    assert "export HF_DATASETS_DISABLE_PROGRESS_BARS=1" in text
    assert "export TQDM_DISABLE=1" in text
    assert "export OLMO_EVAL_TASK_PREP_WORKERS=1" in text
    assert "export OLMO_EVAL_BASIC_SKILLS_LOCAL_ROOT=" in text
    assert "--checkpoint-mode download" in text
    assert "--cleanup-model" in text
    assert "--writeback-apply" in text


def test_write_sbatch_rejects_offline_download_mode(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )
    write_csv(rows, manifest)

    class Args:
        pass

    args = Args()
    args.manifest = manifest
    args.offline = True
    args.checkpoint_mode = "download"

    with pytest.raises(ValueError, match="offline requires --checkpoint-mode local-only"):
        write_sbatch(args)


def test_write_sbatch_rejects_local_only_cleanup_model(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )
    write_csv(rows, manifest)

    class Args:
        pass

    args = Args()
    args.manifest = manifest
    args.offline = False
    args.checkpoint_mode = "local-only"
    args.cleanup_model = True

    with pytest.raises(ValueError, match="local-only cannot be combined with --cleanup-model"):
        write_sbatch(args)


def test_write_sbatch_rejects_invalid_task_prep_workers(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )
    write_csv(rows, manifest)

    args = Namespace(
        manifest=manifest,
        runtime_manifest=None,
        output=tmp_path / "array.sbatch",
        worker_script=tmp_path / "worker.py",
        writeback_script=tmp_path / "writeback.py",
        array_start_index=0,
        array_end_index=0,
        work_dir=tmp_path / "work",
        olmo_eval_dir=tmp_path / "OLMo-Eval",
        array_concurrency=1,
        task_prep_workers=0,
        job_name="olmo-test",
        account="nlp",
        partition="sc-loprio",
        cpus_per_task=8,
        mem="64G",
        time="12:00:00",
        suite="full",
        limit=None,
        key_prefix="olmo_base_eval/easy_bpb",
        writeback_apply=True,
        cleanup_model=False,
        skip_existing=True,
        checkpoint_mode="download",
        hf_revision=None,
        offline=False,
        datasets_offline=False,
    )

    with pytest.raises(ValueError, match="task-prep-workers"):
        write_sbatch(args)


def test_resolve_checkpoint_local_only_uses_staged_files(tmp_path):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]
    model_dir = model_dir_for_row(row, tmp_path)
    model_dir.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        (model_dir / name).write_text("ok")

    assert resolve_checkpoint(row, tmp_path, checkpoint_mode="local-only") == model_dir


def test_resolve_checkpoint_local_only_rejects_missing_staged_files(tmp_path):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]

    with pytest.raises(FileNotFoundError, match="Run stage-checkpoints before local-only evaluation"):
        resolve_checkpoint(row, tmp_path, checkpoint_mode="local-only")


def test_model_dir_for_row_rejects_unsafe_output_name(tmp_path):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]

    with pytest.raises(ValueError, match="path separator"):
        model_dir_for_row(replace(row, output_name="bad/name"), tmp_path)

    with pytest.raises(ValueError, match="relative output_name"):
        model_dir_for_row(replace(row, output_name=".."), tmp_path)


def test_run_one_local_only_makes_no_hub_calls(tmp_path, monkeypatch):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]
    manifest = tmp_path / "manifest.csv"
    write_csv([row], manifest)
    model_dir = model_dir_for_row(row, tmp_path)
    model_dir.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        (model_dir / name).write_text("ok")

    def fail_snapshot(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("snapshot_download must not be called in local-only mode")

    def fake_eval(**kwargs: object):
        output_dir = kwargs["output_dir"]
        assert kwargs["model_dir"] == model_dir
        metrics_json = output_dir / "metrics.json"
        output_dir.mkdir(parents=True)
        metrics_json.write_text("{}")
        return metrics_json

    def fake_writeback(**kwargs: object):
        output_dir = kwargs["output_dir"]
        manifest_json = output_dir / "wandb_writeback_manifest.json"
        output_dir.mkdir(parents=True)
        manifest_json.write_text("{}")
        return manifest_json

    monkeypatch.setattr(olmo_sc, "snapshot_download", fail_snapshot)
    monkeypatch.setattr(olmo_sc, "run_olmo_eval", fake_eval)
    monkeypatch.setattr(olmo_sc, "run_writeback", fake_writeback)

    olmo_sc.run_one(
        Namespace(
            manifest=manifest,
            index=0,
            work_dir=tmp_path,
            olmo_eval_dir=tmp_path / "OLMo-Eval",
            writeback_script=tmp_path / "writeback.py",
            suite="full",
            limit=None,
            key_prefix="olmo_base_eval/easy_bpb",
            writeback_apply=True,
            cleanup_model=False,
            skip_existing=False,
            checkpoint_mode="local-only",
            hf_revision=None,
        )
    )

    assert (tmp_path / "status" / f"{row.output_name}.json").is_file()


def test_run_one_cleanup_model_removes_download_after_eval_failure(tmp_path, monkeypatch):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]
    manifest = tmp_path / "manifest.csv"
    write_csv([row], manifest)

    def fake_snapshot(*_args: object, **kwargs: object) -> None:
        model_dir = Path(kwargs["local_dir"]) / row.hf_checkpoint_path
        model_dir.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
            (model_dir / name).write_text("ok")

    def fail_eval(**_kwargs: object) -> None:
        raise RuntimeError("eval failed")

    monkeypatch.setattr(olmo_sc, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(olmo_sc, "run_olmo_eval", fail_eval)

    with pytest.raises(RuntimeError, match="eval failed"):
        olmo_sc.run_one(
            Namespace(
                manifest=manifest,
                index=0,
                work_dir=tmp_path,
                olmo_eval_dir=tmp_path / "OLMo-Eval",
                writeback_script=tmp_path / "writeback.py",
                suite="full",
                limit=None,
                key_prefix="olmo_base_eval/easy_bpb",
                writeback_apply=True,
                cleanup_model=True,
                skip_existing=False,
                checkpoint_mode="download",
                hf_revision=None,
            )
        )

    assert not (tmp_path / "downloaded_models" / row.output_name).exists()


def test_run_one_passes_shared_cache_environment_to_eval(tmp_path, monkeypatch):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]
    manifest = tmp_path / "manifest.csv"
    write_csv([row], manifest)
    model_dir = model_dir_for_row(row, tmp_path)
    model_dir.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        (model_dir / name).write_text("ok")
    captured_env = {}

    def fake_eval(**kwargs: object):
        captured_env.update(kwargs["env"])
        output_dir = kwargs["output_dir"]
        metrics_json = output_dir / "metrics.json"
        output_dir.mkdir(parents=True)
        metrics_json.write_text("{}")
        return metrics_json

    def fake_writeback(**kwargs: object):
        output_dir = kwargs["output_dir"]
        manifest_json = output_dir / "wandb_writeback_manifest.json"
        output_dir.mkdir(parents=True)
        manifest_json.write_text("{}")
        return manifest_json

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(olmo_sc, "run_olmo_eval", fake_eval)
    monkeypatch.setattr(olmo_sc, "run_writeback", fake_writeback)

    olmo_sc.run_one(
        Namespace(
            manifest=manifest,
            index=0,
            work_dir=tmp_path,
            olmo_eval_dir=tmp_path / "OLMo-Eval",
            writeback_script=tmp_path / "writeback.py",
            suite="full",
            limit=None,
            key_prefix="olmo_base_eval/easy_bpb",
            writeback_apply=True,
            cleanup_model=False,
            skip_existing=False,
            checkpoint_mode="local-only",
            hf_revision=None,
        )
    )

    assert captured_env["HF_HOME"] == str(tmp_path / "hf_home")
    assert captured_env["HUGGINGFACE_HUB_CACHE"] == str(tmp_path / "hf_home" / "hub")
    assert captured_env["HF_ALLOW_CODE_EVAL"] == "1"
    assert captured_env["UV_OFFLINE"] == "1"
    assert captured_env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert captured_env["HF_DATASETS_DISABLE_PROGRESS_BARS"] == "1"
    assert captured_env["TQDM_DISABLE"] == "1"
    assert captured_env["OLMO_EVAL_TASK_PREP_WORKERS"] == "1"


def test_run_one_does_not_skip_dry_run_when_apply_is_requested(tmp_path, monkeypatch, capsys):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]
    manifest = tmp_path / "manifest.csv"
    write_csv([row], manifest)
    model_dir = model_dir_for_row(row, tmp_path)
    model_dir.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        (model_dir / name).write_text("ok")
    output_dir = tmp_path / "outputs" / row.output_name
    writeback_dir = tmp_path / "wandb_writeback" / row.output_name
    status_dir = tmp_path / "status"
    output_dir.mkdir(parents=True)
    writeback_dir.mkdir(parents=True)
    status_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text("{}")
    (writeback_dir / "wandb_writeback_manifest.json").write_text("{}")
    (status_dir / f"{row.output_name}.json").write_text(json.dumps({"writeback_apply": False}))
    calls = {"eval": 0, "writeback": 0}

    def fake_eval(**kwargs: object):
        calls["eval"] += 1
        return kwargs["output_dir"] / "metrics.json"

    def fake_writeback(**kwargs: object):
        calls["writeback"] += 1
        return kwargs["output_dir"] / "wandb_writeback_manifest.json"

    monkeypatch.setattr(olmo_sc, "run_olmo_eval", fake_eval)
    monkeypatch.setattr(olmo_sc, "run_writeback", fake_writeback)

    olmo_sc.run_one(
        Namespace(
            manifest=manifest,
            index=0,
            work_dir=tmp_path,
            olmo_eval_dir=tmp_path / "OLMo-Eval",
            writeback_script=tmp_path / "writeback.py",
            suite="full",
            limit=None,
            key_prefix="olmo_base_eval/easy_bpb",
            writeback_apply=True,
            cleanup_model=False,
            skip_existing=True,
            checkpoint_mode="local-only",
            hf_revision=None,
        )
    )

    assert calls == {"eval": 1, "writeback": 1}
    assert "rerunning_existing_dry_run_for_apply" in capsys.readouterr().out


def test_run_one_skip_preserves_prior_apply_status(tmp_path, monkeypatch):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]
    manifest = tmp_path / "manifest.csv"
    write_csv([row], manifest)
    output_dir = tmp_path / "outputs" / row.output_name
    writeback_dir = tmp_path / "wandb_writeback" / row.output_name
    status_dir = tmp_path / "status"
    output_dir.mkdir(parents=True)
    writeback_dir.mkdir(parents=True)
    status_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text("{}")
    (writeback_dir / "wandb_writeback_manifest.json").write_text("{}")
    status_path = status_dir / f"{row.output_name}.json"
    status_path.write_text(json.dumps({"writeback_apply": True}))

    def fail_eval(**_kwargs: object) -> None:
        raise AssertionError("applied rows should be skipped")

    monkeypatch.setattr(olmo_sc, "run_olmo_eval", fail_eval)

    olmo_sc.run_one(
        Namespace(
            manifest=manifest,
            index=0,
            work_dir=tmp_path,
            olmo_eval_dir=tmp_path / "OLMo-Eval",
            writeback_script=tmp_path / "writeback.py",
            suite="full",
            limit=None,
            key_prefix="olmo_base_eval/easy_bpb",
            writeback_apply=True,
            cleanup_model=False,
            skip_existing=True,
            checkpoint_mode="local-only",
            hf_revision=None,
        )
    )

    status = json.loads(status_path.read_text())
    assert status["status"] == "skipped_existing"
    assert status["writeback_apply"] is True


def test_stage_checkpoints_skips_already_staged_files(tmp_path, monkeypatch, capsys):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]
    manifest = tmp_path / "manifest.csv"
    write_csv([row], manifest)
    model_dir = model_dir_for_row(row, tmp_path)
    model_dir.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        (model_dir / name).write_text("ok")

    def fail_snapshot(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("complete staged checkpoints should not call the Hub")

    monkeypatch.setattr(olmo_sc, "snapshot_download", fail_snapshot)

    stage_checkpoints(
        Namespace(
            manifest=manifest,
            work_dir=tmp_path,
            start_index=None,
            end_index=None,
            index=None,
            only_missing=True,
            max_workers=1,
            hf_revision=None,
        )
    )

    out = capsys.readouterr().out
    assert '"status": "already_staged"' in out
    assert '"staged_count": 1' in out


def test_stage_checkpoints_downloads_missing_files(tmp_path, monkeypatch, capsys):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]
    manifest = tmp_path / "manifest.csv"
    write_csv([row], manifest)
    calls = []

    def fake_snapshot_download(**kwargs: object) -> None:
        calls.append(kwargs)
        model_dir = Path(kwargs["local_dir"]) / row.hf_checkpoint_path
        model_dir.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
            (model_dir / name).write_text("ok")

    monkeypatch.setattr(olmo_sc, "snapshot_download", fake_snapshot_download)

    stage_checkpoints(
        Namespace(
            manifest=manifest,
            work_dir=tmp_path,
            start_index=None,
            end_index=None,
            index=None,
            only_missing=True,
            max_workers=1,
            hf_revision="abc123",
        )
    )

    assert calls == [
        {
            "repo_id": row.hf_repo_id,
            "repo_type": "dataset",
            "allow_patterns": [f"{row.hf_checkpoint_path}/**"],
            "local_dir": tmp_path / "downloaded_models" / row.output_name,
            "revision": "abc123",
        }
    ]
    out = capsys.readouterr().out
    assert '"status": "staged"' in out
    assert '"staged_count": 1' in out


def test_clean_incomplete_dataset_cache_is_dry_run_by_default(tmp_path):
    stale = tmp_path / "hf_home" / "datasets" / "foo" / "bar.incomplete"
    stale.mkdir(parents=True)

    dry_run = clean_incomplete_dataset_cache(tmp_path, apply=False)
    assert dry_run["incomplete_count"] == 1
    assert stale.exists()

    applied = clean_incomplete_dataset_cache(tmp_path, apply=True)
    assert applied["incomplete_count"] == 1
    assert not stale.exists()


def test_clean_incomplete_dataset_cache_handles_incomplete_files(tmp_path):
    stale = tmp_path / "hf_home" / "datasets" / "downloads" / "file.incomplete"
    stale.parent.mkdir(parents=True)
    stale.write_text("partial")

    applied = clean_incomplete_dataset_cache(tmp_path, apply=True)

    assert applied["incomplete_count"] == 1
    assert not stale.exists()


def test_offline_cache_report_requires_hub_datasets_modules_and_no_incomplete(tmp_path):
    report = offline_cache_report(tmp_path)
    assert report["ready"] is False
    assert set(report["missing_dirs"]) == {"hf_home", "hub", "datasets", "modules"}

    for rel in ("hf_home/hub", "hf_home/datasets", "hf_home/modules"):
        (tmp_path / rel).mkdir(parents=True)
    stale = tmp_path / "hf_home" / "datasets" / "task.incomplete"
    stale.mkdir()
    report = offline_cache_report(tmp_path)
    assert report["ready"] is False
    assert report["missing_dirs"] == []
    assert report["incomplete_count"] == 1

    shutil.rmtree(stale)
    report = offline_cache_report(tmp_path)
    assert report["ready"] is False
    assert report["basic_skills_missing"]

    snapshot = write_basic_skills_snapshot(tmp_path)
    report = offline_cache_report(tmp_path)
    assert report["ready"] is True
    assert report["basic_skills_root"] == str(snapshot)


def test_prewarm_datasets_sets_shared_cache_environment(tmp_path, monkeypatch):
    row = rows_from_upload_manifest(
        upload_manifest_text([sample_upload_row()]),
        hf_repo_id="Calvin-Xu/checkpoints",
        start_index=0,
    )[0]
    manifest = tmp_path / "manifest.csv"
    write_csv([row], manifest)
    model_dir = model_dir_for_row(row, tmp_path)
    model_dir.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        (model_dir / name).write_text("ok")
    captured_env = {}

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def fake_snapshot_download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == "allenai/basic-skills"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["cache_dir"] == str(tmp_path / "hf_home" / "hub")
        assert sorted(kwargs["allow_patterns"]) == sorted(
            f"{subset}/validation.json" for subset in olmo_sc.BASIC_SKILLS_SUBTASKS
        )
        return str(write_basic_skills_snapshot(tmp_path))

    def fake_eval(**kwargs: object):
        captured_env.update(kwargs["env"])
        output_dir = kwargs["output_dir"]
        metrics_json = output_dir / "metrics.json"
        output_dir.mkdir(parents=True)
        metrics_json.write_text("{}")
        return metrics_json

    monkeypatch.setattr(olmo_sc, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(olmo_sc, "run_olmo_eval", fake_eval)

    prewarm_datasets(
        Namespace(
            manifest=manifest,
            index=0,
            work_dir=tmp_path,
            olmo_eval_dir=tmp_path / "OLMo-Eval",
            output_dir=None,
            suite="full",
            limit=None,
            checkpoint_mode="local-only",
            hf_revision=None,
        )
    )

    assert captured_env["HF_HOME"] == str(tmp_path / "hf_home")
    assert captured_env["HF_DATASETS_CACHE"] == str(tmp_path / "hf_home" / "datasets")
    assert captured_env["HF_MODULES_CACHE"] == str(tmp_path / "hf_home" / "modules")
    assert captured_env["HF_ALLOW_CODE_EVAL"] == "1"
    assert captured_env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert captured_env["HF_DATASETS_DISABLE_PROGRESS_BARS"] == "1"
    assert captured_env["TQDM_DISABLE"] == "1"
    assert captured_env["OLMO_EVAL_TASK_PREP_WORKERS"] == "1"
    assert "HF_HUB_OFFLINE" not in captured_env
    assert "HF_DATASETS_OFFLINE" not in captured_env
    assert "TRANSFORMERS_OFFLINE" not in captured_env


def test_worker_environment_uses_uv_offline_only_when_hf_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    online = worker_environment(tmp_path)
    assert "UV_OFFLINE" not in online

    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    offline = worker_environment(tmp_path)
    assert offline["UV_OFFLINE"] == "1"


def test_smoke_task_uses_true_bpb_variant_order():
    assert olmo_sc.SMOKE_TASKS == ("arc_easy:olmo3base:bpb",)


def test_csqa_bpb_suite_uses_true_bpb_variant_order():
    assert olmo_sc.task_args("csqa_bpb") == ["-t", "csqa:olmo3base:bpb"]


def test_olmo_eval_fanout_patcher_is_idempotent(tmp_path):
    olmo_eval_dir = tmp_path / "OLMo-Eval"
    runner = olmo_eval_dir / "src" / "olmo_eval" / "runners" / "asynq" / "runner.py"
    basic = olmo_eval_dir / "src" / "olmo_eval" / "evals" / "tasks" / "basic_skills.py"
    olmobase = olmo_eval_dir / "src" / "olmo_eval" / "evals" / "suites" / "olmobase.py"
    runner.parent.mkdir(parents=True)
    basic.parent.mkdir(parents=True)
    olmobase.parent.mkdir(parents=True)
    runner.write_text(
        olmo_patch.RUNNER_IMPORT_OLD + "\n" + olmo_patch.RUNNER_CONSTANT_OLD + "\n" + olmo_patch.RUNNER_EXECUTOR_OLD
    )
    basic.write_text(
        olmo_patch.BASIC_IMPORT_OLD
        + "\n"
        + olmo_patch.BASIC_CONSTANT_OLD
        + "\n"
        + olmo_patch.BASIC_LOAD_OLD
        + "\n"
        + olmo_patch.BASIC_DATASOURCE_OLD
    )
    olmobase.write_text("""
make_suite(
    name="arc:bpb:olmo3base",
    tasks=("arc_challenge:bpb:olmo3base", "arc_easy:bpb:olmo3base"),
)

make_suite(
    name="olmobase:easy:qa:bpb",
    tasks=(
        get_suite("arc:bpb:olmo3base"),
        "csqa:bpb:olmo3base",
        "piqa:bpb:olmo3base",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)
""")

    dry_run = olmo_patch.apply_patches(olmo_eval_dir, dry_run=True)
    assert dry_run == {str(runner): True, str(basic): True, str(olmobase): True}
    assert "OLMO_EVAL_TASK_PREP_WORKERS" not in runner.read_text()

    first = olmo_patch.apply_patches(olmo_eval_dir)
    second = olmo_patch.apply_patches(olmo_eval_dir)

    assert first == {str(runner): True, str(basic): True, str(olmobase): True}
    assert second == {str(runner): False, str(basic): False, str(olmobase): False}
    assert "OLMO_EVAL_TASK_PREP_WORKERS" in runner.read_text()
    assert "OLMO_EVAL_BASIC_SKILLS_LOCAL_ROOT" in basic.read_text()
    olmobase_text = olmobase.read_text()
    assert '"arc_challenge:olmo3base:bpb"' in olmobase_text
    assert '"arc_easy:olmo3base:bpb"' in olmobase_text
    assert '"csqa:olmo3base:bpb"' in olmobase_text
    assert '"piqa:olmo3base:bpb"' in olmobase_text


def test_safe_segment_removes_path_unfriendly_characters():
    assert safe_segment("abc/def:g h") == "abc_def_g_h"
