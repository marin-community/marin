# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import runpy
import shutil
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlencode, urlparse, urlsplit

import click
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from finestore.rl.run_records import AttemptRecord, read_attempt, read_evaluations, write_attempt
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from levanter import model_cache
from marin.execution.artifact import RECORD_FILENAME, Artifact, ArtifactRecord, result_type_name
from marin.execution.lazy import OUT, ArtifactStep, apply, materialized_config, run
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel
from marin.rl.skyrl_records import read_skyrl_metrics
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance

from experiments.post_training import rl as snowball
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_recipe as recipe
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_command import grafana_url
from experiments.post_training.rl_tests.fixtures import (
    SELECTION_IDENTITY,
    append_metrics,
    controls,
    copy_evaluations,
    edit_receipt,
    replace_evaluations,
    write_receipt,
)
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.skyrl_test_support import assert_revisioned_native_configs, native_json
from experiments.post_training.snowball_sft import OPENTHOUGHTS_32K_4NODE, OPENTHOUGHTS_SPEC
from experiments.post_training.task_curation.selection import FULL_VALIDATION_FILENAME, SELECTION_PREFIX

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]


def printed_status(output):
    marker = output.rfind("\npreset:")
    return yaml.safe_load(output[marker + 1 if marker >= 0 else 0 :].split("\n# ", 1)[0])


def test_all_presets_plan_with_their_declared_controls(selection):
    table = snowball_model.SNOWBALL
    for preset in table.presets:
        arguments = ["--preset", preset.name, "--version", preset.revision, "--plan"]
        if preset.tasks_input == "required":
            arguments += ["--tasks", SELECTION_IDENTITY]
        result = CliRunner().invoke(snowball.command(table, argv=tuple(arguments)), arguments)
        assert result.exit_code == 0, (preset.name, result.output, result.exception)

    preset = next(preset for preset in table.presets if preset.gated)
    arguments = ["--preset", preset.name, "--version", preset.revision]
    if preset.tasks_input == "required":
        arguments += ["--tasks", SELECTION_IDENTITY]
    unpinned = replace(table, controls={})
    result = CliRunner().invoke(snowball.command(unpinned, argv=tuple(arguments)), arguments)
    assert result.exit_code == 4 and "no control pin" in result.output, (result.output, result.exception)
    assert "--issue ISSUE_NUMBER --run" in result.output


def test_preset_variants_compose(selection, tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    def export_model(output):
        destination = Path(output) / "hf"
        destination.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json"):
            (destination / name).write_text("{}")
        return Artifact(path=output)

    producer = apply("tests/computed-init", export_model, version="2026.11.17", output=OUT)
    own_model = ArtifactHfModel(producer)
    train = snowball_model.SNOWBALL.by_name["train-long"]
    dev_sft = replace(OPENTHOUGHTS_32K_4NODE, recipe=replace(OPENTHOUGHTS_SPEC, version="dev"))
    dev_model = dev_sft.model()

    with pytest.raises(click.BadParameter, match="the initializer requires an immutable artifact version"):
        snowball.build_run(train, init=dev_model, table=snowball_model.SNOWBALL)

    def reject_export_read(*args, **kwargs):
        raise AssertionError("planning read an unbuilt export")

    with monkeypatch.context() as planning:
        planning.setattr(skyrl_api, "discover_hf_checkpoints", reject_export_read)
        planned = snowball.build_run(train, init=own_model, table=snowball_model.SNOWBALL)
        assert producer in planned.runs[0].deps
        planned.runs[0].lower()
        assert (
            snowball.main(
                snowball_model.SNOWBALL, argv=["--preset", train.name, "--version", train.revision], init=own_model
            )
            == 0
        )
    assert not Path(producer.path()).exists()
    run(producer)
    document = yaml.safe_load(materialized_config(planned.runs[0], str(tmp_path)).launch_config_yaml)
    assert document["inputs"]["model"]["uri"] == str(Path(producer.path()) / "hf")
    assert document["inputs"]["model"]["tokenizer_uri"] is None
    assert document["inputs"]["model"]["tokenizer_revision"] is None
    best = next(row for row in document["skyrl"]["trainer"]["callbacks"] if row["type"] == "best_checkpoint")
    assert best["initial_model_uri"] == document["inputs"]["model"]["uri"]
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    learning = snowball_model.SNOWBALL.by_name["learning-medium"]
    changed = snowball.build_run(
        learning,
        settings=(
            'trainer={"policy_mini_batch_size":64,"train_batch_size":64,"bf16":false,"algorithm":{"batch_invariant":true}}',
            "generator.model_dtype=float16",
            "generator.n_samples_per_prompt=8",
            "trainer.micro_train_batch_size_per_gpu=2",
            "trainer.policy.optimizer_config.lr=2e-6",
            "generator.engine_init_kwargs.probe={}",
        ),
        table=snowball_model.SNOWBALL,
    ).runs[0]
    promoted = replace(
        learning,
        recipe=replace(
            learning.recipe,
            config=learning.recipe.config.with_settings(
                (
                    "trainer.train_batch_size=64",
                    "trainer.policy_mini_batch_size=64",
                    "generator.n_samples_per_prompt=8",
                    "trainer.micro_train_batch_size_per_gpu=2",
                    "trainer.policy.optimizer_config.lr=2e-6",
                )
            ),
        ),
    )
    checkpoint = StoragePath("s3://marin-test/models/checkpoint")
    export = checkpoint / "hf"
    files = {str(export / name).removeprefix("s3://") for name in ("config.json", "tokenizer_config.json")}
    monkeypatch.setattr("s3fs.S3FileSystem.glob", lambda self, path, **kwargs: [str(export / "config.json")])
    monkeypatch.setattr("s3fs.S3FileSystem.modified", lambda self, path: datetime(2026, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(
        "s3fs.S3FileSystem.exists",
        lambda self, path, **kwargs: str(path).removeprefix("s3://") in files | {str(checkpoint).removeprefix("s3://")},
    )
    metadata = {"train_config": {"data": {"tokenizer": "tests/own-tokenizer"}}}
    monkeypatch.setattr(model_cache, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="a" * 40))
    initializer = ArtifactStep.adopt(
        "tests/checkpoint", "2026.11.10", str(checkpoint), kind=LevanterCheckpoint, config=metadata
    )
    alias = tmp_path / initializer.name / initializer.version
    alias.mkdir(parents=True)
    (alias / RECORD_FILENAME).write_text(
        ArtifactRecord(
            name=initializer.name,
            version=initializer.version,
            result_type=result_type_name(LevanterCheckpoint),
            config=metadata,
            source=str(checkpoint),
        ).model_dump_json()
    )
    cli_initializer = recipe.published_model(f"{initializer.name}@{initializer.version}", preset=learning)
    published = []
    for suffix, config in (
        ("own", {}),
        (
            "explicit",
            {
                "tokenizer_uri": "tests/top-tokenizer",
                "tokenizer_revision": "top-revision",
                "train_config": {"data": {"tokenizer": "tests/nested-ignored"}},
            },
        ),
    ):
        name = "tests/checkpoint-" + suffix
        record = tmp_path / name / initializer.version / RECORD_FILENAME
        record.parent.mkdir(parents=True)
        record.write_text(
            ArtifactRecord(
                name=name,
                version=initializer.version,
                result_type=result_type_name(Artifact),
                config=config,
                source=str(checkpoint),
            ).model_dump_json()
        )
        published.append(recipe.published_model(f"{name}@{initializer.version}", preset=learning))

    revisioned = replace(snowball_model.SNOWBALL.by_name["speed-short"], revision="2026.11.12.2")
    steps = (
        changed,
        snowball.build_run(promoted, table=snowball_model.SNOWBALL).runs[0],
        snowball.build_run(
            learning,
            init=ArtifactHfModel(initializer, "tests/own-tokenizer", "tokenizer-revision"),
            table=snowball_model.SNOWBALL,
        ).runs[0],
        snowball.build_run(
            snowball_model.SNOWBALL.by_name["train-long"],
            settings=("trainer.max_steps=2",),
            table=snowball_model.SNOWBALL,
        ).runs[0],
        snowball.build_run(revisioned, control=True, table=snowball_model.SNOWBALL).runs[0],
        snowball.build_run(learning, init=cli_initializer, table=snowball_model.SNOWBALL).runs[0],
        *(snowball.build_run(learning, init=model, table=snowball_model.SNOWBALL).runs[0] for model in published),
    )
    paths = []
    for index, step in enumerate(steps):
        path = tmp_path / f"variant-{index}.yaml"
        path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)
        paths.append(str(path))
    configs = gate.compose_configs(paths)
    for config in configs[:2]:
        trainer, generator = config["skyrl"]["trainer"], config["skyrl"]["generator"]
        assert (
            trainer["train_batch_size"],
            trainer["policy_mini_batch_size"],
            generator["n_samples_per_prompt"],
            trainer["micro_train_batch_size_per_gpu"],
            trainer["micro_forward_batch_size_per_gpu"],
            trainer["rollout_buffer"]["max_in_flight"],
            trainer["policy"]["optimizer_config"]["lr"],
        ) == (64, 64, 8, 2, 2, 64, 2e-6)
        assert (
            trainer["policy"]["optimizer_config"]["weight_decay"],
            trainer["policy"]["optimizer_config"]["adam_betas"],
        ) == (0, [0.9, 0.999])
    trainer, generator = configs[0]["skyrl"]["trainer"], configs[0]["skyrl"]["generator"]
    assert (
        trainer["bf16"],
        generator["model_dtype"],
        trainer["algorithm"]["batch_invariant"],
        generator["engine_init_kwargs"]["probe"],
    ) == (False, "float16", True, {})
    assert configs[2]["inputs"]["model"]["uri"] == str(export)
    assert configs[2]["iris"]["wandb_entity"] == "test-team"
    for config, revision in ((configs[2], "tokenizer-revision"), (configs[5], "a" * 40)):
        assert (config["inputs"]["model"]["tokenizer_uri"], config["inputs"]["model"]["tokenizer_revision"]) == (
            "tests/own-tokenizer",
            revision,
        )
        assert config["inputs"]["model"]["uri"] == str(export)
    assert (configs[6]["inputs"]["model"]["tokenizer_uri"], configs[6]["inputs"]["model"]["tokenizer_revision"]) == (
        None,
        None,
    )
    assert (configs[7]["inputs"]["model"]["tokenizer_uri"], configs[7]["inputs"]["model"]["tokenizer_revision"]) == (
        "tests/top-tokenizer",
        "top-revision",
    )
    callbacks = configs[3]["skyrl"]["trainer"]["callbacks"]
    assert next(row for row in callbacks if row["type"] == "evaluation")["eval_on_train_end"] is True
    assert configs[3]["skyrl"]["trainer"]["max_steps"] == 2
    assert configs[4]["inputs"]["validation_data"][0]["identity"].partition(":")[0].endswith("@" + revisioned.revision)
    summary = Path(recipe.selection_artifact(learning.tasks).path()) / recipe.SUMMARY_FILENAME
    content = json.loads(summary.read_text())
    content["grader_kind_counts"]["train"]["remote_judge"] = 1
    summary.write_text(json.dumps(content))
    with pytest.raises(click.BadParameter, match="AI-judge"):
        snowball.build_run(learning, table=snowball_model.SNOWBALL)


def test_presets_match_revisioned_native_configs(selection, tmp_path, monkeypatch):
    revisions = {
        "speed-short": {"2026.11.13.1": "96e0d088", "2026.11.17.1": "363225a9", "2026.11.19.1": "371233d9"},
        "data-short": {"2026.11.13.1": "7d9a590f", "2026.11.17.1": "772467a8", "2026.11.19.1": "cb07d91c"},
        "speed-medium": {"2026.11.12.1": "e75349ff", "2026.11.17.1": "5cd852f4", "2026.11.19.1": "f8281b9d"},
        "learning-medium": {"2026.11.12.1": "ba638a12", "2026.11.17.1": "2be6e5e2", "2026.11.19.1": "4aa3aed9"},
        "data-medium": {"2026.11.12.1": "ba638a12", "2026.11.17.1": "2be6e5e2", "2026.11.19.1": "4aa3aed9"},
        "speed-long": {"2026.11.11.1": "cd0a5f4e", "2026.11.17.1": "f29155cc", "2026.11.19.1": "9edc7b70"},
        "train-long": {"2026.11.11.1": "4945b701", "2026.11.17.1": "b411c765", "2026.11.19.1": "e56a62fa"},
    }
    configs = assert_revisioned_native_configs(
        snowball_model.SNOWBALL, revisions, tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    for name, config in zip(revisions, configs, strict=True):
        trainer, generator = config["skyrl"]["trainer"], config["skyrl"]["generator"]
        assert config["iris"]["wandb_entity"] == "marin-community"
        assert config["iris"]["allocation"]["num_nodes"] * config["iris"]["allocation"]["gpus_per_node"] <= 40
        generating = name in ("speed-short", "data-short")
        assert (
            config["runtime"]["training_type"],
            trainer["rollout_buffer"]["max_staleness_steps"],
            trainer["algorithm"]["off_policy_correction"],
            trainer["rollout_buffer"]["batch_policy"],
        ) == ((None, 0, "none", "full_batch") if generating else ("async", 2, "tis", "full_batch"))
        if generating:
            assert config["runtime"]["entrypoint"] == "skyrl_train.entrypoints.main_generate"
            assert config["iris"]["allocation"]["num_nodes"] == 1
            assert config["skyrl"]["trajectory_runner"]["max_concurrent_tasks"] == (
                2048 if name == "speed-short" else 1024
            )
            assert trainer["eval_interval"] > 0 and config["inputs"]["validation_data"]
            sampling = generator["eval_sampling_params"]
            assert (sampling["temperature"], sampling["max_generate_length"], sampling["ignore_eos"]) == (
                1,
                4096 if name == "speed-short" else 8192,
                name == "speed-short",
            )
        else:
            assert generator["apply_overlong_filtering"] is False
        assert (trainer["bf16"], generator["model_dtype"], trainer["algorithm"]["batch_invariant"]) == (
            True,
            "bfloat16",
            False,
        )
        if name in ("speed-short", "learning-medium", "data-medium", "train-long"):
            profile = next(
                (
                    row["additional_evaluations"]["eval_lite"]
                    for row in trainer["callbacks"]
                    if "eval_lite" in row.get("additional_evaluations", {})
                ),
                None,
            )
            for source in config["inputs"].get("evaluation_data", {}).get("eval_lite", []):
                assert (
                    source["identity"].partition(":")[0]
                    == f"{SELECTION_PREFIX}/{snowball_model.SNOWBALL.by_name[name].eval_lite}"
                )
                assert source["relative_path"] == "full_validation.parquet"
            sampling = profile["sampling_params"] if profile else generator["eval_sampling_params"]
            assert (
                sampling["temperature"],
                sampling["ignore_eos"],
                sampling["max_generate_length"],
                profile["n_samples_per_prompt"] if profile else generator["eval_n_samples_per_prompt"],
            ) == (0, False, 8192, 1)
        if name in ("learning-medium", "data-medium"):
            assert next(row for row in trainer["callbacks"] if row["type"] == "evaluation")["eval_before_train"] is True
            assert config["run"]["export_hf"] and "hf_model_save" in [row["type"] for row in trainer["callbacks"]]
        if name == "speed-long":
            assert not config["inputs"]["validation_data"] and not config["run"]["export_hf"]
        if name == "train-long":
            kinds = [row["type"] for row in trainer["callbacks"]]
            assert kinds.index("evaluation") < kinds.index("best_checkpoint")
            assert next(row for row in trainer["callbacks"] if row["type"] == "evaluation")["eval_on_train_end"] is False


def test_run_identity_tracks_code_inputs_and_replicates(selection, monkeypatch, tmp_path, recorded_runs):
    learning = snowball_model.SNOWBALL.by_name["learning-medium"]
    data = snowball_model.SNOWBALL.by_name["data-short"]
    with pytest.raises(click.BadParameter, match="accepts no --set"):
        snowball.build_run(
            replace(learning, accepts_settings=False),
            settings=("trainer.policy.optimizer_config.lr=2e-6",),
            table=snowball_model.SNOWBALL,
        )

    def run(preset=learning, *settings, **kwargs):
        return snowball.build_run(preset, settings=settings, **kwargs, table=snowball_model.SNOWBALL).runs[0]

    def name(preset=learning, *settings, **kwargs):
        return run(preset, *settings, **kwargs).name

    assert name(learning, "generator.engine_init_kwargs.probe={}") != name()
    repeated = snowball.build_run(learning, replicates=3, table=snowball_model.SNOWBALL)
    assert [step.name for step in snowball.build_run(learning, replicates=2, table=snowball_model.SNOWBALL).runs] == [
        step.name for step in repeated.runs[:2]
    ]
    assert len({step.name for step in repeated.runs}) == 3
    initializer = ArtifactStep.adopt("tests/initializer", "2026.10.26", selection.path(), kind=LevanterCheckpoint)
    task_set_name = snowball_model.DEFAULT_EVAL_LITE_NAME
    task_set_version = snowball_model.DEFAULT_EVAL_LITE_VERSION
    new_task_set = tmp_path / task_set_name / "2026.11.05"
    new_task_set.mkdir(parents=True)
    (new_task_set / "summary.json").write_bytes(
        (tmp_path / task_set_name / task_set_version / "summary.json").read_bytes()
    )
    changed_task_set = replace(
        learning,
        eval_lite=f"{task_set_name}@2026.11.05",
    )
    pairs = [
        (name(), name(learning, "trainer.policy.optimizer_config.lr=2e-6")),
        (name(data, tasks=selection), name(data, tasks=replace(selection, version="2026.10.27"))),
        (name(), name(init=ArtifactHfModel(initializer, "tests/own-tokenizer", "tokenizer-revision", relative_path=""))),
        (name(), name(changed_task_set)),
    ]
    assert all(left != right for left, right in pairs), pairs
    assert name(
        learning,
        'trainer.policy.optimizer_config={"lr": 2e-06}',
        'trainer.policy.optimizer_config={"weight_decay": 0.1}',
    ) == name(
        learning, "trainer.policy.optimizer_config.lr=0.000002", "trainer.policy.optimizer_config.weight_decay=0.1"
    )
    assert name(learning, "trainer.policy.optimizer_config.lr=2e-6") == name(
        learning, "trainer.policy.optimizer_config.lr=0.000002"
    )
    controls(recorded_runs, "learning-medium")
    first = snowball.build_run(learning, requirements=("eval_lite/mean>=0",), table=snowball_model.SNOWBALL)
    second = snowball.build_run(learning, requirements=("eval_lite/mean>=0.1",), table=snowball_model.SNOWBALL)
    assert first.runs[0].name == second.runs[0].name == name()
    assert first.gate.name != second.gate.name
    original_name = name()
    monkeypatch.setenv("WANDB_ENTITY", "another-team")
    assert name() == original_name
    monkeypatch.setattr(
        Provenance, "capture", lambda *args, **kwargs: Provenance("tree-two", "commit-one", False, "branch", "test")
    )
    assert name() != original_name

    summary_path = Path(selection.path()) / "summary.json"
    summary = json.loads(summary_path.read_text())
    original_exclusions = summary["recipe"]["exclusions"]
    heldout = summary["recipe"]["heldout"]
    summary["recipe"]["heldout"] = [{"version": "different-development-input"}]
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(click.BadParameter, match="different heldout"):
        snowball.build_run(data, tasks=selection, table=snowball_model.SNOWBALL)
    summary["recipe"]["heldout"] = heldout
    summary_path.write_text(json.dumps(summary))
    for current in (data, snowball_model.SNOWBALL.by_name["data-medium"]):
        selected = replace(current, eval_lite=f"{task_set_name}@2026.11.05")
        with pytest.raises(click.BadParameter, match="full EvalLite"):
            snowball.build_run(selected, tasks=selection, table=snowball_model.SNOWBALL)
        summary["recipe"]["exclusions"] = [
            {"uri": str(new_task_set / FULL_VALIDATION_FILENAME), "version": "2026.11.05"}
        ]
        summary_path.write_text(json.dumps(summary))
        assert snowball.build_run(selected, tasks=selection, table=snowball_model.SNOWBALL).gate is not None
        summary["recipe"]["exclusions"] = original_exclusions
        summary_path.write_text(json.dumps(summary))
    summary["recipe"]["exclusions"] = []
    summary_path.write_text(json.dumps(summary))
    for tasks in (recipe.selection_artifact(snowball_model.SNOWBALL.by_name["learning-medium"].tasks), selection):
        with pytest.raises(click.BadParameter, match="full EvalLite"):
            snowball.build_run(data, tasks=tasks, table=snowball_model.SNOWBALL)


def test_grafana_url_selects_native_run_and_utc_window():
    result = urlparse(grafana_url("users/test/run", "marin-east", "rl", "2026-11-06T18:40:00Z", "2026-11-06T19:40:00Z"))
    assert result.path == "/d/marin-rl-runs/rl-post-training"
    assert parse_qs(result.query) == {
        "orgId": ["1"],
        "timezone": ["utc"],
        "from": ["1793990400000"],
        "to": ["1793994000000"],
        "var-training_type": ["rl"],
        "var-cluster": ["marin-east"],
        "var-run": ["users/test/run"],
        "var-job": ["$__all"],
        "var-execution": ["$__all"],
    }
    open_window = parse_qs(urlparse(grafana_url("generation", "marin-east", None, "2026-11-06T18:40:00Z")).query)
    assert open_window["to"] == ["now"] and open_window["var-training_type"] == ["sync"]


@pytest.mark.parametrize(
    "case,expected",
    [
        ("fail", 1),
        ("invalid", 2),
        ("missing", 3),
        ("native-failed", 2),
        ("child", 0),
        ("changed-source", 0),
        ("stale-control", 4),
        ("candidate-compose", 4),
        ("malformed-control", 4),
        ("mixed-native-failed", 2),
        ("control-missing-receipt", 3),
        ("partial-retry", 0),
        ("data-short", 0),
        ("compose-timeout", 4),
        ("unexpected", 3),
        ("missing-requirement", 4),
        ("missing-issue", 4),
        ("ambiguous-issue", 4),
    ],
)
def test_submission_exit_codes_and_publication_retries(
    recorded_runs, selection, tmp_path, monkeypatch, preset_table, case, expected
):
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    publication_case = case in {
        "changed-source",
        "child",
        "mixed-native-failed",
        "control-missing-receipt",
        "partial-retry",
        "data-short",
    }
    preset_name = "data-short" if case == "data-short" else "learning-medium" if publication_case else "speed-medium"
    preset = snowball_model.SNOWBALL.by_name[preset_name]
    if case == "child":
        model = "s3://marin-test/models/child-model"
        preset = replace(
            preset,
            model=replace(
                preset.model,
                step=ArtifactStep.adopt(
                    preset.model.step.name, preset.model.step.version, str(model), kind=preset.model.step.artifact_type
                ),
            ),
        )
    preset_table(preset)
    if case == "child":
        monkeypatch.setattr(
            "s3fs.S3FileSystem.exists",
            lambda self, path, **kwargs: str(path).removeprefix("s3://")
            in {"marin-test/models/python-init", "marin-test/models/child-model"},
        )
    roots = controls(recorded_runs, preset.name) if preset.gated and case != "compose-timeout" else []
    replicas = 2 if case in {"partial-retry", "mixed-native-failed", "control-missing-receipt"} else 1
    control = case == "control-missing-receipt"
    built = snowball.build_run(
        preset,
        table=snowball_model.SNOWBALL,
        control=control,
        replicates=replicas,
        tasks=selection if case == "data-short" else None,
    )
    sources = (
        []
        if case in {"candidate-compose", "compose-timeout"}
        else [
            recorded_runs(
                preset.name,
                training.name,
                speed=150 if case == "fail" else 101,
                score=0.5,
                fingerprint=training.fingerprint(),
                inputs=built.inputs,
                attempt_id=f"native-{index}",
            )
            for index, training in enumerate(built.runs)
        ]
    )
    source = sources[0] if sources else None
    for index, root in enumerate(sources):
        native = read_attempt(str(root / "exports"), f"native-{index}")
        write_attempt(str(root / "exports"), AttemptRecord(run_id=native.run_id, attempt_id=native.attempt_id, issue=73))
    if case == "data-short":
        receipt = json.loads((source / "terminal.json").read_text())
        descriptor = receipt["config"]["inputs"]["validation_data"][0]
        data_path = Path(descriptor["uri"]) / descriptor["relative_path"]
        rows = pq.read_table(data_path).to_pylist()
        for row in rows:
            task = json.loads(row["task_spec"])
            task["id"] = "candidate-" + task["id"]
            row["task_spec"] = json.dumps(task)
        pq.write_table(pa.Table.from_pylist(rows), data_path)
        evaluations = read_evaluations(str(source / "exports"), attempt_id="native-0", evaluation="eval/")
        replace_evaluations(
            source, [row.model_copy(update={"task_id": "candidate-" + row.task_id}) for row in evaluations]
        )
    if case in {"native-failed", "invalid"} or case.startswith(("malformed-", "stale-control")):
        root = roots[0] if case == "malformed-control" else roots[-1] if case.startswith("stale-control") else source
        with edit_receipt(root) as receipt:
            if case in {"native-failed"}:
                receipt["result"]["state"] = "failed"
            elif case in {"invalid"}:
                receipt["result"]["provenance"] = None
            elif case.startswith("malformed-"):
                receipt["config"]["skyrl"]["trainer"]["seed"] = None
            else:
                receipt["config"]["skyrl"]["trainer"]["policy"]["optimizer_config"]["lr"] = 5e-06
        if case == "native-failed":
            (source / "terminal.json").unlink()
    pending = {}
    if publication_case:
        for root in sources:
            for filename in ("terminal.json", "resolved-launch.yaml"):
                path = root / filename
                pending[path] = path.read_bytes()
                path.unlink()
    phase = {"retry": False}
    if case == "compose-timeout":
        real_run = subprocess.run

        def timeout_native(command, *args, **kwargs):
            if any("launch_config.load_launch_config" in str(value) for value in command):
                raise subprocess.TimeoutExpired(command, 120)
            return real_run(command, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", timeout_native)
    if case == "unexpected":
        path = Path(recipe.selection_artifact(preset.tasks).path()) / recipe.SUMMARY_FILENAME
        summary = json.loads(path.read_text())
        summary.pop("grader_kind_counts")
        path.write_text(json.dumps(summary))

    def assert_submission_link():
        output = click.get_binary_stream("stdout").getvalue().decode()
        urls = [
            line.split("run: ", 1)[1].strip() for line in output.splitlines() if "run: https://grafana.oa.dev/" in line
        ]
        query = next(parse_qs(urlparse(url).query) for url in urls if parse_qs(urlparse(url).query)["to"] == ["now"])
        assert query["var-run"] == [built.runs[0].name + "-" + preset.revision]
        assert query["var-cluster"] == [preset.recipe.layout.cluster]
        assert int(query["from"][0]) > 0

    def finish(*args, **kwargs):
        if case != "child":
            assert_submission_link()
        for path, data in pending.items():
            path.write_bytes(data)
        if case == "mixed-native-failed":
            receipt = json.loads((source / "terminal.json").read_text())
            if not phase["retry"]:
                receipt["result"]["state"] = "failed"
                write_receipt(source, receipt)
                (source / "terminal.json").unlink()
                return SimpleNamespace(state=JobState.FAILED)
            metrics = read_skyrl_metrics(str(source), attempt_id="native-0")
            receipt["config"]["run"]["attempt_id"] = "native-retry"
            write_receipt(source, receipt)
            write_attempt(
                str(source / "exports"),
                AttemptRecord(run_id=receipt["config"]["run"]["id"], attempt_id="native-retry", issue=74),
            )
            copy_evaluations(source, "native-0", "native-retry")
            for row in metrics.values():
                row["attempt_id"] = "native-retry"
            append_metrics(source, metrics)
        if control and (not phase["retry"]):
            (source / "terminal.json").unlink()
            (source / "resolved-launch.yaml").unlink()
            return SimpleNamespace(state=JobState.FAILED)
        if case not in {"missing", "native-failed"} and built.gate is not None:
            run(gate.completed_gate(built.gate))
        if case == "changed-source":
            monkeypatch.setattr(
                Provenance,
                "capture",
                lambda *args, **kwargs: Provenance(
                    "tree-edited-before-publication", "commit-one", True, "branch", "test"
                ),
            )
        return SimpleNamespace(state=JobState.FAILED if case in {"native-failed"} else JobState.SUCCEEDED)

    client = MagicMock(spec=IrisClient)
    job = JobName.from_wire("/test-user/dashboard-coordinator")
    client.submit.return_value.job_id = job
    client.job.return_value.wait.side_effect = finish
    if case.startswith(("malformed-", "stale-control")) or case in {
        "candidate-compose",
        "missing-issue",
        "ambiguous-issue",
    }:
        client.submit.side_effect = AssertionError("invalid stored inputs reached the coordinator RPC")
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: nullcontext(client))
    published = []
    unavailable = {"value": case in {"partial-retry"}}

    def publish(record, service_url=None):
        if unavailable["value"] and record.attempt_id == "native-1":
            raise RuntimeError("Credential unavailable; run iris login")
        published.append(record.model_dump(mode="json"))
        return (
            (service_url or "https://progress.test")
            + "/#"
            + urlencode({"run": record.run_id, "attempt": record.attempt_id})
        )

    monkeypatch.setattr("experiments.post_training.rl_publication.publish_result", publish)
    if case == "child":
        finish()
        monkeypatch.setattr(
            skyrl_api, "_run_launcher", lambda *args: pytest.fail("Cached coordinator run invoked native execution")
        )
    arguments = ["--preset", preset.name, "--version", preset.revision, "--run"]
    if case != "missing-issue":
        arguments.extend(["--issue", "73"])
    if case == "ambiguous-issue":
        arguments.append("--no-issue")
    if case != "unexpected":
        arguments.append("--wait")
    if "candidate-compose" in case:
        arguments.extend(["--set", "trainer.algorithm.policy_loss_type=gspo"])
    if case == "missing-requirement":
        arguments.extend(["--require", "typo>=0"])
    if control:
        arguments.append("--control")
    if case == "data-short":
        arguments.extend(["--tasks", SELECTION_IDENTITY])
    if replicas == 2:
        arguments.extend(["--replicates", "2"])
    selected = snowball.command(
        snowball_model.SNOWBALL, argv=tuple(arguments), entry_module="experiments.post_training.snowball_rl"
    )
    invoked = arguments
    if case == "child":

        def reenter(**kwargs):
            with monkeypatch.context() as child:
                for key, value in kwargs["environment"].env_vars.items():
                    child.setenv(key, value)
                child.setenv("IRIS_TASK_ID", "/test-user/coordinator/0")
                child.setattr(client, "submit", lambda **kwargs: pytest.fail("Coordinator submitted another job"))
                child.setattr(
                    "experiments.post_training.rl_publication.publish_result",
                    lambda *args: pytest.fail("Coordinator contacted Marina"),
                )
                info = JobInfo(task_id=JobName.from_wire("/test-user/coordinator/0"))
                with iris_ctx_scope(IrisContext.from_job_info(info)):
                    completed = CliRunner().invoke(selected, invoked)
                assert completed.exit_code == 0, (completed.output, completed.exception)
            set_job_info(None)
            return SimpleNamespace(job_id=job)

        client.submit.side_effect = reenter
    set_job_info(None)
    result = CliRunner().invoke(selected, invoked)
    assert result.exit_code == expected, (result.output, result.exception)
    if case.startswith("malformed-"):
        assert "malformed recorded configuration" in result.output and "seed" in result.output
    if case.startswith("stale-control"):
        assert "skyrl.trainer.policy.optimizer_config.lr" in result.output
    if "candidate-compose" in case:
        assert "native configuration cannot be composed:" in result.output and "gspo" in result.output.lower()
    if case == "compose-timeout":
        assert "timed out after 120 s" in result.output
    if case == "unexpected":
        assert "KeyError" in result.output and "grader_kind_counts" in result.output
    if case == "missing-requirement":
        assert "does not compute" in result.output
    if case == "child":
        assert len(published) == 1
        report = gate.completed_gate(built.gate)
        assert Path(report.path(), gate.GATE_REPORT_FILENAME).exists()
        assert gate.GateReport.raw_load(report.path()).verdict == "pass"
        return
    if not publication_case:
        return
    snapshot = printed_status(result.output)
    if case == "data-short":
        assert "publication" not in snapshot and published == []
        return
    paths = [[f"{training.name}-{preset.revision}", f"native-{index}"] for index, training in enumerate(built.runs)]
    if unavailable["value"]:
        assert snapshot["publication"] == {
            "served_records": paths[:1],
            "unserved_records": paths[1:],
            "errors": ["Credential unavailable; run iris login"],
            "retry": "Repeat the original command with --wait.",
        }
    if case == "mixed-native-failed":
        assert [row["attempt_id"] for row in published] == ["native-0"]
        assert snapshot["publication"] == {
            "served_records": paths[:1],
            "unserved_records": paths[1:],
            "errors": ["Gate report missing for this attempt; repeat the original command with --wait"],
            "retry": "Repeat the original command with --wait.",
        }
    if case == "mixed-native-failed":
        assert published[0]["status"] == "failed" and published[0]["gate_result"] is None
    if expected == 0 and (not unavailable["value"]):
        for row in snapshot["runs"]:
            assert parse_qs(urlsplit(row["rl_progress_url"]).fragment)["attempt"] == [row["attempt_id"]]
    if case == "changed-source":
        return
    unavailable["value"] = False
    phase["retry"] = True
    arguments[arguments.index("--issue") + 1] = "74"
    previously_published = len(published)
    repeated = CliRunner().invoke(
        snowball.command(
            snowball_model.SNOWBALL, argv=tuple(arguments), entry_module="experiments.post_training.snowball_rl"
        ),
        arguments,
    )
    assert repeated.exit_code == (0 if case in {"mixed-native-failed", "control-missing-receipt"} else expected), (
        repeated.output,
        repeated.exception,
    )
    recovered = printed_status(repeated.output)
    assert recovered["publication"]["unserved_records"] == [] and recovered["publication"]["errors"] == []
    assert {row["attempt_id"]: row["issue"] for row in recovered["runs"]} == (
        {"native-retry": 74, "native-1": 73} if case == "mixed-native-failed" else {attempt: 73 for _, attempt in paths}
    )
    if case == "mixed-native-failed":
        retry = next(row for row in published[previously_published:] if row["attempt_id"] == "native-retry")
        assert retry["status"] == "succeeded" and retry["gate_result"]["verdict"] == "pass" and (retry["issue"] == 74)
    else:
        cached = next(row for row in published[previously_published:] if row["attempt_id"] == "native-0")
        assert cached["issue"] == 73


@pytest.mark.parametrize("fault", ["untracked", "dirty"])
def test_control_refuses_actual_untracked_source_before_execution(selection, tmp_path, monkeypatch, fault):
    root = tmp_path / "git-source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "experiments").mkdir()
    (root / "experiments/untracked.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(recipe, "REPO_ROOT", root)
    if fault == "dirty":
        subprocess.run(["git", "-C", str(root), "add", "experiments/untracked.py"], check=True)
        monkeypatch.setattr(
            Provenance, "capture", lambda *args, **kwargs: Provenance("tree", "commit", True, "branch", "test")
        )
    with pytest.raises(click.BadParameter, match="stage source" if fault == "untracked" else "clean checkout"):
        snowball.build_run(snowball_model.SNOWBALL.by_name["speed-medium"], control=True, table=snowball_model.SNOWBALL)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--wait",),
        ("python-init-conflict",),
        ("python-init-rescore",),
        ("--set", "trainer.dump_eval_results=false"),
        ("--set", 'trainer={"algorithm":{"off_policy_correction":"none"}}'),
        ("--set", "trainer.policy.optimizer_config.lr=wrong"),
    ],
)
def test_refusals_use_exit_four(selection, arguments, capsys, tmp_path):
    if arguments == ("python-init-rescore",):
        assert (
            snowball.main(snowball_model.SNOWBALL, argv=["rescore"], init=snowball_model.SNOWBALL.presets[0].model) == 4
        )
        assert "Python init" in capsys.readouterr().err
        return
    initializer = snowball_model.SNOWBALL.presets[0].model if arguments == ("python-init-conflict",) else None
    preset_name = "speed-medium"
    if initializer is not None:
        preset_name = "train-long"
        name, version = "tests/valid-published-init", "2026.11.17"
        root = tmp_path / name / version
        _json(
            root / RECORD_FILENAME,
            ArtifactRecord(
                name=name, version=version, result_type=result_type_name(Artifact), source=str(root)
            ).model_dump(mode="json"),
        )
        arguments = ("--init", f"{name}@{version}")
    result = CliRunner().invoke(
        snowball.command(
            snowball_model.SNOWBALL,
            argv=tuple(arguments),
            init=initializer,
            entry_module="experiments.post_training.snowball_rl",
        ),
        ["--preset", preset_name, "--version", snowball_model.SNOWBALL.by_name[preset_name].revision, *arguments],
    )
    assert result.exit_code == 4, (result.output, result.exception)
    if initializer is not None:
        assert "Python init and --init" in result.output
    if "trainer.dump_eval_results=false" in arguments:
        assert "owned" in result.output


def test_train_long_reenters_the_model_module_and_reports_checkpoint_evaluations(
    recorded_runs, preset_table, tmp_path, monkeypatch
):
    preset = snowball_model.SNOWBALL.by_name["train-long"]
    preset = replace(preset, recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, host_disk="3TB")))
    table = preset_table(preset)
    initializer = ArtifactHfModel(
        ArtifactStep.adopt("tests/python-init", "2026.11.17", "s3://marin-test/models/python-init", kind=Artifact),
        "tests/python-tokenizer",
        "python-tokenizer-revision",
        relative_path="",
    )
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    monkeypatch.setenv("HF_TEST_KEY", "test-initializer-credential")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    monkeypatch.setattr(
        "s3fs.S3FileSystem.exists",
        lambda self, path, **kwargs: str(path).removeprefix("s3://") == "marin-test/models/python-init",
    )
    built = snowball.build_run(
        preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=("trainer.max_steps=12",)
    )
    source = recorded_runs(preset.name, built.runs[0].name, fingerprint=built.runs[0].fingerprint(), inputs=built.inputs)
    _json(source / "exports/best_checkpoint.json", {"step": 10})
    staged = source.with_name("before-native-execution")
    source.rename(staged)
    native_calls = []

    def native(command):
        inspection = native_json()(
            "import json,sys\nfrom cloud.iris import launch\n"
            "args=launch.create_parser().parse_args(sys.argv[1:])\n"
            "config=launch.load_launch_config(args.config)\n"
            "print(json.dumps({'forward_env':args.forward_env,'disk':config.iris.allocation.disk,"
            "'tokenizer':config.inputs.model.tokenizer_uri,'model':config.run.model,"
            "'attempt':config.run.attempt_id,'export':config.artifacts.export_root}))\n",
            command[command.index("marinskyrl") + 1 :],
        )
        assert (inspection["forward_env"], inspection["disk"], inspection["tokenizer"]) == (
            ["HF_TEST_KEY"],
            "3TB",
            "tests/python-tokenizer",
        )
        launched = read_attempt(inspection["export"], inspection["attempt"])
        assert launched is not None and launched.issue == "opted_out"
        native_calls.append(command)
        shutil.copytree(staged, source, dirs_exist_ok=True)
        stored = json.loads((source / "terminal.json").read_text())
        stored["config"]["run"]["model"] = inspection["model"]
        write_receipt(source, stored)
        write_attempt(
            inspection["export"],
            AttemptRecord(
                run_id=stored["config"]["run"]["id"],
                attempt_id=stored["config"]["run"]["attempt_id"],
                issue=launched.issue,
            ),
        )
        response = stored["result"]
        response["model"] = dict(
            policy_export_uri=str(source / "policy"),
            global_step=12,
            tokenizer_uri="tests/python-tokenizer",
            tokenizer_revision="python-tokenizer-revision",
            checkpoint_root=str(source / "checkpoints"),
            terminal_manifest_uri=str(source / "terminal.json"),
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(skyrl_api, "_run_launcher", native)
    published = []

    def publish(record, service_url=None):
        published.append(record)
        return (
            (service_url or "http://progress.test")
            + "/#"
            + urlencode({"run": record.run_id, "attempt": record.attempt_id})
        )

    monkeypatch.setattr("experiments.post_training.rl_publication.publish_result", publish)
    arguments = [
        "--preset",
        preset.name,
        "--version",
        preset.revision,
        "--run",
        "--wait",
        "--no-issue",
        "--url",
        "http://127.0.0.1:55777",
        "--max-concurrent",
        "3",
        "--set",
        "trainer.max_steps=12",
    ]
    job = JobName.from_wire("/test-user/train-coordinator")
    client = MagicMock(spec=IrisClient)
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: nullcontext(client))
    mode, coordinator_module = "valid", "first_model_entry"

    def submit(**kwargs):
        assert kwargs["entrypoint"].command == ["python", "-m", coordinator_module, *arguments]
        environment = kwargs["environment"].env_vars
        if mode != "overlapping-environment-mismatch":
            assert environment["HF_TEST_KEY"] == "test-initializer-credential"
        assert "UNRELATED_SECRET" not in environment
        if mode == "failed-native":
            return SimpleNamespace(job_id=job)
        with monkeypatch.context() as child:
            for key, value in environment.items():
                child.setenv(key, value)
            child.setenv("IRIS_TASK_ID", "/test-user/train-coordinator/0")
            child.setattr(sys, "argv", [coordinator_module, *arguments])
            child.setattr(
                "experiments.post_training.rl_publication.publish_result",
                lambda *args: pytest.fail("Coordinator contacted Marina"),
            )
            info = JobInfo(task_id=JobName.from_wire("/test-user/train-coordinator/0"))
            with iris_ctx_scope(IrisContext.from_job_info(info)), pytest.raises(SystemExit) as completed:
                runpy.run_module(coordinator_module, run_name="__main__", alter_sys=True)
            assert completed.value.code == (0 if mode == "valid" else 4)
        set_job_info(None)
        return SimpleNamespace(job_id=job)

    client.submit.side_effect = submit
    client.job.return_value.wait.side_effect = lambda *args, **kwargs: SimpleNamespace(
        state=JobState.SUCCEEDED if mode == "valid" else JobState.FAILED
    )

    def module(name, *, entry_module=None, use_init=True, forward_env=("HF_TEST_KEY",), policy_forward=None):
        path = tmp_path / (name + ".py")
        dependency = ""
        if policy_forward is not None:
            dependency = (
                f"producer=rl.build_run(snowball_rl.SNOWBALL.by_name['train-long'],table=snowball_rl.SNOWBALL,"
                f"settings=('trainer.max_steps=2',),forward_env={policy_forward!r}).runs[0]\n"
                f"result=SkyRLRun.raw_load({str(source)!r}).model_copy(update={{'hf_model_uri':'s3://marin-test/models/python-init'}})\n"
                "directory=Path(producer.path());directory.mkdir(parents=True,exist_ok=True)\n"
                "record=ArtifactRecord(name=producer.name,version=producer.version,fingerprint=producer.fingerprint(),"
                "result_type=result_type_name(SkyRLRun),result=result.result_payload())\n"
                "(directory/RECORD_FILENAME).write_text(record.model_dump_json())\n"
                "init=SkyRLPolicyModel(producer)\n"
            )
        path.write_text(
            "from pathlib import Path\nfrom marin.execution.lazy import ArtifactStep\n"
            "from marin.execution.artifact import Artifact,ArtifactRecord,RECORD_FILENAME,result_type_name\n"
            "from marin.rl.skyrl import ArtifactHfModel,SkyRLRun,SkyRLPolicyModel\n"
            "from experiments.post_training import rl, snowball_rl\n"
            "step=ArtifactStep.adopt('tests/python-init','2026.11.17','s3://marin-test/models/python-init',kind=Artifact)\n"
            "init=ArtifactHfModel(step,'tests/python-tokenizer','python-tokenizer-revision',relative_path='')\n"
            + dependency
            + f"raise SystemExit(rl.main(snowball_rl.SNOWBALL,init={'init' if use_init else 'None'},"
            f"forward_env={forward_env!r},entry_module={entry_module!r}))\n"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        monkeypatch.setitem(sys.modules, name, importlib.util.module_from_spec(spec))

    @click.command()
    def entry():
        monkeypatch.setattr(sys, "argv", ["first_model_entry", *arguments])
        with pytest.raises(SystemExit) as completed:
            runpy.run_module("first_model_entry", run_name="__main__", alter_sys=True)
        raise click.exceptions.Exit(completed.value.code)

    for override in (None, "second_model_entry"):
        coordinator_module = override or "first_model_entry"
        module("first_model_entry", entry_module=override)
        if override is not None:
            module(override)
        result = CliRunner().invoke(entry)
        assert result.exit_code == 0, (result.output, result.exception)
        assert "best_checkpoint_step: 10" in result.output and f"{preset.evaluations[0].name}@12/scored" in result.output
        assert "http://127.0.0.1:55777/" in result.output and "test-initializer-credential" not in result.output
    assert len(native_calls) == 1
    assert published[-1].optimizer_steps == published[-1].planned_updates == 12
    assert (published[-1].model, published[-1].issue) == ("snowball", "opted_out")

    for mode in (
        "initializer-mismatch",
        "environment-mismatch",
        "overlapping-environment-mismatch",
        "dependency-environment-mismatch",
    ):
        coordinator_module = "second_model_entry"
        module(
            "first_model_entry",
            entry_module=coordinator_module,
            forward_env=("WANDB_API_KEY",) if mode == "overlapping-environment-mismatch" else ("HF_TEST_KEY",),
            policy_forward=("WANDB_API_KEY",) if mode == "dependency-environment-mismatch" else None,
        )
        module(
            coordinator_module,
            use_init=mode != "initializer-mismatch",
            forward_env=() if mode in {"environment-mismatch", "overlapping-environment-mismatch"} else ("HF_TEST_KEY",),
            policy_forward=() if mode == "dependency-environment-mismatch" else None,
        )
        result = CliRunner().invoke(entry)
        assert result.exit_code == 3 and "coordinator rebuilt a different RL graph" in result.output, (
            result.output,
            result.exception,
        )
        assert "test-initializer-credential" not in result.output and len(native_calls) == 1

    evaluations = read_evaluations(str(source / "exports"), attempt_id="recorded-attempt", evaluation="eval/")
    for missing in (False, True):
        changed = []
        damaged = 0
        for row in evaluations:
            if row.step == 10:
                if missing:
                    continue
                if damaged < 2:
                    row = row.model_copy(
                        update={
                            "answers": [
                                answer.model_copy(update={"error_treatment": "passthrough"}) for answer in row.answers
                            ]
                        }
                    )
                    damaged += 1
            changed.append(row)
        replace_evaluations(source, changed)
        planning_args = ["--preset", preset.name, "--version", preset.revision, "--set", "trainer.max_steps=12"]
        result = CliRunner().invoke(
            snowball.command(table, argv=tuple(planning_args), init=initializer, forward_env=("HF_TEST_KEY",)),
            planning_args,
        )
        assert result.exit_code == 1 and f"{preset.evaluations[0].name}/step-10/" in result.output, (
            result.output,
            result.exception,
        )
        facts = {
            fact["name"]: fact for row in printed_status(result.output)["runs"] for fact in row["evaluation_invariants"]
        }
        prefix = preset.evaluations[0].name + "/step-10/"
        if missing:
            assert not facts[prefix + "batch_finished"]["holds"]
        else:
            assert facts[prefix + "scored_fraction"]["value"] == pytest.approx(0.98)
            assert not facts[prefix + "scored_fraction"]["holds"]
            assert facts[prefix + "batch_finished"]["holds"] and facts[prefix + "grader_error_fraction"]["holds"]
    replace_evaluations(source, evaluations)
    mode, coordinator_module = "failed-native", "first_model_entry"
    module("first_model_entry")
    with edit_receipt(source) as receipt:
        receipt["result"]["state"] = "failed"
    result = CliRunner().invoke(entry)
    assert result.exit_code == 3 and published[-1].status == "failed", (result.output, result.exception)
    with edit_receipt(source) as receipt:
        receipt["result"]["state"] = "succeeded"
        receipt["config"]["skyrl"]["trainer"]["seed"] = None
    with monkeypatch.context() as refused:
        refused.setattr(client, "submit", lambda **kwargs: pytest.fail("Malformed status reached submission"))
        result = CliRunner().invoke(entry)
        assert result.exit_code == 4 and "seed" in result.output, (result.output, result.exception)
