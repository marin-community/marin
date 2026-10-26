# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import click
import marin.rl.skyrl as adapter
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, StepContext, resolve
from marin.rl.skyrl import SkyRLPolicyModel, SkyRLRun
from marin.rl.skyrl import _launcher_command as launcher_command_for_test
from rigging.provenance import Provenance

from experiments.post_training import snowball_rl as snowball


@pytest.fixture
def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(snowball, "SELECTION_ROOT", str(tmp_path))
    root = tmp_path / "snowball-100k" / snowball.DEFAULT_SELECTION_VERSION
    root.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "recipe": {"heldout": {"version": "2026.10.23.1", "rows": 910}},
                "validation_rows": 100,
                "grader_kind_counts": {"train": {"remote_judge": 0}},
            }
        )
    )
    return ArtifactStep.adopt("tests/candidate", "2026.10.26", str(root))


@pytest.mark.timeout(180)  # The real bootstrap installs its independent, locked CPU environment.
def test_presets_compose_and_preserve_measurement_contracts(selection, tmp_path):
    assert Path(adapter.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2] / "lib")
    paths = []
    launches = {}
    for name, preset in snowball.PRESETS.items():
        layouts = (snowball.L112, snowball.P40) if name == "train-long" else (None,)
        for layout in layouts:
            run = snowball.build_run(
                preset, tasks=selection if preset.focus == snowball.Focus.DATA else None, layout=layout
            )
            for index, step in enumerate((*run.runs, *run.post_run_evaluations)):
                config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
                path = tmp_path / f"{name}-{layout.name if layout else 'default'}-{index}.yaml"
                path.write_text(config.launch_config_yaml)
                paths.append(str(path))
                if index == 0:
                    launches[name] = yaml.safe_load(config.launch_config_yaml)
            if name == "train-long":
                assert run.runs[0] in run.post_run_evaluations[0].deps
    command = launcher_command_for_test(paths[0])
    bootstrap = command[: command.index("marinskyrl")]
    result = subprocess.run(
        [
            *bootstrap,
            "python",
            "-c",
            """
import json, sys
from cloud.iris.launch_config import load_launch_config
configs = [load_launch_config(path) for path in sys.argv[1:]]
for config in configs:
    trainer, generator = config.skyrl.trainer, config.skyrl.generator
    if generator.eval_sampling_params.ignore_eos or generator.sampling_params.ignore_eos:
        assert not generator.apply_overlong_filtering
        assert generator.sampling_params.get("min_tokens") is None
    if config.runtime.training_type is not None and trainer.eval_interval > 0:
        callbacks = [entry.type for entry in trainer.callbacks]
        assert "evaluation" in callbacks
        if "best_checkpoint" in callbacks:
            assert callbacks.index("evaluation") < callbacks.index("best_checkpoint")
print(json.dumps([config.iris.allocation.num_nodes for config in configs]))
""",
            *paths,
        ],
        capture_output=True,
        text=True,
        timeout=170,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout.splitlines()[-1])) == len(paths)
    # Candidate selection is the only difference between the two learning experiments.
    learning, data = launches["learning-medium"], launches["data-medium"]
    assert learning["skyrl"] == data["skyrl"]
    assert learning["inputs"]["model"] == data["inputs"]["model"]
    assert learning["inputs"]["validation_data"] == data["inputs"]["validation_data"]
    assert learning["inputs"]["train_data"] != data["inputs"]["train_data"]


def test_run_identity_tracks_code_inputs_and_replicates(selection, monkeypatch):
    provenance = Provenance("tree-one", "commit-one", False, "branch", "test")
    monkeypatch.setattr(Provenance, "capture", lambda *args: provenance)
    preset = snowball.PRESETS["learning-medium"]
    baseline = snowball.build_run(preset, replicates=2)
    repeated = snowball.build_run(preset, replicates=3)
    assert [step.name for step in baseline.runs] == [step.name for step in repeated.runs[:2]]
    assert len({step.name for step in repeated.runs}) == 3
    changed = snowball.build_run(preset, settings=(snowball.parse_setting("trainer.policy.optimizer_config.lr=2e-6"),))
    assert changed.runs[0].name != baseline.runs[0].name
    provenance = replace(provenance, tree_hash="tree-two")
    assert snowball.build_run(preset).runs[0].name != baseline.runs[0].name
    data = snowball.PRESETS["data-short"]
    other = replace(selection, version="2026.10.27")
    assert snowball.build_run(data, tasks=selection).runs[0].name != snowball.build_run(data, tasks=other).runs[0].name


@pytest.mark.timeout(180)
def test_changed_batch_shape_composes_at_execution_boundary(selection, tmp_path):
    run = snowball.build_run(
        snowball.PRESETS["learning-medium"],
        settings=tuple(
            map(
                snowball.parse_setting,
                (
                    "trainer.train_batch_size=64",
                    "trainer.policy_mini_batch_size=64",
                    "generator.n_samples_per_prompt=8",
                ),
            )
        ),
    )
    step = run.runs[0]
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    path = tmp_path / "batch.yaml"
    path.write_text(config.launch_config_yaml)
    command = launcher_command_for_test(str(path))
    result = subprocess.run(
        [
            *command[: command.index("marinskyrl")],
            "python",
            "-c",
            """
import sys
from cloud.iris.launch_config import load_launch_config
config = load_launch_config(sys.argv[1])
assert config.skyrl.trainer.train_batch_size * config.skyrl.generator.n_samples_per_prompt == 512
print("batch valid")
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=170,
    )
    assert result.returncode == 0, result.stderr


def test_finished_policy_resolves_terminal_export_for_post_evaluation(tmp_path):
    root = tmp_path / "finished"
    root.mkdir()
    result = SkyRLRun(
        path=str(root),
        hf_model_uri=str(root / "policy"),
        global_step=100,
        tokenizer_uri="tests/tokenizer",
        tokenizer_revision="tokenizer-commit",
        checkpoint_root=str(root / "checkpoints"),
        draft_checkpoint_root=None,
        terminal_manifest_uri=str(root / "terminal.json"),
        iris_job_id="/test/run",
    )
    record = ArtifactRecord(result_type=result_type_name(SkyRLRun), result=result.result_payload())
    (root / RECORD_FILENAME).write_text(record.model_dump_json())
    source = ArtifactStep.adopt("tests/finished", "2026.10.26", str(root), kind=SkyRLRun)
    model = SkyRLPolicyModel(source)
    context = StepContext.for_run(str(tmp_path / "evaluation"), str(tmp_path), deps=model.deps())
    resolved = model.resolve(context)
    assert resolved.uri == str(root / "policy")
    assert (resolved.tokenizer_uri, resolved.tokenizer_revision) == ("tests/tokenizer", "tokenizer-commit")
    record.result["hf_model_uri"] = None
    (root / RECORD_FILENAME).write_text(record.model_dump_json())
    with pytest.raises(ValueError, match="export_hf"):
        model.resolve(StepContext.for_run(str(tmp_path / "other"), str(tmp_path), deps=model.deps()))


def test_panel_scores_join_permuted_answers_and_aggregate_tasks(tmp_path):
    # Retained task/score fields from the E2 GPU dump at snowball-rl-e2-structural/2026.10.26.2.
    failed = "structure-closure:8ce02d80cfa6b3a7e9594ff985c23fc60ca02a462173cbdfed58e575d98a4417"
    passed = "structure-closure:e2947c55f619a21fe06d88f9ebf16ab4800ab1f19a6767fcb8c8894b6995cad7"
    stem = "stem:fixture"
    answers = [
        {"task_id": passed, "score": 1.0},
        {"task_id": failed, "score": 0.0},
        {"task_id": failed, "score": [1.0]},
        {"task_id": failed, "score": 0.0, "exception_type": "TimeoutError"},
        {"task_id": stem, "score": 0.0},
    ]
    selection = tmp_path / "selection"
    selection.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"task_id": failed, "family": "structural", "aaii_category": "format"},
                {"task_id": passed, "family": "structural", "aaii_category": "format"},
                {"task_id": stem, "family": "stem", "aaii_category": "scientific_reasoning"},
            ]
        ),
        selection / snowball.MANIFEST_FILENAME,
    )
    (tmp_path / "resolved-launch.yaml").write_text(
        yaml.safe_dump(
            {
                "inputs": {"validation_data": [{"uri": str(selection)}]},
            }
        )
    )
    for session in ("global_step_25_evals", "eval_only"):
        directory = tmp_path / "exports" / "dumped_evals" / session
        directory.mkdir(parents=True)
        (directory / "structure-closure.jsonl").write_text("\n".join(map(json.dumps, answers)))
        (directory / "aggregated_results.jsonl").write_text('{"avg_score": 0.5}\n')
    evaluator = snowball.EVALUATORS[snowball.EvalHarness.PANEL]
    scores = evaluator.scores(snowball.EVALUATIONS["panel-100"], str(tmp_path))
    rows = scores.to_pylist()
    assert scores.schema == snowball.EVAL_SCORE_SCHEMA
    assert {row["step"] for row in rows} == {-1, 25}
    final = [row for row in rows if row["step"] == 25]
    assert [(row["item"], row["attempt"], row["scored"]) for row in final] == [
        (passed, 0, True),
        (failed, 0, True),
        (failed, 1, True),
        (failed, 2, False),
        (stem, 0, True),
    ]
    assert {(row["family"], row["aaii_category"]) for row in rows} == {
        ("structural", "format"),
        ("stem", "scientific_reasoning"),
    }
    statistics = snowball.evaluation_statistics(pa.Table.from_pylist(final, schema=scores.schema))
    assert statistics["panel-100/mean"] == 0.5
    assert statistics["panel-100/family_mean"] == 0.375
    assert statistics["panel-100/mixed_fraction"] == pytest.approx(1 / 3)
    assert statistics["panel-100/scored_fraction"] == 0.8
    assert statistics["panel-100/aaii_category/format/mean"] == 0.75
    path = tmp_path / "exports/dumped_evals/eval_only/structure-closure.jsonl"
    path.write_text('{"uid":"0","score":0.0}\n')
    with pytest.raises(ValueError, match="TaskSpec id"):
        evaluator.scores(snowball.EVALUATIONS["panel-100"], str(tmp_path))


def test_resume_reads_failed_run_without_rebuilding_it(selection, tmp_path, monkeypatch):
    original = snowball.build_run(snowball.PRESETS["train-long"], layout=snowball.P40).runs[0]
    document = yaml.safe_load(
        original.build_config(StepContext.for_fingerprint(original.runtime_args, original.deps)).launch_config_yaml
    )
    source = tmp_path / "failed"
    source.mkdir()
    checkpoints = source / "checkpoints"
    (checkpoints / "global_step_10").mkdir(parents=True)
    marker = checkpoints / "latest_ckpt_global_step.txt"
    marker.write_text("10")
    (checkpoints / "global_step_10/metadata.json").write_text("{}")
    document["artifacts"]["checkpoint_root"] = str(checkpoints)
    terminal = {"config": document, "result": {"state": "failed", "iris_job_id": "/tests/failed", "model": None}}
    (source / "terminal.json").write_text(json.dumps(terminal))
    (source / ".executor_status").write_text("FAILED")
    earlier = ArtifactStep.adopt(original.name, original.version, str(source), kind=SkyRLRun)
    resumed = snowball.build_run(snowball.PRESETS["train-long"], layout=snowball.P40, resume_from=earlier).runs[0]
    assert earlier not in resumed.deps
    metadata = next(dep for dep in resumed.deps if dep.artifact_type is SkyRLRun)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    resolve(metadata)
    config = resumed.build_config(
        StepContext.for_run(
            str(tmp_path / "resumed"), str(tmp_path / "artifacts"), deps=resumed.deps, runtime_args=resumed.runtime_args
        )
    )
    assert config.output.checkpoint_root == str(checkpoints)
    assert yaml.safe_load(config.launch_config_yaml)["artifacts"]["checkpoint_root"] == str(checkpoints)
    assert (source / ".executor_status").read_text() == "FAILED"
    assert not (source / RECORD_FILENAME).exists()
    marker.unlink()
    with pytest.raises(click.BadParameter, match="14 days"):
        snowball.build_run(snowball.PRESETS["train-long"], layout=snowball.P40, resume_from=earlier)
