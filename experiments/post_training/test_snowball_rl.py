# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import click
import marin.rl.skyrl as adapter
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from iris.rpc import controller_pb2, job_pb2
from marin.execution.artifact import RECORD_FILENAME
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import SkyRLRun
from marin.rl.skyrl import _launcher_command as launcher_command_for_test
from marin.training.training import LevanterCheckpoint
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
                "full_validation_rows": 910,
                "grader_kind_counts": {"train": {"remote_judge": 0}},
            }
        )
    )
    return ArtifactStep.adopt("tests/candidate", "2026.10.26", str(root))


@pytest.fixture(autouse=True)
def source_provenance(monkeypatch):
    monkeypatch.setattr(
        snowball, "_provenance", lambda **kwargs: Provenance("tree-one", "commit-one", False, "branch", "test")
    )


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
    changed = snowball.build_run(
        snowball.PRESETS["learning-medium"],
        settings=tuple(
            map(
                snowball.parse_setting,
                (
                    "trainer={train_batch_size: 64, policy_mini_batch_size: 64}",
                    "generator.n_samples_per_prompt=8",
                    "trainer.policy.megatron_config.logprob_chunk_size=512",
                ),
            )
        ),
    ).runs[0]
    changed_config = changed.build_config(StepContext.for_fingerprint(changed.runtime_args, changed.deps))
    path = tmp_path / "changed-batch.yaml"
    path.write_text(changed_config.launch_config_yaml)
    paths.append(str(path))
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
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.inference_engines.vllm.utils import apply_openai_sampling
configs = [load_launch_config(path) for path in sys.argv[1:]]
for index, config in enumerate(configs):
    trainer, generator = config.skyrl.trainer, config.skyrl.generator
    name = config.run.id
    if "train-long-" in name:
        evaluation = get_vllm_sampling_params(generator.eval_sampling_params)
        if evaluation["logprobs"] is not None:
            evaluation["logprobs"] = True
        apply_openai_sampling(evaluation, dict(generator.sampling_params), True)
        training = get_vllm_sampling_params(generator.sampling_params)
        training.update(logprobs=True, min_tokens=0)
        apply_openai_sampling(training, dict(generator.sampling_params), True)
        training["min_tokens"] = 1
        try:
            apply_openai_sampling(training, dict(generator.sampling_params), True)
        except ValueError:
            pass
        else:
            raise AssertionError("training distribution validation accepted min_tokens=1")
    if "speed-short-" in name or "data-short-" in name or "/eval/panel-910" in name:
        assert config.runtime.entrypoint == "skyrl_train.entrypoints.main_generate"
        assert config.iris.allocation.num_nodes == 1
        assert trainer.eval_interval > 0 and config.inputs.validation_data
        expected_admission = 2048 if "speed-short-" in name else 1024 if "data-short-" in name else 910
        assert config.skyrl.trajectory_runner.max_concurrent_tasks == expected_admission
    if "/eval/panel-910" in name and "train-long-" in name:
        expected_window = 65536 if generator.engine_init_kwargs.get("tool_parser") == "hermes" else 24576
        assert generator.engine_init_kwargs.max_model_len == expected_window
    if index == len(configs) - 1:
        assert trainer.train_batch_size == 64 and generator.n_samples_per_prompt == 8
    if generator.eval_sampling_params.ignore_eos or generator.sampling_params.ignore_eos:
        assert not generator.apply_overlong_filtering
        assert generator.sampling_params.get("min_tokens") is None
        assert generator.sampling_params.max_generate_length == 4096
    if config.inputs.validation_data:
        assert trainer.eval_interval > 0
    if config.runtime.training_type is not None and config.inputs.validation_data:
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
    for value in ("generator.sampling_params.ignore_eos=true", "generator.sampling_params={ignore_eos: true}"):
        with pytest.raises(click.BadParameter, match="owned by"):
            snowball.build_run(snowball.PRESETS["learning-medium"], settings=(snowball.parse_setting(value),))
    for value in ("+extra_env.NCCL_NET=Socket", "generator.apply_overlong_filtering=true"):
        with pytest.raises(click.BadParameter, match="owned by"):
            snowball.build_run(snowball.PRESETS["speed-medium"], settings=(snowball.parse_setting(value),))


def test_run_identity_tracks_code_inputs_and_replicates(selection, monkeypatch):
    provenance = Provenance("tree-one", "commit-one", False, "branch", "test")
    monkeypatch.setattr(snowball, "_provenance", lambda **kwargs: provenance)
    preset = snowball.PRESETS["learning-medium"]
    baseline = snowball.build_run(preset, replicates=2)
    repeated = snowball.build_run(preset, replicates=3)
    assert [step.name for step in baseline.runs] == [step.name for step in repeated.runs[:2]]
    assert len({step.name for step in repeated.runs}) == 3
    changed = snowball.build_run(preset, settings=(snowball.parse_setting("trainer.policy.optimizer_config.lr=2e-6"),))
    assert changed.runs[0].name != baseline.runs[0].name
    mapped = []
    for rate in (2e-6, 3e-6):
        mapped.append(
            snowball.build_run(
                preset,
                settings=tuple(
                    map(
                        snowball.parse_setting,
                        (
                            f"trainer.policy.optimizer_config={{lr: {rate:.8f}}}",
                            "trainer.policy.optimizer_config={weight_decay: 0.1}",
                        ),
                    )
                ),
            ).runs[0]
        )
    assert mapped[0].name != mapped[1].name
    dotted = snowball.build_run(
        preset,
        settings=tuple(
            map(
                snowball.parse_setting,
                (
                    "trainer.policy.optimizer_config.lr=0.000002",
                    "trainer.policy.optimizer_config.weight_decay=0.1",
                ),
            )
        ),
    ).runs[0]
    assert mapped[0].name == dotted.name
    provenance = replace(provenance, tree_hash="tree-two")
    assert snowball.build_run(preset).runs[0].name != baseline.runs[0].name
    data = snowball.PRESETS["data-short"]
    other = replace(selection, version="2026.10.27")
    assert snowball.build_run(data, tasks=selection).runs[0].name != snowball.build_run(data, tasks=other).runs[0].name
    initializer = ArtifactStep.adopt("tests/initializer", "2026.10.26", selection.path(), kind=LevanterCheckpoint)
    assert snowball.build_run(preset, init=initializer).runs[0].name != snowball.build_run(preset).runs[0].name
    long = snowball.PRESETS["train-long"]
    assert snowball.build_run(long, layout=snowball.P40).runs[0].name != snowball.build_run(long).runs[0].name
    original = snowball.build_run(long).runs[0]
    snowball.build_run(
        long, settings=(snowball.parse_setting("trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=true"),)
    )
    assert snowball.build_run(long).runs[0].fingerprint() == original.fingerprint()
    dated = snowball.parse_setting("+extra_env.EXPERIMENT_DATE=2026-10-27")
    assert snowball.build_run(long, settings=(dated,)).runs[0].name != snowball.build_run(long).runs[0].name


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
    for answer in answers:
        score = answer["score"]
        answer["verification"] = {
            "status": "verified",
            "score": sum(score) if isinstance(score, list) else score,
            "score_min": 0.0,
            "score_max": 1.0,
        }
    answers[0]["score"] = 0.8
    answers[0]["output_response"] = "first\u2028second\u2029third\u0085fourth"
    answers[0]["verification"].update(score=5.0, score_min=1.0, score_max=5.0)
    answers[3]["verification"] = {"status": "error", "score": None, "score_min": 0.0, "score_max": 1.0}
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
                "config": {"inputs": {"validation_data": [{"uri": str(selection)}]}},
                "train_data_sources": [],
                "val_data_sources": [],
            }
        )
    )
    for session in ("global_step_25_evals", "eval_only"):
        directory = tmp_path / "exports" / "dumped_evals" / session
        directory.mkdir(parents=True)
        (directory / "structure-closure.jsonl").write_text(
            "\n".join(json.dumps(answer, ensure_ascii=False) for answer in answers) + "\n"
        )
        (directory / "aggregated_results.jsonl").write_text('{"avg_score": 0.5}\n')
    evaluator = snowball.EVALUATORS[snowball.EvalHarness.PANEL]
    scores = evaluator.scores(snowball.EVALUATIONS["panel-100"], str(tmp_path))
    rows = scores.to_pylist()
    assert scores.schema == snowball.EVAL_SCORE_SCHEMA
    assert {row["step"] for row in rows} == {-1, 25}
    final = [row for row in rows if row["step"] == 25]
    assert [(row["item"], row["attempt"], row["scored"], row["score"]) for row in final] == [
        (passed, 0, True, 1.0),
        (failed, 0, True, 0.0),
        (failed, 1, True, 1.0),
        (failed, 2, False, 0.0),
        (stem, 0, True, 0.0),
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


@pytest.mark.parametrize("receipt", ["failed-attempt", "iris"])
def test_resume_reads_failed_run_without_rebuilding_it(selection, tmp_path, monkeypatch, receipt):
    original = snowball.build_run(snowball.PRESETS["train-long"], layout=snowball.P40).runs[0]
    document = yaml.safe_load(
        original.build_config(StepContext.for_fingerprint(original.runtime_args, original.deps)).launch_config_yaml
    )
    source = tmp_path / original.name / original.version
    source.mkdir(parents=True)
    checkpoints = source / "checkpoints"
    (checkpoints / "global_step_10").mkdir(parents=True)
    marker = checkpoints / "latest_ckpt_global_step.txt"
    marker.write_text("10")
    (checkpoints / "global_step_10/metadata.json").write_text("{}")
    attempts = source / "attempts"
    attempts.mkdir()
    document["run"]["attempt_id"] = "attempt-fixture"
    document["iris"]["job_name"] = "train-long-fixture-attempt-fixture"
    document["artifacts"].update(checkpoint_root=str(checkpoints), attempts_root=str(attempts))
    (source / "resolved-launch.yaml").write_text(
        json.dumps({"config": document, "train_data_sources": [], "val_data_sources": []})
    )
    if receipt == "failed-attempt":
        (attempts / "attempt-fixture.json").write_text(
            json.dumps(
                {"config": document, "result": {"state": "failed", "iris_job_id": "/tests/failed", "model": None}}
            )
        )
    state = {"value": job_pb2.JOB_STATE_KILLED}
    job_id = "/tests/parent/" + document["iris"]["job_name"]

    @contextmanager
    def endpoint(**kwargs):
        yield SimpleNamespace(url="http://iris.test", credentials=None)

    @contextmanager
    def rpc(*args):
        yield SimpleNamespace(
            list_jobs=lambda request: controller_pb2.Controller.ListJobsResponse(
                jobs=[job_pb2.JobStatus(job_id=job_id, state=state["value"])]
            )
        )

    monkeypatch.setattr(snowball, "connect_controller", endpoint)
    monkeypatch.setattr(snowball, "rpc_client", rpc)
    monkeypatch.setattr(snowball, "_artifact_prefix", lambda cluster: str(tmp_path))
    (source / ".executor_status").write_text("FAILED")
    command = CliRunner().invoke(
        snowball.main,
        [
            "--preset",
            "train-long",
            "--layout",
            "p40",
            "--resume-from",
            f"{original.name}@{original.version}",
            "--version",
            "dev",
        ],
    )
    assert command.exit_code == 0, command.output
    earlier = ArtifactStep.adopt(original.name, original.version, str(source), kind=SkyRLRun)
    resumed = snowball.build_run(snowball.PRESETS["train-long"], layout=snowball.P40, resume_from=earlier).runs[0]
    assert resumed.name != original.name
    assert earlier not in resumed.deps
    assert all(dep.artifact_type is not SkyRLRun for dep in resumed.deps)
    config = resumed.build_config(
        StepContext.for_run(
            str(tmp_path / "resumed"), str(tmp_path / "artifacts"), deps=resumed.deps, runtime_args=resumed.runtime_args
        )
    )
    assert config.output.checkpoint_root == str(checkpoints)
    assert yaml.safe_load(config.launch_config_yaml)["artifacts"]["checkpoint_root"] == str(checkpoints)
    assert (source / ".executor_status").read_text() == "FAILED"
    assert not (source / RECORD_FILENAME).exists()
    assert not (source / "terminal.json").exists()
    if receipt == "iris":
        state["value"] = job_pb2.JOB_STATE_RUNNING
        with pytest.raises(click.BadParameter, match=job_id):
            snowball.build_run(snowball.PRESETS["train-long"], layout=snowball.P40, resume_from=earlier)
        state["value"] = job_pb2.JOB_STATE_KILLED
    marker.unlink()
    with pytest.raises(click.BadParameter, match="14 days"):
        snowball.build_run(snowball.PRESETS["train-long"], layout=snowball.P40, resume_from=earlier)
