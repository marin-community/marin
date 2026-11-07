# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import marin.rl.skyrl as adapter
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from iris.cluster.client.job_info import set_job_info
from iris.rpc import controller_pb2
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import SkyRLRun, skyrl_cpu_command
from rigging.provenance import Provenance
from taskcompendium.convert.answers import answer_task
from taskcompendium.grading_result import Outcome
from taskcompendium.models import Source
from taskcompendium.pipeline.models import RawRow
from verifyit.spec import ExactSpec

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring


@pytest.fixture
def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "SELECTION_ROOT", str(tmp_path))
    monkeypatch.setattr(presets, "artifact_prefix", lambda cluster: str(tmp_path))
    root = tmp_path / "snowball-100k" / presets.DEFAULT_SELECTION_VERSION
    root.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "recipe": {"heldout": {"version": "2026.10.23.1", "rows": 910}},
                "validation_rows": 100,
                "full_validation_rows": 910,
                "grader_kind_counts": {"train": {"remote_judge": 0}, "validation": {}, "full_validation": {}},
            }
        )
    )
    panel_name, panel_version = (presets.DEFAULT_TEST_PANEL_NAME, presets.DEFAULT_TEST_PANEL_VERSION)
    panel = tmp_path / panel_name / panel_version
    panel.mkdir(parents=True)
    (panel / "summary.json").write_text(
        json.dumps(
            {
                "validation_rows": 100,
                "full_validation_rows": 1500,
                "grader_kind_counts": {"validation": {}, "full_validation": {}},
                "recipe": {
                    "pass_rates": {
                        "uri": str(tmp_path / "test-panel-rates" / scoring.PASS_RATES_FILENAME),
                        "rows": 1500,
                        "model_revision": presets.MODEL_REVISION,
                        "version": panel_version,
                    }
                },
            }
        )
    )
    return ArtifactStep.adopt("tests/candidate", "2026.10.26", str(root))


@pytest.fixture
def source_provenance(monkeypatch):
    monkeypatch.setattr(
        snowball, "_provenance", lambda **kwargs: Provenance("tree-one", "commit-one", False, "branch", "test")
    )


@pytest.fixture
def synthetic_current_controls(monkeypatch):
    long_pin = gate.ControlPin("learning-long", "09026fd100", presets.PRESET_REVISION, 4)
    monkeypatch.setattr(
        gate,
        "CONTROLS",
        {
            name: tuple(replace(pin, revision=presets.PRESET_REVISION) for pin in pins)
            for name, pins in gate.CONTROLS.items()
        }
        | {"learning-long": (long_pin,)},
    )


@pytest.fixture
def native_json():
    def execute(script, paths):
        source = str(presets.REPO_ROOT / "vendor/skyrl")
        program = (
            "from pathlib import Path as SourcePath\nfrom cloud.iris import launch_config as SourceLaunch\n"
            "import skyrl_train as SourceTrain\n"
            f"assert all(SourcePath(module.__file__).resolve().is_relative_to({source!r}) "
            "for module in (SourceLaunch, SourceTrain))\n"
            "print('SNOWBALL_NATIVE_SOURCE_ORIGINS_PASS')\n" + script
        )
        result = subprocess.run(
            skyrl_cpu_command("python", "-c", program, *paths), capture_output=True, text=True, timeout=170
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.splitlines()[-1])

    return execute


@pytest.fixture
def recorded_runs(selection, tmp_path, monkeypatch, request):

    monkeypatch.setattr(
        Provenance, "capture", lambda *args, **kwargs: Provenance("tree-one", "commit-one", False, "branch", "test")
    )
    set_job_info(None)
    request.addfinalizer(lambda: set_job_info(None))
    monkeypatch.setattr(presets, "artifact_prefix", lambda cluster: str(tmp_path))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    panel = Path(presets.default_selection().path())
    tasks = [
        {
            "task_id": f"task-{i}",
            "family": ("large", "small")[i == 0],
            "aaii_category": ("coding", "format")[i == 0],
            "agentic": False,
        }
        for i in range(910)
    ]
    pq.write_table(pa.Table.from_pylist(tasks), panel / presets.TASK_MANIFEST_FILENAME)
    test_panel = Path(presets.selection_artifact(presets.EVALUATIONS["test_panel"].source).path())
    full_tasks = [{**tasks[0 if i == 0 else 1], "task_id": f"task-{i}"} for i in range(1500)]
    pq.write_table(pa.Table.from_pylist(full_tasks), test_panel / presets.TASK_MANIFEST_FILENAME)
    revision = presets.MODEL_REVISION
    rows = [
        {
            "task_id": f"task-{i}",
            "model_revision": revision,
            "passed": (0, 4, 8)[i % 3],
            "samples": 8,
            "mean_score": (0.05, 0.5, 0.95)[i % 3],
        }
        for i in range(910)
    ]
    panel_rates = tmp_path / "test-panel-rates" / scoring.PASS_RATES_FILENAME
    panel_rates.parent.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{**rows[i % len(rows)], "task_id": f"task-{i}"} for i in range(1500)]), panel_rates
    )

    def write(
        preset,
        name,
        *,
        speed=100.0,
        score=0.5,
        metrics=None,
        unscored=0,
        duplicate_task=False,
        drop_task=False,
        provenance=None,
        argv=(),
        initial_score=None,
        omit_step=None,
        version=presets.PRESET_REVISION,
        record_version=None,
        fixed=True,
        mixed=128,
        fingerprint="",
        answers=None,
        missing_receipt=False,
        unresolved_receipt=False,
        first_panel_task="task-0",
        quality_unscored=0,
        quality_metrics=None,
        initial_unscored=0,
        first_answer_task="task-0",
        attempt_id="recorded-attempt",
        panel_identity="",
        run_id=None,
    ):
        preset = presets.PRESETS[preset] if isinstance(preset, str) else preset
        root = tmp_path / name / version
        result_run = SkyRLRun(
            path=str(root),
            hf_model_uri=None,
            global_step=preset.updates,
            tokenizer_uri="tests/tokenizer",
            tokenizer_revision="test-revision",
            checkpoint_root=str(root / "checkpoints"),
            draft_checkpoint_root=None,
            terminal_manifest_uri=str(root / "terminal.json"),
            iris_job_id="/tests/native-run",
        )
        record = ArtifactRecord(
            name=name,
            version=record_version or version,
            fingerprint=fingerprint,
            result_type=result_type_name(SkyRLRun),
            result=result_run.result_payload(),
        )
        write_json(root / RECORD_FILENAME, record.model_dump(mode="json"))
        commit = gate.CONTROLS.get(gate.SHARED_CONTROLS.get(preset.name, preset.name), gate.CONTROLS["learning-medium"])[
            0
        ].main_commit
        source = {"tree_hash": "native-tree", "base_commit": commit, "dirty": False, "command_line": list(argv)}
        result = {
            "state": "succeeded",
            "iris_job_id": "/tests/native-run",
            "provenance": source | (provenance or {}),
            "started_at": "2026-11-06T18:40:00+00:00",
            "finished_at": "2026-11-06T19:40:00+00:00",
        }
        (root / ".executor_status").write_text("SUCCESS")
        for step in (0,) if preset.recipe.layout.role_plan.generate_only else range(1, preset.updates + 1):
            row = {f"generate/outcome/{outcome}": 0.0 for outcome in (*Outcome, "rollout_failure")}
            row |= {"step": step, "trainer/global_step": step}
            if not preset.recipe.layout.role_plan.generate_only:
                row["timing/step_wall/unaccounted"] = 0.0
            row |= {metric.key: 0.0 for metric in preset.reported if metric.key.startswith("timing/step_wall/")}
            row |= {
                "attempt_id": attempt_id,
                "generate/outcome/graded": 512.0,
                "loss/policy_loss": 0.01,
                "policy/policy_loss": 0.01,
                "policy/policy_update_steps": 1,
                "policy/raw_grad_norm": 1.0,
                "policy/mismatch/staleness0/kl_k3": speed / 1000,
                "policy/mismatch/pooled/kl_k3": speed / 1000,
                "policy/mismatch/staleness0/selected_tokens": 1000,
                "async/performance/loss_tokens_per_configured_policy_gpu_second": speed,
                "async/performance/response_tokens_per_configured_inference_gpu_second": 4 * speed,
                "generate/min_num_tokens": 4096 if fixed else 12,
                "generate/max_num_tokens": 4096,
                "generate/tokens_per_second_per_gpu": speed,
                "vllm/median_generation_throughput": 100.0,
                "startup/init_weight_sync_state": speed / 10,
                "timing/step_wall/group_admission": speed,
                "timing/step_wall/evaluation": 700,
                "timing/step_wall/checkpoint_work": 900,
            }
            if preset.recipe.layout.role_plan.generate_only:
                row["generate/wall_seconds"] = 1.0
            if step != omit_step:
                write_json(root / f"exports/metrics/step-{step:06d}.json", row | (metrics or {}))
        launch = {
            "run": {"id": run_id or f"{name}-{version}", "attempt_id": attempt_id},
            "runtime": {"task_env": {"NCCL_NET": "Socket", "VLLM_LOGGING_LEVEL": "ERROR"}},
            "iris": {"allocation": {"num_nodes": preset.recipe.layout.role_plan.num_nodes, "gpus_per_node": 8}},
            "skyrl": {
                "context_budget": {"max_new_tokens_per_turn": 8192, "max_turns": 1},
                "trainer": {
                    "seed": presets.SEED,
                    "max_steps": preset.updates,
                    "max_prompt_length": 8192,
                    "algorithm": {"batch_invariant": False},
                    "callbacks": [],
                    "policy": {"optimizer_config": {"lr": 1e-6}},
                },
                "generator": {
                    "eval_n_samples_per_prompt": preset.evaluations[0].answers if preset.evaluations else 1,
                    "eval_sampling_params": {"temperature": 0.0, "max_generate_length": 8192},
                    "backend": "vllm",
                    "model_dtype": "bfloat16",
                    "vllm_attention_backend": "FLASH_ATTN",
                    "enable_prefix_caching": True,
                    "max_num_seqs": 1024,
                    "max_num_batched_tokens": 8192,
                    "enable_chunked_prefill": True,
                    "enforce_eager": False,
                    "engine_init_kwargs": {},
                },
            },
            "inputs": {
                "model": {"identity": "initializer@2026.11.06"},
                "train_data": [{"identity": "training@2026.11.06"}],
                "validation_data": [],
                "evaluation_data": {},
            },
        }
        if unresolved_receipt:
            launch |= {
                "artifacts": {"attempts_root": str(root / "attempts")},
                "run": {"attempt_id": "missing"},
                "iris": {"job_name": "lookup-missing-job"},
            }

            @contextmanager
            def endpoint(**kwargs):
                yield SimpleNamespace(url="http://iris.test", credentials=None)

            @contextmanager
            def rpc(*args):
                yield SimpleNamespace(list_jobs=lambda request: controller_pb2.Controller.ListJobsResponse())

            monkeypatch.setattr(adapter, "connect_controller", endpoint)
            monkeypatch.setattr(adapter, "rpc_client", rpc)
        if initial_score is None and any(evaluation.at_start for evaluation in preset.evaluations):
            initial_score = 0.3
        sessions = {-1 if preset.updates == 0 else preset.updates: score}
        sessions |= {0: initial_score} if initial_score is not None else {}
        for index, evaluation in enumerate(preset.evaluations):
            task_options = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--tasks"]
            default = presets.default_selection()
            selected = evaluation.tasks(
                presets.selection_artifact(task_options[-1]) if task_options else default, default
            )
            run_panel = Path(selected.path())
            if first_panel_task != "task-0":
                original = pq.read_table(run_panel / presets.TASK_MANIFEST_FILENAME).to_pylist()
                run_panel = root / f"heldout-{evaluation.name}"
                run_panel.mkdir()
                (run_panel / "summary.json").write_bytes((Path(selected.path()) / "summary.json").read_bytes())
                manifest = [
                    {**row, "task_id": first_panel_task} if row["task_id"] == "task-0" else row for row in original
                ]
                pq.write_table(pa.Table.from_pylist(manifest), run_panel / presets.TASK_MANIFEST_FILENAME)
            count = evaluation.sample or presets.selection_summary(selected)[f"{evaluation.file}_rows"]
            identifiers = [
                f"train-task-{i}" if evaluation.file == "train" else f"task-{i}" if i else first_panel_task
                for i in range(count)
            ]
            task_path = run_panel / f"{evaluation.file}.parquet"
            if not task_path.exists():
                task = answer_task(
                    RawRow("fixture", Source(dataset="fixture", revision="1", row="fixture", importer_revision="1"), {}),
                    prompt="Return cat.",
                    spec=ExactSpec(("cat",)),
                )
                pq.write_table(
                    pa.table(
                        {"task_spec": [task.model_copy(update={"id": item}).model_dump_json() for item in identifiers]}
                    ),
                    task_path,
                )
            binding = selected
            if evaluation.sample is not None:
                binding = presets.sampled_tasks_step(selected, evaluation.file, evaluation.sample, presets.SEED)
            descriptor = {
                "uri": str(run_panel),
                "relative_path": f"{evaluation.file}.parquet",
                "kind": "directory",
                "identity": f"{binding.name}@{binding.version}",
                "local_path": str(run_panel),
            }
            if panel_identity is None:
                descriptor.pop("identity")
            elif panel_identity:
                descriptor["identity"] = panel_identity
            if index:
                launch["skyrl"]["trainer"]["callbacks"].append(
                    {
                        "type": "evaluation",
                        "additional_evaluations": {
                            evaluation.name: {
                                "sampling_params": {"temperature": 0.0, "max_generate_length": 8192},
                                "n_samples_per_prompt": evaluation.answers,
                            }
                        },
                    }
                )
                launch["inputs"]["evaluation_data"][evaluation.name] = [descriptor]
            else:
                launch["inputs"]["validation_data"] = [descriptor]
            for step, average in sessions.items():
                if index and not preset.recipe.layout.role_plan.generate_only and step == 0:
                    continue
                failed_answers = initial_unscored if step == 0 else quality_unscored if index else unscored
                rows = answers or [
                    answer(
                        first_answer_task if i == 0 and evaluation.file != "train" else identifier,
                        (float(i < mixed and attempt == 0) if evaluation.file == "train" else average),
                        tokens=(preset.recipe.fixed_reply_tokens or 1) if fixed and index == 0 else 1,
                        error="GraderUnavailable" if i < failed_answers else None,
                    )
                    for i, identifier in enumerate(identifiers[: len(identifiers) - drop_task])
                    for attempt in range(evaluation.answers)
                ]
                session = "eval_only" if step == -1 else f"global_step_{step}_evals"
                export = root / "exports" / evaluation.name if index else root / "exports"
                path = export / "dumped_evals" / session / "panel.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = rows[:1] * duplicate_task + rows
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                if evaluation.file != "train":
                    metric_step = 0 if step == -1 else step
                    metric_path = root / f"exports/metrics/step-{metric_step:06d}.json"
                    if metric_step != omit_step:
                        outcome_prefix = f"eval/{evaluation.name}/all/outcome/" if index else "eval/all/outcome/"
                        counts = {outcome_prefix + outcome: 0 for outcome in (*Outcome, "rollout_failure")}
                        failures = sum(bool(row.get("exception_type")) for row in rows)
                        counts.update(
                            {outcome_prefix + "graded": len(rows) - failures, outcome_prefix + "unavailable": failures}
                        )
                        previous = (
                            json.loads(metric_path.read_text())
                            if metric_path.exists()
                            else {"step": metric_step, "attempt_id": attempt_id}
                        )
                        updated = (
                            previous
                            | counts
                            | {
                                outcome_prefix.removesuffix("outcome/") + "num_attempted": len(rows),
                                outcome_prefix.removesuffix("outcome/") + "num_scored": len(rows) - failures,
                            }
                            | (quality_metrics or {})
                        )
                        for key, value in (quality_metrics or {}).items():
                            if value is None:
                                del updated[key]
                        write_json(metric_path, updated)
        write_json(
            root / f"exports/attempts/{attempt_id}/tracking.json",
            {
                "run_id": run_id or f"{name}-{version}",
                "attempt_id": attempt_id,
                "iris_job_id": "/tests/native-run",
                "coordinator_iris_job_id": "/tests/coordinator",
                "wandb_url": None,
                "started_at": "2026-11-06T17:20:00+00:00",
                "engine": {"vllm_version": "fixture-vllm"},
            },
        )
        if not (missing_receipt or unresolved_receipt):
            write_json(root / "terminal.json", {"config": launch, "result": result})
        if not missing_receipt:
            write_json(root / "resolved-launch.yaml", {"config": launch})
        return root

    return write


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def answer(task, score, *, low=0, high=1, status="verified", tokens=1, error=None):
    verification = {"status": status, "score": score, "score_min": low, "score_max": high}
    return {"task_id": task, "response_token_ids": [1] * tokens, "exception_type": error, "verification": verification}
