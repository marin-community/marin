# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import marin.rl.skyrl as adapter
import pyarrow as pa
import pyarrow.parquet as pq
from iris.cluster.client.job_info import set_job_info
from iris.resources.state import JobState
from iris.rpc import controller_pb2
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.rl.skyrl import SkyRLRun
from rigging.timing import Timestamp
from taskcompendium.convert.answers import answer_task
from taskcompendium.grading_result import Outcome
from taskcompendium.models import Source
from taskcompendium.pipeline.models import RawRow
from verifyit.spec import ExactSpec

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.inventory import loop_publication
from experiments.post_training.inventory import publish_loop_result as publication
from experiments.post_training.task_curation.inventory import TASK_SPEC_COLUMN
from infra.marina.apps.rl_progress.server.app import comparison_fields


def publication_transport(monkeypatch):

    transport = ReportTransport()
    monkeypatch.setattr(publication, "marina_request", transport)
    return transport


def recorded_runs(selection, tmp_path, monkeypatch, publication_transport, request):

    set_job_info(None)
    request.addfinalizer(lambda: set_job_info(None))
    native_job = SimpleNamespace(
        started_at=Timestamp.from_ms(1794000000000), finished_at=Timestamp.from_ms(1794003600000)
    )
    monkeypatch.setattr(
        loop_publication,
        "open_iris_client",
        lambda **kwargs: nullcontext(SimpleNamespace(job_status=lambda job_id: native_job)),
    )
    status_job = SimpleNamespace(state=JobState.SUCCEEDED, started_at=None, finished_at=None)
    monkeypatch.setattr(
        snowball, "open_iris_client", lambda **kwargs: nullcontext(SimpleNamespace(job_status=lambda job_id: status_job))
    )
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
    test_panel = Path(presets.selection_artifact(presets.DEFAULT_TEST_PANEL).path())
    full_tasks = [{**tasks[0 if i == 0 else 1], "task_id": f"task-{i}"} for i in range(1500)]
    pq.write_table(pa.Table.from_pylist(full_tasks), test_panel / presets.TASK_MANIFEST_FILENAME)
    for directory, manifest in ((panel, tasks), (test_panel, full_tasks)):
        for filename, selected in (("full_validation", manifest), ("validation", manifest[:100])):
            task_parquet(directory / f"{filename}.parquet", [row["task_id"] for row in selected])
    rates = tmp_path / scoring.PASS_RATES_NAME / scoring.PASS_RATE_REVISION / "pass_rates.parquet"
    rates.parent.mkdir(parents=True)
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
    pq.write_table(pa.Table.from_pylist(rows), rates)
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
        source_first_task="task-0",
        panel_identity="",
    ):
        preset = presets.PRESETS[preset] if isinstance(preset, str) else preset
        root = tmp_path / name / version
        result = SkyRLRun(
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
            result=result.result_payload(),
        )
        write_json(root / RECORD_FILENAME, record.model_dump(mode="json"))
        commit = gate.CONTROLS.get(preset.name, gate.CONTROLS["learning-medium"])[0].main_commit
        source = {"tree_hash": "native-tree", "base_commit": commit, "dirty": False, "command_line": list(argv)}
        result = {"state": "succeeded", "iris_job_id": "/tests/native-run", "provenance": source | (provenance or {})}
        if not (missing_receipt or unresolved_receipt):
            write_json(root / "terminal.json", {"config": {}, "result": result})
        (root / ".executor_status").write_text("SUCCESS")
        for step in (0,) if preset.recipe.layout.role_plan.generate_only else range(1, preset.updates + 1):
            row = {f"generate/outcome/{outcome}": 0.0 for outcome in (*Outcome, "rollout_failure")}
            row |= {"step": step, "trainer/global_step": step}
            if not preset.recipe.layout.role_plan.generate_only:
                row["timing/step_wall/unaccounted"] = 0.0
            row |= {metric.key: 0.0 for metric in preset.reported if metric.key.startswith("timing/step_wall/")}
            row |= {
                "attempt_id": "recorded-attempt",
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
            "run": {"id": f"{name}-{version}", "attempt_id": "recorded-attempt"},
            "skyrl": {
                "trainer": {"seed": presets.SEED, "max_steps": preset.updates},
                "generator": {"eval_n_samples_per_prompt": preset.evaluations[0].answers if preset.evaluations else 1},
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
                manifest = [
                    {**row, "task_id": first_panel_task} if row["task_id"] == "task-0" else row for row in original
                ]
                pq.write_table(pa.Table.from_pylist(manifest), run_panel / presets.TASK_MANIFEST_FILENAME)
                task_parquet(run_panel / f"{evaluation.file}.parquet", [row["task_id"] for row in manifest])
            count = evaluation.sample or presets.selection_summary(selected)[f"{evaluation.file}_rows"]
            identifiers = [
                f"train-task-{i}" if evaluation.file == "train" else f"task-{i}" if i else first_panel_task
                for i in range(count)
            ]
            if evaluation.sample is not None:
                run_panel = root / f"input-{evaluation.name}"
                task_parquet(run_panel / f"{evaluation.file}.parquet", identifiers)
            elif source_first_task != "task-0":
                manifest = pq.read_table(run_panel / presets.TASK_MANIFEST_FILENAME).to_pylist()
                run_panel = root / f"recorded-{evaluation.name}"
                run_panel.mkdir()
                pq.write_table(
                    pa.Table.from_pylist([*manifest, {**manifest[0], "task_id": source_first_task}]),
                    run_panel / presets.TASK_MANIFEST_FILENAME,
                )
                task_parquet(run_panel / f"{evaluation.file}.parquet", [source_first_task, *identifiers[1:]])
            binding = selected
            if evaluation.sample is not None:
                binding = presets.sampled_tasks_step(
                    selected, evaluation.file, evaluation.sample, presets.SEED, local_only=evaluation.local_only
                )
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
                launch["inputs"]["evaluation_data"][evaluation.name] = [descriptor]
            else:
                launch["inputs"]["validation_data"] = [descriptor]
            for step, average in sessions.items():
                failed_answers = initial_unscored if step == 0 else quality_unscored if index else unscored
                rows = answers or [
                    answer(
                        identifier,
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
                            else {"step": metric_step, "attempt_id": "recorded-attempt"}
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


def task_parquet(path, identifiers):
    template = answer_task(
        RawRow("template", Source(dataset="fixture", revision="1", row="template", importer_revision="1"), {}),
        prompt="Reply yes.",
        spec=ExactSpec(expected=("yes",)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{TASK_SPEC_COLUMN: template.model_copy(update={"id": item}).model_dump_json()} for item in identifiers]
        ),
        path,
    )


def result_record():
    return {
        "run_id": "test/control",
        "attempt_id": "attempt-1",
        "label": "control",
        "run_date": "2026-11-02T00:00:00Z",
        "date_basis": "submission_acknowledged",
        "started_at": "2026-11-02T00:00:00Z",
        "completed_at": "2026-11-02T01:00:00Z",
        "elapsed_seconds": 3600,
        "timing_scope": "coordinator_submission_to_terminal",
        "status": "succeeded",
        "optimizer_steps": 25,
        "source_revision": "source-1",
        "data_identity": "data-1",
        "initializer_identity": "model-1",
        "evaluation": {
            "name": "fast",
            "version": "1",
            "task_count": 100,
            "task_set_sha256": "a" * 64,
            "protocol_sha256": None,
        },
        "baseline": {"reward": 0.2, "attempted": 100, "scored": 100},
        "final": {"reward": 0.4, "attempted": 100, "scored": 100},
        "artifact_uri": "s3://example/test",
        "evidence": ["synthetic unit-test result"],
    }


class ReportTransport:
    def __init__(self):
        self.rows = {}
        self.requests = []
        self.fail_readback = False

    def __call__(self, origin, method, path, *, json_body=None):
        self.requests.append((origin, method, path))
        if path.startswith("/api/marina/applets/"):
            return {"authenticated_url": f"https://applets.marina.oa.dev/a/{publication.APPLET_ID}/"}
        if method == "POST":
            parameters = json_body["parameters"]
            key = (parameters["run_id"], parameters["attempt_id"])
            if "record" in parameters:
                if key in self.rows:
                    return {"row_count": 0}
                self.rows[key] = json.loads(parameters["record"])
                return {"row_count": 1}
            if self.fail_readback:
                self.fail_readback = False
                raise ConnectionError("synthetic outage after insertion")
            return {"rows": [{"record": copy.deepcopy(self.rows[key])}]}
        return {"runs": [record | comparison_fields(record) for record in self.rows.values()]}
