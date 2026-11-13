# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import marin.rl.skyrl as adapter
import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.run_records import (
    METRIC_METADATA_KEYS,
    AttemptRecord,
    LaunchReceipt,
    MetricRow,
    read_attempt,
    read_attempts,
    register_run_tables,
    run_records_root,
    update_attempt,
)
from finestore.store import DataStore
from iris.cluster.client.job_info import set_job_info
from iris.rpc import controller_pb2
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import StepContext
from marin.rl.skyrl import SkyRLRun
from rigging.provenance import Provenance
from taskcompendium.convert.answers import answer_task
from taskcompendium.grading_result import Outcome
from taskcompendium.models import Source
from taskcompendium.pipeline.models import RawRow
from verifyit.spec import ExactSpec

from experiments.post_training import snowball_rl_command as command
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_recipe as recipe
from experiments.post_training.skyrl_test_support import native_json as native_json

SELECTION_IDENTITY = "fixture-selection@2026.11.10"
NATIVE_RECORD = json.loads(Path(__file__).with_name("native_record.json").read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_receipt(root: Path, receipt: dict) -> None:
    write_json(root / "terminal.json", receipt)
    write_json(root / "resolved-launch.yaml", {"config": receipt["config"]})
    config = receipt["config"]
    export_root = config["artifacts"]["export_root"]
    attempt_id = config["run"]["attempt_id"]
    facts = (
        {
            "started_at": receipt["result"]["started_at"],
            "vllm_version": "fixture-vllm",
            "metrics_started_at": receipt["result"]["started_at"],
        }
        if read_attempt(export_root, attempt_id) is None
        else {}
    )
    native = LaunchReceipt.model_validate(receipt)
    with DataStore.open(run_records_root(export_root)) as store:
        update_attempt(
            store,
            AttemptRecord(
                run_id=config["run"]["id"], attempt_id=attempt_id, config=native.config, result=native.result, **facts
            ),
        )


def metric_rows(metrics: dict[int, dict]) -> Iterator[dict]:
    return (
        MetricRow(attempt_id=values["attempt_id"], step=step, name=name, value=value).model_dump()
        for step, values in metrics.items()
        for name, value in values.items()
        if name not in METRIC_METADATA_KEYS
    )


def write_metrics(root: Path, metrics: dict[int, dict]) -> None:
    export_root = str(root / "exports")
    records = read_attempts(export_root)
    shutil.rmtree(run_records_root(export_root), ignore_errors=True)
    with DataStore.open(run_records_root(export_root)) as store:
        tables = register_run_tables(store)
        for record in records.values():
            update_attempt(store, record)
        tables.metrics.extend(metric_rows(metrics))


def append_metrics(root: Path, metrics: dict[int, dict]) -> None:
    with DataStore.open(run_records_root(str(root / "exports"))) as store:
        register_run_tables(store).metrics.extend(metric_rows(metrics))


def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gate,
        "CONTROLS",
        {
            name: tuple(replace(pin, revision=presets.PRESETS[name].revision) for pin in pins)
            for name, pins in gate.CONTROLS.items()
        },
    )
    monkeypatch.setattr(command, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(presets, "SELECTION_ROOT", str(tmp_path))
    monkeypatch.setattr(presets, "artifact_prefix", lambda cluster: str(tmp_path))
    training = {preset.tasks for preset in presets.PRESETS.values()}
    task_sets = {preset.eval_lite for preset in presets.PRESETS.values()}
    summary = {
        "recipe": {"heldout": [{"version": "2026.10.23.1", "rows": 910}]},
        "training_rows": 100000,
        "validation_rows": 100,
        "full_validation_rows": 910,
        "grader_kind_counts": {"train": {}, "validation": {}, "full_validation": {}},
    }
    for binding in training:
        write_json(Path(presets.selection_artifact(binding).path()) / "summary.json", summary)
    candidate = presets.selection_artifact(SELECTION_IDENTITY)
    candidate_summary = deepcopy(summary)
    candidate_summary["recipe"]["exclusions"] = [
        {"uri": str(Path(task_set.path()) / presets.FULL_VALIDATION_FILENAME), "version": task_set.version}
        for binding in sorted(task_sets)
        for task_set in (presets.selection_artifact(binding),)
    ]
    write_json(Path(candidate.path()) / "summary.json", candidate_summary)
    for binding in task_sets:
        write_json(
            Path(presets.selection_artifact(binding).path()) / "summary.json",
            {
                "recipe": {},
                "validation_rows": 100,
                "full_validation_rows": 1500,
                "grader_kind_counts": {"validation": {}, "full_validation": {}},
            },
        )
    return candidate


def source_provenance(monkeypatch):
    monkeypatch.setattr(
        Provenance, "capture", lambda *args, **kwargs: Provenance("tree-one", "commit-one", False, "branch", "test")
    )


def answer(task, score, *, tokens=1, error=None, low=0, high=1, status="verified"):
    row = deepcopy(NATIVE_RECORD["answer"])
    row.update(task_id=task, response_token_ids=[1] * tokens, exception_type=error)
    row["verification"].update(status=status, score=score, score_min=low, score_max=high)
    return row


def recorded_runs(selection, tmp_path, monkeypatch, request):
    set_job_info(None)
    request.addfinalizer(lambda: set_job_info(None))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    for binding in {preset.eval_lite for preset in presets.PRESETS.values()}:
        task_set = Path(presets.selection_artifact(binding).path())
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "task_id": f"task-{i}",
                        "family": "small" if i == 0 else "large",
                        "aaii_category": "format" if i == 0 else "coding",
                        "agentic": False,
                    }
                    for i in range(1500)
                ]
            ),
            task_set / presets.TASK_MANIFEST_FILENAME,
        )

    @contextmanager
    def endpoint(**kwargs):
        yield SimpleNamespace(url="http://iris.test", credentials=None)

    @contextmanager
    def rpc(*args):
        yield SimpleNamespace(list_jobs=lambda request: controller_pb2.Controller.ListJobsResponse())

    monkeypatch.setattr(adapter, "connect_controller", endpoint)
    monkeypatch.setattr(adapter, "rpc_client", rpc)
    task = answer_task(
        RawRow("fixture", Source(dataset="fixture", revision="1", row="fixture", importer_revision="1"), {}),
        prompt="Return cat.",
        spec=ExactSpec(("cat",)),
    )

    templates = {}

    def composed(preset, inputs):
        key = (repr(preset), inputs.model_dump_json())
        if key not in templates:
            source_prefix = tmp_path / "composed" / str(len(templates))
            spec = recipe.run_spec(
                preset,
                name="fixture",
                tasks=presets.selection_artifact(inputs.tasks) if inputs.tasks else None,
                init=presets.published_artifact(inputs.init) if inputs.init else None,
                settings=inputs.settings,
            )
            step = recipe.run_step(preset, spec)
            source_root = Path(step.path(str(source_prefix)))
            source_root.mkdir(parents=True)
            document = source_root / "launch.yaml"
            context = StepContext.for_run(
                str(source_root),
                presets.artifact_prefix(preset.recipe.layout.cluster),
                runtime_args=step.runtime_args,
                deps=step.deps,
            )
            document.write_text(step.build_config(context).launch_config_yaml)
            config = gate.compose_configs([str(document)])[0]
            templates[key] = (source_root, config)
        return templates[key]

    def write(
        preset_name: str,
        name,
        *,
        speed=100.0,
        score=0.5,
        initial_score=0.3,
        metrics=None,
        inputs=None,
        fingerprint="",
        legacy=False,
        missing_steps=(),
        attempt_id="recorded-attempt",
    ):
        preset = presets.PRESETS[preset_name]
        root = tmp_path / name / preset.revision
        root.mkdir(parents=True, exist_ok=True)
        inputs = inputs or gate.GateInputs(preset_name=preset.name)
        source_root, config = composed(preset, inputs)
        old_id, old_attempt = config["run"]["id"], config["run"]["attempt_id"]
        attempt = attempt_id
        run_id = f"{name}-{preset.revision}"

        def relocate(value):
            if isinstance(value, dict):
                return {key: relocate(child) for key, child in value.items()}
            if isinstance(value, list):
                return [relocate(child) for child in value]
            if isinstance(value, str):
                return value.replace(str(source_root), str(root)).replace(old_id, run_id).replace(old_attempt, attempt)
            return value

        native = {"config": relocate(config), "result": deepcopy(NATIVE_RECORD["receipt"]["result"])}
        launch, result = native["config"], native["result"]
        pin = gate.control_pins(preset)[0] if preset.gated else None
        result.update(
            iris_job_id="/tests/native-run",
            started_at="2026-11-06T18:40:00+00:00",
            finished_at="2026-11-06T19:40:00+00:00",
        )
        result["provenance"].update(
            base_commit=pin.commit if pin else "commit-one",
            dirty=False,
            command_line=["unrelated-command", "--preset", "train-long"],
        )
        write_json(root / gate.GATE_INPUTS_FILENAME, inputs.model_dump(mode="json"))
        write_json(
            root / RECORD_FILENAME,
            ArtifactRecord(
                name=name,
                version=preset.revision,
                fingerprint=fingerprint,
                result_type=result_type_name(SkyRLRun),
                result=SkyRLRun(
                    path=str(root),
                    hf_model_uri=None,
                    global_step=preset.updates,
                    tokenizer_uri="tests/tokenizer",
                    tokenizer_revision="test-revision",
                    checkpoint_root=str(root / "checkpoints"),
                    draft_checkpoint_root=None,
                    terminal_manifest_uri=str(root / "terminal.json"),
                    iris_job_id="/tests/native-run",
                ).result_payload(),
            ).model_dump(mode="json"),
        )
        (root / ".executor_status").write_text("SUCCESS")
        generating = isinstance(preset.recipe.layout.workload, presets.Generation)
        stored_metrics = {}
        for step in (0,) if generating else range(1, preset.updates + 1):
            row = deepcopy(NATIVE_RECORD["metrics"])
            row = {key: value for key, value in row.items() if not key.startswith("timing/step_wall/")}
            row.update(
                {
                    "step": step,
                    "trainer/global_step": step,
                    "timing/step_wall/unaccounted": 0,
                    "timing/step_wall/group_admission": speed,
                    "timing/step_wall/evaluation": 700,
                    "timing/step_wall/checkpoint_work": 900,
                    "generate/min_num_tokens": 4096,
                    "generate/max_num_tokens": 4096,
                    "policy/mismatch/staleness0/kl_k3": speed / 1000,
                    "policy/mismatch/staleness0/selected_tokens": 1000,
                    "policy/mismatch/staleness0/finite_fraction": 1.0,
                    "startup/init_weight_sync_state": speed / 10,
                    "async/performance/loss_tokens_per_configured_policy_gpu_second": speed,
                    "async/performance/response_tokens_per_configured_inference_gpu_second": 4 * speed,
                }
            )
            row.update(
                {
                    "generate/outcome/" + value: 512 if value == Outcome.GRADED else 0
                    for value in (*Outcome, "rollout_failure")
                }
            )
            if generating:
                row = {
                    key: value
                    for key, value in row.items()
                    if not key.startswith(("generate/outcome/", "loss/", "policy/", "timing/step_wall/"))
                }
                row.update({"generate/wall_seconds": 1.0, "generate/tokens_per_second_per_gpu": speed})
            stored_metrics[step] = row
        default, selected = presets.task_selections(
            preset, presets.selection_artifact(inputs.tasks) if inputs.tasks else None
        )
        for index, evaluation in enumerate(preset.evaluations):
            source = evaluation.tasks(selected, default, preset.eval_lite)
            sources = (
                launch["inputs"]["validation_data"]
                if index == 0
                else launch["inputs"]["evaluation_data"][evaluation.name]
            )
            assert len(sources) == 1
            descriptor = sources[0]
            directory = Path(descriptor["uri"])
            directory.mkdir(parents=True, exist_ok=True)
            count = evaluation.sample or presets.selection_summary(source)[f"{evaluation.file}_rows"]
            ids = [f"train-task-{i}" if evaluation.file == "train" else f"task-{i}" for i in range(count)]
            path = directory / descriptor["relative_path"]
            if not path.exists():
                pq.write_table(
                    pa.table({"task_spec": [task.model_copy(update={"id": item}).model_dump_json() for item in ids]}),
                    path,
                )
            sessions = [(-1 if generating else preset.updates, score)]
            if evaluation.at_start:
                sessions.insert(0, (0, initial_score))
            for step, average in sessions:
                rows = [
                    answer(
                        item,
                        float(i < int(count * average) and sample == 0) if evaluation.file == "train" else average,
                        tokens=preset.recipe.fixed_reply_tokens or 1 if index == 0 else 1,
                    )
                    for i, item in enumerate(ids)
                    for sample in range(evaluation.answers)
                ]
                export = root / "exports" / evaluation.name if index else root / "exports"
                session = "eval_only" if step == -1 else f"global_step_{step}_evals"
                path = export / "dumped_evals" / session / "task_set.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                values = stored_metrics.setdefault(max(step, 0), {"step": max(step, 0)})
                prefix = f"eval/{evaluation.name}/all/outcome/" if index else "eval/all/outcome/"
                values.update(
                    {
                        prefix + value: len(rows) if value == Outcome.GRADED else 0
                        for value in (*Outcome, "rollout_failure")
                    }
                )
                if generating and legacy and index == 0:
                    values.update(
                        {"generate/outcome/" + value: values[prefix + value] for value in (*Outcome, "rollout_failure")}
                    )
        for values in stored_metrics.values():
            values.update(metrics or {})
            values["attempt_id"] = attempt
        write_metrics(root, {step: values for step, values in stored_metrics.items() if step not in missing_steps})
        write_receipt(root, native)
        return root

    return write
