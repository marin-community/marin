# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import marin.rl.skyrl as adapter
import pyarrow as pa
import pyarrow.parquet as pq
from iris.cluster.client.job_info import set_job_info
from iris.rpc import controller_pb2
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import artifact_identity
from marin.rl.skyrl import SkyRLRun, skyrl_cpu_command
from rigging.provenance import Provenance
from taskcompendium.convert.answers import answer_task
from taskcompendium.grading_result import Outcome
from taskcompendium.models import Source
from taskcompendium.pipeline.models import RawRow
from verifyit.spec import ExactSpec

from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets

SELECTION_IDENTITY = "fixture-selection@2026.11.10"
NATIVE_RECORD = json.loads(Path(__file__).with_name("native_record.json").read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_receipt(root: Path, receipt: dict) -> None:
    write_json(root / "terminal.json", receipt)
    write_json(root / "resolved-launch.yaml", {"config": receipt["config"]})


def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gate,
        "CONTROLS",
        {
            name: tuple(replace(pin, revision=presets.PRESET_REVISION) for pin in pins)
            for name, pins in gate.CONTROLS.items()
        },
    )
    monkeypatch.setattr(gate, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(presets, "SELECTION_ROOT", str(tmp_path))
    monkeypatch.setattr(presets, "artifact_prefix", lambda cluster: str(tmp_path))
    root = tmp_path / "snowball-100k" / presets.DEFAULT_SELECTION_VERSION
    panel = tmp_path / presets.DEFAULT_EVAL_LITE_NAME / presets.DEFAULT_EVAL_LITE_VERSION
    summary = {
        "recipe": {"heldout": [{"version": "2026.10.23.1", "rows": 910}]},
        "training_rows": 100000,
        "validation_rows": 100,
        "full_validation_rows": 910,
        "grader_kind_counts": {"train": {}, "validation": {}, "full_validation": {}},
    }
    write_json(root / "summary.json", summary)
    candidate = presets.selection_artifact(SELECTION_IDENTITY)
    candidate_summary = deepcopy(summary)
    candidate_summary["recipe"]["exclusions"] = [
        {"uri": str(panel / presets.FULL_VALIDATION_FILENAME), "version": panel.name}
    ]
    write_json(Path(candidate.path()) / "summary.json", candidate_summary)
    write_json(
        panel / "summary.json",
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


def answer(task, score, *, tokens=1, error=None, low=0, high=1, status="verified"):
    row = deepcopy(NATIVE_RECORD["answer"])
    row.update(task_id=task, response_token_ids=[1] * tokens, exception_type=error)
    row["verification"].update(status=status, score=score, score_min=low, score_max=high)
    return row


def recorded_runs(selection, tmp_path, monkeypatch, request):
    set_job_info(None)
    request.addfinalizer(lambda: set_job_info(None))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    panel = Path(presets.selection_artifact(presets.EVALUATIONS["eval_lite"].source).path())
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
        panel / presets.TASK_MANIFEST_FILENAME,
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
        attempt_id="recorded-attempt",
    ):
        preset = presets.PRESETS[preset_name]
        root = tmp_path / name / presets.PRESET_REVISION
        root.mkdir(parents=True, exist_ok=True)
        native = deepcopy(NATIVE_RECORD["receipt"])
        launch, result = native["config"], native["result"]
        attempt = attempt_id
        launch["run"].update(id=f"{name}-{presets.PRESET_REVISION}", attempt_id=attempt)
        launch["artifacts"].update(export_root=str(root / "exports"), attempts_root=str(root / "attempts"))
        launch["iris"]["allocation"].update(num_nodes=preset.recipe.layout.role_plan.num_nodes, gpus_per_node=8)
        launch["skyrl"]["trainer"].update(max_steps=preset.updates, seed=presets.SEED, callbacks=[])
        launch["inputs"]["validation_data"] = []
        launch["inputs"]["evaluation_data"] = {}
        pin = gate.control_pins(preset.name)[0] if presets.gated(preset) else None
        result.update(
            iris_job_id="/tests/native-run",
            started_at="2026-11-06T18:40:00+00:00",
            finished_at="2026-11-06T19:40:00+00:00",
        )
        result["provenance"].update(
            base_commit=pin.commit if pin else result["provenance"]["base_commit"],
            dirty=False,
            command_line=["unrelated-command", "--preset", "train-long"],
        )
        inputs = inputs or gate.GateInputs(preset_name=preset.name)
        write_json(root / gate.GATE_INPUTS_FILENAME, inputs.model_dump(mode="json"))
        write_json(
            root / RECORD_FILENAME,
            ArtifactRecord(
                name=name,
                version=presets.PRESET_REVISION,
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
        generating = preset.recipe.layout.role_plan.generate_only
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
            row.pop("attempt_id", None)
            if legacy:
                row["attempt_id"] = attempt
                if not generating:
                    row["reward/rloo_n_total_groups"] = 64
            if generating:
                row = {
                    key: value
                    for key, value in row.items()
                    if not key.startswith(("generate/outcome/", "loss/", "policy/", "timing/step_wall/"))
                }
                row.update({"generate/wall_seconds": 1.0, "generate/tokens_per_second_per_gpu": speed})
            write_json(root / f"exports/metrics/step-{step:06d}.json", row)
        default = presets.default_selection()
        selected = presets.selection_artifact(inputs.tasks) if inputs.tasks else default
        for index, evaluation in enumerate(preset.evaluations):
            source = evaluation.tasks(selected, default)
            directory = Path(source.path())
            count = evaluation.sample or presets.selection_summary(source)[f"{evaluation.file}_rows"]
            ids = [f"train-task-{i}" if evaluation.file == "train" else f"task-{i}" for i in range(count)]
            path = directory / f"{evaluation.file}.parquet"
            if not path.exists():
                pq.write_table(
                    pa.table({"task_spec": [task.model_copy(update={"id": item}).model_dump_json() for item in ids]}),
                    path,
                )
            binding = (
                presets.sampled_tasks_step(source, evaluation.file, evaluation.sample, presets.SEED)
                if evaluation.sample
                else source
            )
            descriptor = deepcopy(NATIVE_RECORD["receipt"]["config"]["inputs"]["validation_data"][0])
            descriptor.update(
                uri=str(directory),
                relative_path=path.name,
                local_path=str(directory),
                identity=artifact_identity(binding),
            )
            sampling = {
                **launch["skyrl"]["generator"]["eval_sampling_params"],
                "temperature": 0.0,
                "max_generate_length": 8192,
            }
            if index:
                launch["inputs"]["evaluation_data"][evaluation.name] = [descriptor]
                launch["skyrl"]["trainer"]["callbacks"].append(
                    {
                        "type": "evaluation",
                        "additional_evaluations": {
                            evaluation.name: {"sampling_params": sampling, "n_samples_per_prompt": evaluation.answers}
                        },
                    }
                )
            else:
                launch["inputs"]["validation_data"] = [descriptor]
                launch["skyrl"]["generator"].update(
                    eval_n_samples_per_prompt=evaluation.answers, eval_sampling_params=sampling
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
                path = export / "dumped_evals" / session / "panel.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                metric_path = root / f"exports/metrics/step-{max(step, 0):06d}.json"
                values = json.loads(metric_path.read_text()) if metric_path.exists() else {"step": max(step, 0)}
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
                write_json(metric_path, values)
        for path in (root / "exports/metrics").glob("step-*.json"):
            values = json.loads(path.read_text())
            values.update(metrics or {})
            write_json(path, values)
        write_json(
            root / f"exports/attempts/{attempt}/tracking.json",
            {
                "run_id": launch["run"]["id"],
                "attempt_id": attempt,
                "iris_job_id": "/tests/native-run",
                "coordinator_iris_job_id": "/tests/coordinator",
                "started_at": result["started_at"],
                "engine": {"vllm_version": "fixture-vllm"},
            },
        )
        write_receipt(root, native)
        return root

    return write
