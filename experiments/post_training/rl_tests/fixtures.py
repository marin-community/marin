# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import urlencode

import marin.rl.skyrl as adapter
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from finestore.reader import ReadView
from finestore.rl.run_records import (
    EVALUATIONS_TABLE,
    METRIC_METADATA_KEYS,
    METRICS_TABLE,
    AttemptRecord,
    EvaluationAnswerRow,
    EvaluationTaskRow,
    LaunchReceipt,
    MetricRow,
    read_attempt,
    read_attempts,
    register_run_tables,
    run_records_root,
    update_attempt,
    write_evaluations,
)
from finestore.rl.strata import (
    MembershipState,
    SelectionDefinition,
    SelectionManifest,
    SelectionTaskRow,
    TaskStrata,
    read_selection_manifest,
)
from finestore.schema import arrow_schema
from finestore.store import DataStore
from fsspec.config import conf
from iris.client.client import IrisClient
from iris.cluster.client.job_info import set_job_info
from iris.cluster.types import JobName
from iris.rpc import controller_pb2
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import StepContext
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance
from taskcompendium.convert.answers import answer_task
from taskcompendium.grading_result import Outcome
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import Source
from taskcompendium.pipeline.models import RawRow
from verifyit.spec import ExactSpec

from experiments.post_training import rl as runner
from experiments.post_training import rl_command as command
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_presets as presets
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_measurements import RUN_INPUTS_FILENAME, RunInputs
from experiments.post_training.skyrl_test_support import (
    selection_summary,
    with_current_control_revisions,
    write_training_families,
)
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    TASK_MANIFEST_FILENAME,
    VALIDATION_FILENAME,
)

SELECTION_IDENTITY = "fixture-selection@2026.11.10"
NATIVE_RECORD = json.loads(Path(__file__).with_name("native_record.json").read_text())


def printed_status(output):
    # CliRunner merges stderr into result.output; status is on result.stdout. A plan prints its
    # lowered steps after the status, so the YAML ends where the first StepSpec begins.
    output = output.partition("\nStepSpec(")[0]
    marker = output.rfind("\npreset:")
    return yaml.safe_load(output[marker + 1 if marker >= 0 else 0 :].split("\n# ", 1)[0])


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def cache_native_compositions(monkeypatch) -> None:
    """Reuse successful compositions of equivalent launch documents within one test."""
    compose_configs = gate.compose_configs
    cached = {}
    # Attempt IDs also appear in derived fields such as the Iris job name.
    attempt_placeholder = "<generated-attempt-id>"

    def compose(paths, *, validate_driver=False):
        documents = [Path(path).read_text() for path in paths]
        attempts = [yaml.safe_load(document)["run"]["attempt_id"] for document in documents]
        key = validate_driver, tuple(
            document.replace(attempt, attempt_placeholder) for document, attempt in zip(documents, attempts, strict=True)
        )
        if key not in cached:
            configs = compose_configs(paths, validate_driver=validate_driver)
            cached[key] = [
                json.dumps(config).replace(attempt, attempt_placeholder)
                for config, attempt in zip(configs, attempts, strict=True)
            ]
        return [
            json.loads(config.replace(attempt_placeholder, attempt))
            for config, attempt in zip(cached[key], attempts, strict=True)
        ]

    monkeypatch.setattr(gate, "compose_configs", compose)


def fake_s3(monkeypatch, *paths: str) -> None:
    allowed = set(paths)
    monkeypatch.setattr("s3fs.S3FileSystem.exists", lambda _, path, **kw: str(path).removeprefix("s3://") in allowed)


def fake_s3_checkpoint(monkeypatch, uri: str, root: Path) -> None:
    # Cached s3fs instances can carry sync wrappers created before the patch.
    monkeypatch.setitem(conf, "s3", {**conf.get("s3", {}), "skip_instance_cache": True})
    prefix = uri.removeprefix("s3://")

    def local(path):
        name = str(path).removeprefix("s3://")
        if name == prefix or name.startswith(prefix + "/"):
            return root / name.removeprefix(prefix).lstrip("/")
        return None

    def files(_self, path, **_kwargs):
        pattern = local(path)
        return (
            []
            if pattern is None
            else [f"{prefix}/{entry.relative_to(root)}" for entry in root.glob(str(pattern.relative_to(root)))]
        )

    monkeypatch.setattr("s3fs.S3FileSystem.glob", files)
    monkeypatch.setattr(
        "s3fs.S3FileSystem.exists", lambda _, path, **kw: (path := local(path)) is not None and path.exists()
    )
    monkeypatch.setattr(
        "s3fs.S3FileSystem.modified", lambda _, path: datetime.fromtimestamp(local(path).stat().st_mtime, UTC)
    )


def write_artifact_record(root: Path, name: str, version: str, kind: type, **fields) -> None:
    record = ArtifactRecord(name=name, version=version, result_type=result_type_name(kind), **fields)
    write_json(root / RECORD_FILENAME, record.model_dump(mode="json"))


def write_published_artifact(root: Path, name: str, version: str, kind: type, *, source: str, config=None) -> None:
    write_artifact_record(root / name / version, name, version, kind, config=config, source=source)


def write_skyrl_run_record(root: Path, name: str, version: str, *, fingerprint: str, **fields) -> None:
    values = dict(
        path=str(root),
        hf_model_uri=None,
        global_step=0,
        tokenizer_uri="tests/tokenizer",
        tokenizer_revision="test-revision",
        checkpoint_root=str(root / "checkpoints"),
        draft_checkpoint_root=None,
        terminal_manifest_uri=str(root / "terminal.json"),
        iris_job_id="/tests/native-run",
    )
    run = SkyRLRun(**(values | fields))
    write_artifact_record(root, name, version, SkyRLRun, fingerprint=fingerprint, result=run.result_payload())


def fake_iris(monkeypatch, name):
    client = MagicMock(spec=IrisClient)
    job = JobName.from_wire(f"/test-user/{name}")
    client.submit.return_value.job_id = job
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kw: nullcontext(client))
    monkeypatch.setattr(runner, "open_iris_client", lambda **kw: nullcontext(client))
    return client, job


def fake_publication(monkeypatch, capture, origin):
    def publish(record, service_url=None):
        capture(record)
        return (service_url or origin) + "/#" + urlencode({"run": record.run_id, "attempt": record.attempt_id})

    monkeypatch.setattr("experiments.post_training.rl_publication.publish_result", publish)


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
            "evaluation_storage_version": 1,
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


def _rewrite_records(root: Path, *, metrics: list[dict] | None = None, evaluations: list[dict] | None = None) -> None:
    export_root = str(root / "exports")
    records = read_attempts(export_root)
    view = ReadView(run_records_root(export_root))
    metrics = list(view.iter_rows(METRICS_TABLE, columns=list(MetricRow.model_fields))) if metrics is None else metrics
    evaluations = (
        list(view.iter_rows(EVALUATIONS_TABLE, columns=list(EvaluationTaskRow.model_fields)))
        if evaluations is None
        else evaluations
    )
    shutil.rmtree(run_records_root(export_root), ignore_errors=True)
    with DataStore.open(run_records_root(export_root)) as store:
        tables = register_run_tables(store)
        for record in records.values():
            update_attempt(store, record)
        tables[METRICS_TABLE].extend(metrics)
        tables[EVALUATIONS_TABLE].extend(evaluations)


def write_metrics(root: Path, metrics: dict[int, dict]) -> None:
    _rewrite_records(root, metrics=list(metric_rows(metrics)))


def replace_evaluations(root: Path, rows: list[EvaluationTaskRow]) -> None:
    _rewrite_records(root, evaluations=[row.model_dump() for row in rows])


def copy_evaluations(root: Path, old_attempt: str, new_attempt: str) -> None:
    export_root = str(root / "exports")
    rows = list(
        ReadView(run_records_root(export_root)).iter_rows(
            EVALUATIONS_TABLE, columns=list(EvaluationTaskRow.model_fields), where=[("attempt_id", "==", old_attempt)]
        )
    )
    with DataStore.open(run_records_root(export_root)) as store:
        register_run_tables(store)[EVALUATIONS_TABLE].extend(dict(row, attempt_id=new_attempt) for row in rows)


@contextmanager
def edit_receipt(root: Path) -> Iterator[dict]:
    receipt = json.loads((root / "terminal.json").read_text())
    yield receipt
    write_receipt(root, receipt)


def controls(write, name, **kwargs):
    pin = gate.control_pin(snowball_model.SNOWBALL.by_name[name], table=snowball_model.SNOWBALL)
    return [
        write(
            name,
            gate.control_name(pin.preset, pin.commit, index, table=snowball_model.SNOWBALL),
            speed=100 + index,
            score=0.49 + index * 0.01,
            **kwargs,
        )
        for index in range(pin.replicates)
    ]


def preset_table(monkeypatch):
    def select(preset):
        table = replace(
            snowball_model.SNOWBALL,
            presets=tuple(preset if row.name == preset.name else row for row in snowball_model.SNOWBALL.presets),
        )
        monkeypatch.setattr(snowball_model, "SNOWBALL", table)
        return table

    return select


def append_metrics(root: Path, metrics: dict[int, dict]) -> None:
    with DataStore.open(run_records_root(str(root / "exports"))) as store:
        register_run_tables(store)[METRICS_TABLE].extend(metric_rows(metrics))


def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(command, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(recipe, "SELECTION_ROOT", str(tmp_path / recipe.SELECTION_PREFIX))
    monkeypatch.setattr(recipe, "artifact_prefix", lambda cluster: str(tmp_path))
    training = {preset.tasks for preset in snowball_model.SNOWBALL.by_name.values()}
    task_sets = {preset.held_out_tasks for preset in snowball_model.SNOWBALL.presets if preset.held_out_tasks}
    summary = selection_summary(910)
    for binding in training:
        write_json(Path(recipe.selection_artifact(binding).path()) / "summary.json", summary)
    candidate = recipe.selection_artifact(SELECTION_IDENTITY)
    candidate_summary = deepcopy(summary)
    counts = {"agentic": 256, "nonagentic": 256}
    candidate_summary["training_family_counts"] = counts
    candidate_summary["recipe"]["exclusions"] = [
        {"uri": str(Path(task_set.path()) / FULL_VALIDATION_FILENAME), "version": task_set.version}
        for binding in sorted(task_sets)
        for task_set in (recipe.selection_artifact(binding),)
    ]
    write_json(Path(candidate.path()) / "summary.json", candidate_summary)
    write_training_families(StoragePath(candidate.path()), counts)
    for binding in task_sets:
        write_json(
            Path(recipe.selection_artifact(binding).path()) / "summary.json",
            selection_summary(20),
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


def evaluation_task_rows(
    rows: Sequence[dict],
    *,
    attempt_id: str,
    step: int,
    evaluation: str,
    selection: SelectionManifest | None,
    normalized: Sequence[float | None] | None = None,
) -> list[EvaluationTaskRow]:
    by_task = defaultdict(list)
    for ordinal, row in enumerate(rows):
        verification = row.get("verification") or {}
        raw, low, high = (verification.get(key) for key in ("score", "score_min", "score_max"))
        score = normalized[ordinal] if normalized is not None else (raw - low) / (high - low)
        by_task[row["task_id"]].append(
            EvaluationAnswerRow(
                ordinal=ordinal,
                native_uid=str(ordinal),
                optimization_reward=row.get("score", 0.0),
                verifier_score=score,
                raw_verifier_score=raw,
                score_min=low,
                score_max=high,
                verification_status=verification.get("status"),
                passed=score == 1,
                outcome=Outcome.GRADED if score is not None else Outcome.UNAVAILABLE,
                error_type=row.get("exception_type"),
                error_treatment=row.get("error_treatment"),
                response_token_count=len(row["response_token_ids"]),
            )
        )
    return [
        EvaluationTaskRow(
            attempt_id=attempt_id,
            step=step,
            evaluation=evaluation,
            task_id=item,
            data_source="fixture",
            panel_name=selection.definition.name if selection else None,
            panel_version=selection.definition.version if selection else None,
            strata=selection.identity(item).strata if selection else TaskStrata(),
            task_digest=selection.tasks[item].task_digest if selection else None,
            score=sum(answer.verifier_score or 0.0 for answer in answers) / len(answers),
            outcome=Outcome.GRADED,
            error_type=None,
            answers=answers,
        )
        for item, answers in by_task.items()
    ]


def recorded_runs(selection, tmp_path, monkeypatch, request):
    monkeypatch.setattr(snowball_model, "SNOWBALL", with_current_control_revisions(snowball_model.SNOWBALL))
    set_job_info(None)
    request.addfinalizer(lambda: set_job_info(None))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    task = answer_task(
        RawRow("fixture", Source(dataset="fixture", revision="1", row="fixture", importer_revision="1"), {}),
        prompt="Return cat.",
        spec=ExactSpec(("cat",)),
    )

    panel_size = getattr(request, "param", None)
    for binding in {preset.held_out_tasks for preset in snowball_model.SNOWBALL.presets if preset.held_out_tasks}:
        task_set = Path(recipe.selection_artifact(binding).path())
        tasks = [task.model_copy(update={"id": f"task-{i}"}) for i in range(panel_size or 100)]
        rows = [
            SelectionTaskRow(
                task_id=value.id,
                task_digest=canonical_sha256(value.model_dump(mode="json")),
                family="small" if i == 0 else "large",
                aai_category="format" if i == 0 else "coding",
                source="fixture",
                agentic=False,
                state=MembershipState.ACTIVE,
            )
            for i, value in enumerate(tasks)
        ]
        validation = pa.table({"task_spec": [value.model_dump_json() for value in tasks]})
        pq.write_table(validation, task_set / VALIDATION_FILENAME)
        if panel_size is not None:
            pq.write_table(validation, task_set / FULL_VALIDATION_FILENAME)
        pq.write_table(
            pa.Table.from_pylist([row.model_dump(mode="json") for row in rows], schema=arrow_schema(SelectionTaskRow)),
            task_set / TASK_MANIFEST_FILENAME,
        )
        summary = json.loads((task_set / "summary.json").read_text())
        if panel_size is not None:
            summary.update(validation_rows=validation.num_rows, full_validation_rows=validation.num_rows)
        summary["selection_strata"] = SelectionDefinition(
            name=recipe.selection_artifact(binding).name,
            version=recipe.selection_artifact(binding).version,
            sampling="fixture",
            task_files=("validation.parquet",),
        ).model_dump(mode="json")
        write_json(task_set / "summary.json", summary)

    @contextmanager
    def endpoint(**kwargs):
        yield SimpleNamespace(url="http://iris.test", credentials=None)

    @contextmanager
    def rpc(*args):
        yield SimpleNamespace(list_jobs=lambda request: controller_pb2.Controller.ListJobsResponse())

    monkeypatch.setattr(adapter, "connect_controller", endpoint)
    monkeypatch.setattr(adapter, "rpc_client", rpc)

    templates = {}

    def composed(preset, inputs):
        key = (repr(preset), inputs.model_dump_json())
        if key not in templates:
            source_prefix = tmp_path / "composed" / str(len(templates))
            spec = recipe.run_spec(
                preset,
                name="fixture",
                tasks=recipe.selection_artifact(inputs.tasks) if inputs.tasks else None,
                init=recipe.model_from_reference(inputs.init) if inputs.init else None,
                settings=inputs.settings,
            )
            step = recipe.run_step(preset, spec)
            source_root = Path(step.path(str(source_prefix)))
            source_root.mkdir(parents=True)
            document = source_root / "launch.yaml"
            context = StepContext.for_run(
                str(source_root),
                recipe.artifact_prefix(preset.recipe.layout.cluster),
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
        missing_steps=(),
        attempt_id="recorded-attempt",
    ):
        preset = snowball_model.SNOWBALL.by_name[preset_name]
        root = tmp_path / name / preset.revision
        root.mkdir(parents=True, exist_ok=True)
        inputs = inputs or RunInputs(preset_name=preset.name)
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
        max_steps = launch["skyrl"]["trainer"]["max_steps"]
        generating = isinstance(preset.recipe.layout.workload, presets.Generation)
        pin = gate.control_pin(preset, table=snowball_model.SNOWBALL) if preset.gated else None
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
        write_receipt(root, native)
        write_json(root / RUN_INPUTS_FILENAME, inputs.model_dump(mode="json"))
        write_skyrl_run_record(
            root, name, preset.revision, fingerprint=fingerprint, global_step=0 if generating else max_steps
        )
        (root / ".executor_status").write_text("SUCCESS")
        stored_metrics = {}
        for step in (0,) if generating else range(1, max_steps + 1):
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
        default, selected = recipe.task_selections(
            preset, recipe.selection_artifact(inputs.tasks) if inputs.tasks else None
        )
        for index, evaluation in enumerate(preset.evaluations):
            source = recipe.evaluation_source(evaluation, selected, default, preset.held_out_tasks)
            sources = (
                launch["inputs"]["validation_data"]
                if index == 0
                else launch["inputs"]["evaluation_data"][evaluation.name]
            )
            assert len(sources) == 1
            descriptor = sources[0]
            directory = Path(descriptor["uri"])
            directory.mkdir(parents=True, exist_ok=True)
            count = evaluation.sample or recipe.selection_summary(source)[f"{evaluation.file}_rows"]
            ids = [f"train-task-{i}" if evaluation.file == "train" else f"task-{i}" for i in range(count)]
            path = directory / descriptor["relative_path"]
            if not path.exists():
                pq.write_table(
                    pa.table({"task_spec": [task.model_copy(update={"id": item}).model_dump_json() for item in ids]}),
                    path,
                )
            sessions = [(-1 if generating else max_steps, score)]
            if preset.checkpoint_every and evaluation.every:
                sessions = [(step, score) for step in range(evaluation.every, max_steps, evaluation.every)] + sessions
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
                evaluation_name = f"eval/{evaluation.name}/" if index else "eval/"
                task_rows = evaluation_task_rows(
                    rows,
                    attempt_id=attempt,
                    step=step,
                    evaluation=evaluation_name,
                    selection=read_selection_manifest(str(directory)),
                )
                write_evaluations(str(root / "exports"), task_rows)
                values = stored_metrics.setdefault(max(step, 0), {"step": max(step, 0)})
                prefix = f"eval/{evaluation.name}/all/outcome/" if index else "eval/all/outcome/"
                values.update(
                    {
                        prefix + value: len(rows) if value == Outcome.GRADED else 0
                        for value in (*Outcome, "rollout_failure")
                    }
                )
        for values in stored_metrics.values():
            values.update(metrics or {})
            values["attempt_id"] = attempt
        write_metrics(root, {step: values for step, values in stored_metrics.items() if step not in missing_steps})
        return root

    return write
