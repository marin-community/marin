# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw TaskSpec ingestion preserves logical trials and declared reductions."""

import asyncio
import json
import math
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from evaldash.app import Ingestor, MemoryRecordStore
from evaldash.curated import import_curated_record
from evaldash.metrics import build_comparison, build_model_detail, build_panel, panel_request
from evaldash.record_reconciliation import VerificationSchedule, _inspect_record
from evaldash.results_db import ObservationKind, SourceState
from evaldash.samples import _archive_task_table
from finestore.eval import (
    EvalSample,
    EvaluationStore,
    Grading,
    SampleKind,
    sample_from_archive_row,
    sample_to_archive_row,
)
from finestore.reader import ReadView
from finestore.schema import arrow_schema
from finestore.store import DataStore
from fray.local_backend import LocalClient
from marin.evaluation.dataset_evaluator import DatasetEvaluator
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import (
    BenchmarkMetadataRef,
    BenchmarkMetricRef,
    CuratedRef,
    CuratedSettings,
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HardwareRef,
    ModelRef,
    Provenance,
    TaskCoverage,
    write_record,
)
from marin.evaluation.runner import EvaluationError, EvaluationOutcome
from marin.execution.artifact import write_artifact
from marin.rollouts.evaluation import ATTEMPTS_TABLE, RUN_OBJECT, Attempt
from marin.rollouts.evaluation_results import logical_samples
from taskcompendium.evaluation.corpus import EvaluationManifest, evaluation_protocol
from taskcompendium.evaluation.selection import SELECTIONS_TABLE, Selection
from taskcompendium.evaluation.spec import EvaluationProtocol, LikelihoodMetric, Metric, Sampling, TrialPolicy
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks.math import math_dataset
from experiments.evaluation.policy_finalization import PolicyJudge
from experiments.evaluation.preparation import prepare_evaluation
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.images.build import (
    EnvironmentArtifact,
    environment_artifact,
    identity_digest,
)
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.pipeline import HfSource


@pytest.fixture
def raw_run(tmp_path):
    root = str(tmp_path / "run" / "results")
    record = EvalRunRecord(
        run_id="run",
        group_id="group",
        created_at="2026-11-05T00:00:00Z",
        user="test",
        model=ModelRef(name="solver", location="test/solver", backend="vllm"),
        eval=EvalRef(
            name="colors",
            mechanism="curated",
            curated=CuratedRef(datasets=("source",), settings=CuratedSettings()),
        ),
        hardware=HardwareRef(platform="gpu", accelerator="H100x1", region_or_cluster="test"),
        provenance=Provenance(git_sha="revision", launch_host="test"),
        status="succeeded",
        results_path=root,
        error=None,
        metrics={},
        jobs={},
        log_tails={},
    )
    return record


def write_observations(
    record,
    *,
    summary="mean",
    metrics=("accuracy",),
    metric_kind="continuous",
    num_fewshot=None,
    protocol: EvaluationProtocol | None = None,
    manifest: EvaluationManifest | None = None,
    observations,
):
    protocol = protocol or EvaluationProtocol(
        benchmark="colors",
        version="1",
        operation="continuation_likelihood" if summary == "perplexity" else "chat_generation",
        sampling=(
            None
            if summary == "perplexity"
            else Sampling(temperature=0, top_p=None, top_k=None, max_tokens=16, stop=(), chat_template_kwargs={})
        ),
        likelihood_metrics=(
            tuple(LikelihoodMetric(name=name, normalization="raw", scoring="log_likelihood") for name in metrics)
            if summary == "perplexity"
            else ()
        ),
        trials=TrialPolicy(count=1, seed=0),
        metrics=tuple(Metric(name=name, kind=metric_kind, higher_is_better=True) for name in metrics),
        primary_metric=metrics[0],
        num_fewshot=num_fewshot,
        summary=summary,
        expected_population=2,
    )
    snapshot = (
        manifest.model_dump(mode="json", exclude_none=True)
        if manifest is not None
        else {"rows": 2, "source_dataset": "colors", "source_revision": "pin"}
    )
    dataset_id = canonical_sha256(snapshot) if manifest is not None else "dataset"
    with DataStore.open(record.results_path) as store:
        EvaluationStore(store)
        store.table(
            ATTEMPTS_TABLE,
            schema=arrow_schema(Attempt),
            primary_key=("dataset", "source_row", "trial", "retry", "phase"),
            schema_version=1,
        )
        store.table(
            SELECTIONS_TABLE, schema=arrow_schema(Selection), primary_key=("dataset", "source_row"), schema_version=1
        )
        with store.transaction() as transaction:
            transaction.write_object(
                RUN_OBJECT,
                json.dumps(
                    {
                        "datasets": [
                            {
                                "dataset": dataset_id,
                                "protocol": protocol.model_dump(mode="json"),
                                "manifest": snapshot,
                            }
                        ]
                    }
                ).encode(),
            )
            for index in range(2):
                transaction.table(SELECTIONS_TABLE).add(
                    Selection(
                        dataset=dataset_id,
                        source_row=f"rows:{index}",
                        source_ordinal=index,
                        status="selected",
                        reason=None,
                        rank=index,
                        context_window=1024,
                        max_tokens=16,
                        prompt_tokens=1,
                        reference_tokens=None,
                    ).model_dump(mode="json")
                )
            for index, retry, phase, outcome, values, weights in observations:
                attempt = Attempt(
                    dataset=dataset_id,
                    task_id=str(index),
                    source_row=f"rows:{index}",
                    trial=0,
                    retry=retry,
                    phase=phase,
                    seed=0,
                    outcome=outcome,
                    error=None,
                    result_blob="evidence",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
                transaction.table(ATTEMPTS_TABLE).add(attempt.model_dump(mode="json"))
                if phase == "finished":
                    sample = EvalSample(
                        task="colors",
                        doc_id=str(index),
                        kind=SampleKind.GENERATION,
                        output="red",
                        metrics=values,
                        grading=Grading(
                            method="fixture",
                            metric=metrics[0],
                            score=values.get(metrics[0]),
                            detail=json.dumps({"detail": {"metric_weights": weights}}),
                        ),
                        doc=json.dumps({"groups": {"color": "red"}}),
                    )
                    transaction.table("samples").add(sample_to_archive_row(sample, trial_id=f"{dataset_id}/0/{retry}"))


def test_regional_curation_builds_compare_and_grader_changes_remain_separate(raw_run, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "test.jsonl").write_text(
        "\n".join(json.dumps({"problem": f"{index} + 1?", "answer": str(index + 1)}) for index in range(2)) + "\n"
    )
    source = math_dataset(
        "colors", "colors", HfSource("fixture/colors", "a" * 40, ("test.jsonl",), SourceFormat.JSONL), "answer", 2, 1, 0
    )
    dataset = source.config
    assert dataset is not None and dataset.recipe.grader is not None
    revised = replace(dataset.recipe.grader, requirements=(*dataset.recipe.grader.requirements, "numpy==2.3.5"))
    records = []
    runtime = CampaignRuntime()
    options = PipelineOptions(SourceProcessingMode.QUICK, runtime, inputs=InputOverrides(root=str(inputs)))
    client = LocalClient()
    try:
        with (
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            for region, grader in (
                ("east", dataset.recipe.grader),
                ("west", dataset.recipe.grader),
                ("revised", revised),
            ):
                declaration = replace(source, config=replace(dataset, recipe=replace(dataset.recipe, grader=grader)))
                output = str(tmp_path / region / "corpus")
                with artifact_prefix(str(tmp_path / region / "artifacts")):
                    built = EnvironmentArtifact(
                        path=environment_artifact(grader).path(), identity=identity_digest(grader)
                    )
                    write_artifact(built.result_payload(), built.path)
                    publish = prepare_evaluation(declaration, options).lower().fn
                    assert publish is not None
                    publish(output)
                protocol, manifest = evaluation_protocol(output)
                record = raw_run.model_copy(
                    update={
                        "run_id": region,
                        "model": raw_run.model.model_copy(update={"name": region}),
                        "results_path": str(tmp_path / region / "results"),
                    }
                )
                write_observations(
                    record,
                    protocol=protocol,
                    manifest=manifest,
                    observations=[
                        (0, 0, "finished", "graded", {"accuracy": 1.0}, {}),
                        (1, 0, "finished", "graded", {"accuracy": 0.0}, {}),
                    ],
                )
                records.append(import_curated_record(record))
    finally:
        client.shutdown()
    comparison = build_comparison(records[:2], panel_request(), ("east", "west"))
    assert comparison["shared"] == ["colors [evalchemy; chat]"]
    assert build_comparison([records[0], records[2]], panel_request(), ("east", "revised"))["shared"] == []


def test_raw_ingestion_and_browser_choose_terminal_trials_without_counting_retries(raw_run, tmp_path):
    write_observations(
        raw_run,
        observations=[
            (0, 0, "finished", "infra_error", {}, {}),
            (0, 1, "finished", "graded", {"accuracy": 0.0}, {}),
            (0, 2, "started", None, {}, {}),
            (1, 0, "finished", "graded", {"accuracy": 1.0}, {}),
        ],
    )
    write_record(raw_run, str(tmp_path))
    store = MemoryRecordStore()
    assert asyncio.run(Ingestor(store, (str(tmp_path),), 1).run_once()) == ()
    imported = store.get_record("run")
    assert imported is not None
    assert imported["metrics"]["colors"]["accuracy"] == 0.5
    assert imported["coverage"]["colors"]["n_scored"] == 2
    assert imported["coverage"]["colors"]["n_attempted"] == 2
    samples = list(logical_samples(ReadView(raw_run.results_path)))
    assert sorted(dict(sample["metrics"])["accuracy"] for sample in samples) == [0, 1]
    table = _archive_task_table(raw_run.results_path, "colors")
    assert sorted(dict(row["metrics"])["accuracy"] for row in table.to_pylist()) == [0, 1]
    assert import_curated_record(raw_run).canonical_metrics["colors"]["accuracy"] == 0.5


@pytest.mark.parametrize(
    "name,historical_metric,canonical_metric,native_metric,kind,shots,normalized_metric",
    [
        ("mmlu", "acc,none", "accuracy", "acc,none", "binary", 5, "accuracy"),
        ("piqa", "acc_norm,none", "normalized_accuracy", "acc_norm,none", "binary", 0, "normalized_accuracy"),
        ("gsm8k", "exact_match,flexible-extract", "accuracy", "exact_match,strict-match", "binary", 5, "accuracy"),
        ("humanevalplus", "pass@1", "pass_at_1", "accuracy", "binary", 0, "accuracy"),
        ("humaneval-10shot", "pass@1", "pass_at_1", "accuracy", "binary", 10, "accuracy"),
        ("mrcr", "mrcr_accuracy", "mrcr_accuracy", "accuracy", "continuous", 0, "accuracy"),
        ("math500", "accuracy_avg", "accuracy", "accuracy", "binary", 0, "accuracy"),
        (
            "ifbench",
            "strict_prompt_accuracy",
            "strict_prompt_accuracy",
            "strict_prompt_accuracy",
            "binary",
            0,
            "strict_prompt_accuracy",
        ),
    ],
)
def test_native_summary_reuses_historical_scores_in_panel_comparison_and_history(
    raw_run, name, historical_metric, canonical_metric, native_metric, kind, shots, normalized_metric
):
    raw_run = raw_run.model_copy(update={"evaluation": raw_run.evaluation.model_copy(update={"name": name})})
    write_observations(
        raw_run,
        metrics=(native_metric,),
        metric_kind=kind,
        num_fewshot=shots,
        observations=[
            (0, 0, "finished", "graded", {native_metric: 1.0}, {}),
            (1, 0, "finished", "graded", {native_metric: 0.0}, {}),
        ],
    )
    native = import_curated_record(raw_run)
    historical = raw_run.model_copy(
        update={
            "run_id": "historical",
            "created_at": "2026-10-01T00:00:00Z",
            "model": raw_run.model.model_copy(update={"name": "historical-model"}),
            "evaluation": EvalRef(
                name=name,
                mechanism="evalchemy",
                tasks=(
                    EvalTaskRef(
                        name=name,
                        num_fewshot=shots,
                        benchmark=BenchmarkMetadataRef(
                            schema_version=1,
                            task=name,
                            primary_metric=canonical_metric,
                            metric_kind=kind,
                            metrics=(
                                BenchmarkMetricRef(
                                    name=canonical_metric,
                                    source_name=historical_metric.split(",")[0],
                                    kind=kind,
                                    higher_is_better=True,
                                ),
                            ),
                            n_benchmark=10,
                            n_attempted=10,
                        ),
                    ),
                ),
            ),
            "metrics": {name: {historical_metric: 0.6, "sample_len": 10.0}},
            "canonical_metrics": {name: {canonical_metric: 0.6}},
            "coverage": {name: TaskCoverage(n_benchmark=10, n_attempted=10, n_scored=10)},
        }
    )
    records = [historical, native]
    panel = build_panel(records, panel_request())
    cells = {row["model"]: row["cells"][name] for row in panel["rows"]}
    assert cells["historical-model"]["metric"] == cells["solver"]["metric"] == normalized_metric
    assert cells["historical-model"]["value"] == 0.6
    assert cells["solver"]["value"] == 0.5
    assert cells["historical-model"]["run_id"] == "historical"
    assert cells["historical-model"]["n_scored"] == 10
    comparison = build_comparison(records, panel_request(), ("historical-model", "solver"))
    assert comparison["shared"] == [name]
    same_model = native.model_copy(update={"model": historical.model})
    detail = build_model_detail([historical, same_model], "historical-model")
    assert detail is not None
    assert [point["run_id"] for point in detail["history"][name]] == ["historical", "run"]


@pytest.mark.parametrize(
    "summary,metric,observations,expected",
    [
        (
            "weighted_mean",
            "instruction",
            [
                (0, 0, "finished", "graded", {"instruction": 1.0}, {"instruction": 1}),
                (1, 0, "finished", "graded", {"instruction": 0.0}, {"instruction": 3}),
            ],
            0.25,
        ),
        (
            "perplexity",
            "ppl",
            [
                (0, 0, "finished", "graded", {"ppl": -2.0}, {}),
                (1, 0, "finished", "graded", {"ppl": -4.0}, {}),
            ],
            math.exp(3),
        ),
    ],
)
def test_raw_import_uses_the_protocol_reduction(raw_run, summary, metric, observations, expected):
    write_observations(raw_run, summary=summary, metrics=(metric,), observations=observations)
    result = import_curated_record(raw_run)
    assert result.metrics["colors"][metric] == pytest.approx(expected)
    assert result.metrics["colors"][f"color=red/{metric}"] == pytest.approx(expected)


def test_failed_and_unfinished_trials_remain_missing_coverage(raw_run):
    write_observations(
        raw_run, observations=[(0, 0, "finished", "infra_error", {}, {}), (1, 0, "started", None, {}, {})]
    )
    result = import_curated_record(raw_run)
    assert result.metrics["colors"] == {}
    assert result.coverage["colors"].n_scored == 0
    assert result.coverage["colors"].errors == {"infra_error": 1, "unfinished": 1}


def test_likelihood_evidence_does_not_require_generation_grader_detail(raw_run):
    write_observations(raw_run, observations=[(0, 0, "finished", "graded", {"accuracy": 1.0}, {})])
    row = next(iter(logical_samples(ReadView(raw_run.results_path))))
    sample = sample_from_archive_row(row)
    assert sample.grading is not None
    sample = sample.model_copy(
        update={
            "grading": sample.grading.model_copy(
                update={"detail": json.dumps([{"loglikelihood": -2.0, "is_greedy": True}])}
            )
        }
    )
    with DataStore.open(raw_run.results_path) as store:
        EvaluationStore(store)
        with store.transaction() as transaction:
            transaction.table("samples").add(sample_to_archive_row(sample, trial_id=row["trial_id"]))
    result = import_curated_record(raw_run)
    assert result.metrics["colors"]["accuracy"] == 1
    assert result.coverage["colors"].n_scored == 1


def test_reconciliation_refreshes_resumed_raw_results_without_rewriting_launch_record(raw_run, tmp_path):
    write_observations(raw_run, observations=[(0, 0, "finished", "graded", {"accuracy": 0.0}, {})])
    write_record(raw_run, str(tmp_path))
    path = str(tmp_path / "run" / "record.json")
    schedule = VerificationSchedule(checked_at=datetime.now(UTC), retry_after=1, revalidate_after=60)
    first = _inspect_record(path, None, schedule)
    assert first.record is not None
    assert first.record.coverage["colors"].n_scored == 1
    state = SourceState(
        path=path,
        run_id="run",
        object_version=first.object_version,
        last_verified_at=schedule.checked_at,
        next_verify_at=schedule.checked_at,
        missing_since=None,
        error=None,
    )
    assert _inspect_record(path, state, schedule).kind is ObservationKind.UNCHANGED
    write_observations(raw_run, observations=[(1, 0, "finished", "graded", {"accuracy": 1.0}, {})])
    resumed = _inspect_record(path, state, schedule)
    assert resumed.kind is ObservationKind.CHANGED
    assert resumed.record is not None
    assert resumed.record.metrics["colors"]["accuracy"] == 0.5
    assert resumed.record.coverage["colors"].n_scored == 2


@pytest.mark.parametrize("stored_version", ["1730000000000000", "local-object-hash"])
def test_reconciliation_recovers_after_invalid_record_is_repaired(raw_run, tmp_path, stored_version):
    write_record(raw_run, str(tmp_path))
    path = str(tmp_path / "run" / "record.json")
    schedule = VerificationSchedule(checked_at=datetime.now(UTC), retry_after=1, revalidate_after=60)
    state = SourceState(
        path=path,
        run_id="run",
        object_version=stored_version,
        last_verified_at=schedule.checked_at,
        next_verify_at=schedule.checked_at,
        missing_since=None,
        error="invalid record",
    )
    recovered = _inspect_record(path, state, schedule)
    assert recovered.kind is ObservationKind.CHANGED
    assert recovered.error is None
    assert recovered.record is not None
    assert recovered.record.run_id == "run"


@pytest.mark.parametrize("second_outcome", ["graded", "infra_error"])
def test_policy_finalization_uses_the_same_scored_trial_coverage_as_evaldash(raw_run, monkeypatch, second_outcome):
    write_observations(
        raw_run,
        observations=[
            (0, 0, "finished", "graded", {"accuracy": 0.0}, {}),
            (1, 0, "finished", second_outcome, {"accuracy": 0.0} if second_outcome == "graded" else {}, {}),
        ],
    )
    model = ModelConfig(name="judge", location="test/judge", revision="pin", tokenizer_revision="pin")
    source = DatasetEvaluator(("dataset",), model, CuratedSettings())
    executor = PolicyJudge(source, model, AcceleratorChoice(platform=Platform.GPU, gpu_type="H100"), 0)
    session = SimpleNamespace(jobs=(), wait_until_ready=lambda **_kwargs: None)
    monkeypatch.setattr("experiments.evaluation.policy_finalization.inference_config_for_model", lambda *_a, **_k: None)
    monkeypatch.setattr("experiments.evaluation.policy_finalization.precache_inference_model", lambda _config: None)
    monkeypatch.setattr(
        "experiments.evaluation.policy_finalization.remote_inference", lambda _config: nullcontext(session)
    )
    monkeypatch.setattr(DatasetEvaluator, "__call__", lambda *_a, **_k: EvaluationOutcome(metrics={}))
    expected = import_curated_record(raw_run).coverage
    if second_outcome == "infra_error":
        with pytest.raises(EvaluationError) as caught:
            executor(None, raw_run.results_path, {}, backend_state=lambda: None)
        assert caught.value.status == "artifact_failed"
        assert caught.value.coverage == expected
        return
    result = executor(None, raw_run.results_path, {}, backend_state=lambda: None)
    assert result.coverage == expected
    assert result.coverage["colors"].n_scored == 2
