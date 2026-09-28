# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
from pathlib import Path

import haliax as hax
import jax
import numpy as np
import pytest
from levanter.data.text.cache import load_lm_dataset_cache
from levanter.data.text.datasets import TokenSeqDataset
from levanter.store.cache import CacheMetadata
from levanter.tokenizers import load_tokenizer
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import materialized_config
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.training.training import LevanterCheckpoint

from experiments.domain_phase_mix import analyze_starcoder_tpp10 as analysis
from experiments.domain_phase_mix import evaluate_starcoder_tpp10_uncheatable as uncheatable
from experiments.domain_phase_mix import launch_starcoder_tpp10 as launcher
from experiments.domain_phase_mix import plot_starcoder_tpp10_refinement as refinement_analysis
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as preparation
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.starcoder_epoch_matching import file_sha256


def test_uncheatable_resume_checks_components_checkpoint_and_restored_control(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "PREFIX", str(tmp_path))
    request = {"run_name": "checkpoint", "fingerprint": "frozen"}
    spec = {
        "spec_sha256": "evaluation",
        "checkpoints": {"checkpoint": {"step": 2531}},
        "primary_bpb": {"checkpoint": 1.2},
    }
    assert uncheatable.verified_result(spec, request) is None
    metrics = {f"eval/uncheatable_eval/{name}/bpb": float(i) for i, name in enumerate(uncheatable.COMPONENTS, 1)}
    # The micro average must not replace the paper's equal-component average.
    metrics.update({"eval/uncheatable_eval/bpb": 5.0, experiment.PRIMARY_METRIC: 1.2})
    result = {
        "spec_sha256": "evaluation",
        "run_name": "checkpoint",
        "fingerprint": "frozen",
        "checkpoint": {"step": 2531},
        "metrics": metrics,
        "macro_bpb": 4.0,
    }
    path = uncheatable.result_path(spec, request)
    uncheatable.persist_submission_plan(result, path)
    assert uncheatable.verified_result(spec, request)["macro_bpb"] == 4.0
    result["fingerprint"] = "different-checkpoint"
    Path(path).write_text(json.dumps(result))
    with pytest.raises(ValueError, match="identity mismatch"):
        uncheatable.verified_result(spec, request)
    result["fingerprint"] = "frozen"
    result["metrics"][experiment.PRIMARY_METRIC] = 1.3
    Path(path).write_text(json.dumps(result))
    with pytest.raises(ValueError, match="PALOMA control"):
        uncheatable.verified_result(spec, request)


def test_uncheatable_rejects_partial_components():
    metrics = {f"eval/uncheatable_eval/{name}/bpb": 1.0 for name in uncheatable.COMPONENTS[:-1]}
    metrics["eval/uncheatable_eval/bpb"] = 1.0
    with pytest.raises(KeyError, match="ao3_english"):
        uncheatable.component_scores(metrics)


@pytest.fixture(scope="module")
def design() -> dict:
    return experiment.load_design()


def test_model_horizons_match_total_tpp_and_finite_pool_allocations(design: dict) -> None:
    # Independent model-tree counts are part of build_design. These checks protect the scientific controls.
    assert max(abs(model["tpp"] / 10 - 1) for model in design["models"].values()) < 0.001
    result = asyncio.run(experiment.audit_allocations(design))
    assert result["maximum_epoch_relative_error"] < 0.001
    endpoint = result["coordinates"][-1]
    assert endpoint["proxy_allocation"]["starcoder"] < experiment.PARENT_SEQUENCES
    assert endpoint["target_epochs"] > 15


def test_subset_draws_are_nested_distinct_and_permutation_preserves_all_parent_sequences() -> None:
    parent = experiment.parent_permutation()
    assert np.array_equal(np.sort(parent), np.arange(experiment.PARENT_SEQUENCES))
    subsets = [parent[experiment.subset_indices(seed)] for seed in experiment.SUBSET_SEEDS]
    assert all(len(np.unique(ids)) == experiment.MATCHED_SEQUENCES for ids in subsets)
    for i in range(len(subsets)):
        for j in range(i):
            overlap = len(np.intersect1d(subsets[i], subsets[j]))
            assert 0 < overlap < experiment.MATCHED_SEQUENCES // 4


def test_stage_expansion_reuses_training_identities_and_calibration_keeps_token_budget(design: dict) -> None:
    calibration, _ = launcher.build_plan(design, "calibration")
    pilot, _ = launcher.build_plan(design, "pilot")
    dense, _ = launcher.build_plan(design, "dense")
    full = {r["run_name"]: r["fingerprint"] for r in dense["runs"]}
    assert all(full[r["run_name"]] == r["fingerprint"] for r in pilot["runs"])
    assert len({r["tokens"] for r in calibration["runs"]}) == 1
    assert {r["batch_size"] for r in calibration["runs"]} == {32, 128}


def test_zero_weight_endpoint_preserves_named_web_and_starcoder_order(design: dict, tmp_path: Path) -> None:
    caches = preparation.data_steps(design)
    tok = load_tokenizer(experiment.TOKENIZER)
    metadata = CacheMetadata(preparation.FORMAT.build_preprocessor(tok).metadata)
    for i, step in enumerate(caches.values()):
        path = step.path(str(tmp_path))
        write_record(
            ArtifactRecord(
                output_path=path,
                fingerprint=step.fingerprint(),
                config={"tokenizer": experiment.TOKENIZER, "format": {"text_key": "text"}},
            )
        )
        if step == caches["evaluation"]:
            continue
        ids = np.repeat(np.arange(1000 * (i + 1), 1000 * (i + 1) + 512, dtype=np.int32), experiment.SEQ_LEN)
        preparation.write_part(
            [{"input_ids": ids}],
            path + "/train",
            metadata=metadata,
            identity={"synthetic_component": step.name},
            expected_tokens=len(ids),
        )
    requests = [
        r
        for r in experiment.select_runs(design, "dense")
        if r.trainer_seed == experiment.TRAINER_SEEDS[0]
        and r.percent in (0, 5, 100)
        and (r.arm == experiment.Arm.UNMATCHED or (r.arm == experiment.Arm.MATCHED and r.percent == 5))
    ]
    digests = {}
    baseline = None
    starcoder = None
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(experiment.DATA_SEED))
    for row in requests:
        step = launcher.training_step(design, row, caches)
        config = materialized_config(step, str(tmp_path)).pod.train_config
        assert config.data.max_train_batches is None
        assert config.data.experiment_budget is None
        streams = config.data.train_sets(
            hax.Axis("position", experiment.SEQ_LEN), key=shuffle_key, initial_batch_size=row.batch_size
        )
        observed = {}
        for name, stream in streams.items():
            items = asyncio.run(stream.get_batch(range(64)))
            identities = [int(np.asarray(item.tokens)[0]) for item in items]
            observed[name] = launcher.canonical_sha256({"sequence_ids": identities})
        digests[row.run_name] = observed
        if row.percent == 0:
            assert "starcoder" not in streams
            baseline = observed
        elif row.percent == 5:
            assert "starcoder" in streams
            assert {name: observed[name] for name in experiment.WEB_COUNTS} == baseline
            if row.arm == experiment.Arm.UNMATCHED:
                starcoder = observed["starcoder"]
        elif row.percent == 100:
            assert set(streams) == {"starcoder"}
            assert observed["starcoder"] == starcoder
    (tmp_path / "zero_weight_stream_audit.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "design_sha256": design["design_sha256"],
                "scope": (
                    "Real native train_sets on local synthetic finite caches, first 64 source sequence identities. "
                    "Per-component streams agree across p=0/.05 and all matched subset handles; "
                    "unmatched StarCoder agrees at p=.05/1. Mixture interleaving necessarily changes with p. "
                    "This is an offline implementation check, not a fingerprint of prepared GCS tokens."
                ),
                "stream_sha256": digests,
            },
            indent=2,
        )
        + "\n"
    )


def test_token_budget_stops_input_and_finished_cache_reuses_without_reading(tmp_path: Path) -> None:
    tok = load_tokenizer(experiment.TOKENIZER)
    metadata = CacheMetadata(preparation.FORMAT.build_preprocessor(tok).metadata)

    def docs():
        yield {"text": "def function():\n    return 42\n" * 20}
        raise AssertionError("Token budget should stop reading source documents")

    path = str(tmp_path / "cache")
    receipt = preparation.write_part(
        preparation.token_records(docs(), tokenizer_path=experiment.TOKENIZER, quota=17),
        path,
        metadata=metadata,
        identity={"test": "bounded-cache"},
        expected_tokens=17,
    )
    assert receipt["tokens"] == 17
    assert preparation.cache_token_count(path, metadata) == 17

    def no_read():
        raise AssertionError("A completed cache must not read input again")
        yield

    assert (
        preparation.write_part(
            no_read(), path, metadata=metadata, identity={"test": "bounded-cache"}, expected_tokens=17
        )
        == receipt
    )
    with pytest.raises(ValueError):
        preparation.write_part(no_read(), path, metadata=metadata, identity={"test": "changed"}, expected_tokens=17)


def test_materialized_subset_contains_exact_requested_parent_sequences(tmp_path: Path) -> None:
    tok = load_tokenizer(experiment.TOKENIZER)
    metadata = CacheMetadata(preparation.FORMAT.build_preprocessor(tok).metadata)
    parent = tmp_path / "parent"
    ids = np.repeat(np.arange(10, 18, dtype=np.int32), experiment.SEQ_LEN)
    preparation.write_part(
        [{"input_ids": ids}],
        str(parent / "train"),
        metadata=metadata,
        identity={"test": "parent"},
        expected_tokens=len(ids),
    )
    recipe = preparation.ParentRecipe(
        str(parent),
        str(tmp_path / "subset"),
        experiment.TOKENIZER,
        "synthetic-design",
        preparation.file_sha256(Path(preparation.__file__)),
    )
    preparation.materialize_indices(recipe, np.array([6, 1, 4]), "synthetic-index-digest", parent_sequences=8)
    cache = load_lm_dataset_cache(str(tmp_path / "subset/train"), preparation.FORMAT, tok)
    seq = TokenSeqDataset(cache, experiment.SEQ_LEN)
    rows = asyncio.run(seq.get_batch([0, 1, 2]))
    assert [np.unique(row["input_ids"]).tolist() for row in rows] == [[16], [11], [14]]


def test_compressed_source_read_cannot_exceed_its_budget() -> None:
    source = preparation.BudgetedReader(io.BytesIO(b"abcdefgh"), 4)
    assert source.read(4) == b"abcd"
    with pytest.raises(ValueError):
        source.read(4)


def test_analysis_preserves_unfavorable_selection_and_does_not_pool_subset_information(design: dict) -> None:
    plan, _ = launcher.build_plan(design, "pilot")
    minima = {experiment.SUBSET_SEEDS[0]: 10, experiment.SUBSET_SEEDS[1]: 30, experiment.SUBSET_SEEDS[2]: 50}
    values = {}
    for r in plan["runs"]:
        p = r["percent"]
        center = 70 if r["arm"] == "target" else 100 if r["arm"] == "unmatched" else minima[r["subset_seed"]]
        values[r["run_name"]] = 1 + ((p - center) / 100) ** 2
    result = analysis.analyze(plan, values)
    assert result["target_selected_percent"] == 70
    assert result["unmatched_selected_percent"] == 100
    assert [r["selected_percent"] for r in result["matched_subsets"]] == [10, 30, 50]
    assert result["mean_paired_regret_difference_bpb"] == pytest.approx((0.36 + 0.16 + 0.04) / 3 - 0.09)


@pytest.fixture
def refinement_measurements(design: dict) -> tuple[dict, dict, dict[str, float], list[dict]]:
    pilot, _ = launcher.build_plan(design, "pilot")
    dense, _ = launcher.build_plan(design, "dense")
    refinement = {
        "plan_sha256": "synthetic-refinement",
        "runs": [r for r in dense["runs"] if r["percent"] in refinement_analysis.REFINEMENT_GRID],
    }
    minima = dict(zip(experiment.SUBSET_SEEDS, (40, 55, 65), strict=True))
    values = {}
    for row in pilot["runs"] + refinement["runs"]:
        percent = row["percent"]
        if row["arm"] == "target":
            # A tie at 55/60 should choose 55, including when the new point wins over the pilot grid.
            distance = max(55 - percent, 0, percent - 60)
        else:
            distance = percent - (100 if row["arm"] == "unmatched" else minima[row["subset_seed"]])
        values[row["run_name"]] = 1 + (distance / 100) ** 2
    endpoints = [
        {
            **row,
            "artifact_status": STATUS_SUCCESS,
            "value": values[row["run_name"]],
            "runtime_verified": True,
            "wandb_state": "finished",
        }
        for row in refinement["runs"]
    ]
    return pilot, refinement, {r["run_name"]: values[r["run_name"]] for r in pilot["runs"]}, endpoints


def test_complete_refinement_uses_new_target_minimum_and_retains_subset_choices(refinement_measurements) -> None:
    result = refinement_analysis.summarize(*refinement_measurements)
    complete = result["complete_common_grid_analysis"]
    assert result["pilot_common_grid_analysis"]["target_selected_percent"] == 50
    assert complete["target_selected_percent"] == 55
    assert complete["unmatched_target_regret_bpb"] == pytest.approx(0.16)
    assert [r["selected_percent"] for r in complete["matched_subsets"]] == [40, 55, 65]
    assert [r["target_regret_bpb"] for r in complete["matched_subsets"]] == pytest.approx([0.0225, 0, 0.0025])
    assert complete["verified_artifact_count"] == 102
    assert result["verified_but_unplotted"] == []


@pytest.mark.parametrize("arm", ["target", "matched"])
def test_partial_refinement_does_not_publish_common_grid_regret(refinement_measurements, arm: str) -> None:
    pilot, refinement, values, endpoints = refinement_measurements
    row = next(r for r in endpoints if r["arm"] == arm and r["percent"] == 80)
    row["artifact_status"] = "running"
    result = refinement_analysis.summarize(pilot, refinement, values, endpoints)
    assert "complete_common_grid_analysis" not in result
    assert result["refinement_complete"] == 44
    assert result["missing_run_names"] == [row["run_name"]]
    assert len(result["verified_but_unplotted"]) == (1 if arm == "matched" else 0)


def test_complete_refinement_rejects_success_without_verified_runtime(refinement_measurements) -> None:
    pilot, refinement, values, endpoints = refinement_measurements
    endpoints[-1]["runtime_verified"] = False
    with pytest.raises(ValueError, match="Unverified snapshot endpoint"):
        refinement_analysis.summarize(pilot, refinement, values, endpoints)


@pytest.fixture
def mixed_bpb_records(tmp_path):
    # The actual neighboring target measurements that produced the spurious 70% bump.
    measurements = [(65, 0.7665561437606812, 1.3150529861450195, 2), (70, 0.7843883633613586, 1.3133516311645508, None)]
    runs, values, records = [], {}, []
    metric = experiment.PRIMARY_METRIC
    for percent, bpb, loss, schema in measurements:
        request = {
            "run_name": f"target-{percent}",
            "percent": percent,
            "total_steps": 11491,
            "fingerprint": str(percent),
        }
        final = {metric: bpb, metric.removesuffix("bpb") + "loss": loss, "eval/bpb_schema_version": schema}
        source = tmp_path / f"{percent}.jsonl"
        source.write_text(json.dumps({"step": 11490, **final}) + "\n")
        runs.append(request)
        values[request["run_name"]] = bpb
        records.append(
            {**request, "final": final, "final_records": 1, "source": str(source), "source_sha256": file_sha256(source)}
        )
    manifest = tmp_path / "metrics.json"
    manifest.write_text(json.dumps(records))
    population = {"examples": 5673, "tokens": 11612631, "bytes": 28741166}
    return runs, values, manifest, population


def test_metric_normalization_removes_schema_crossing_and_preserves_originals(mixed_bpb_records):
    runs, raw_values, manifest, population = mixed_bpb_records
    values, provenance = refinement_analysis.normalize_final_metrics(runs, raw_values, manifest, population)
    assert min(raw_values, key=raw_values.get) == "target-65"
    assert min(values, key=values.get) == "target-70"
    assert values["target-65"] == pytest.approx(0.766556258779136, abs=1e-12)
    assert values["target-70"] == pytest.approx(0.7655645236000794, abs=1e-12)
    assert {r["run_name"]: r["reported_bpb"] for r in provenance} == raw_values
    assert [r["reported_schema"] for r in provenance] == [2, None]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_loss",
        "wrong_raw_value",
        "wrong_source_hash",
        "wrong_final_step",
        "unknown_schema",
        "schema2_disagreement",
        "wrong_identity",
    ],
)
def test_metric_normalization_rejects_unaudited_measurements(mixed_bpb_records, fault):
    runs, values, manifest, population = mixed_bpb_records
    records = json.loads(manifest.read_text())
    record = records[0]
    source = Path(record["source"])
    event = json.loads(source.read_text())
    if fault == "missing_loss":
        del event[experiment.PRIMARY_METRIC.removesuffix("bpb") + "loss"]
    elif fault == "wrong_raw_value":
        values[record["run_name"]] += 0.1
    elif fault == "wrong_source_hash":
        record["source_sha256"] = "0" * 64
    elif fault == "wrong_final_step":
        event["step"] -= 1
    elif fault == "unknown_schema":
        event["eval/bpb_schema_version"] = 3
    elif fault == "schema2_disagreement":
        event[experiment.PRIMARY_METRIC.removesuffix("bpb") + "loss"] += 0.1
    else:
        record["fingerprint"] = "different-model"
    source.write_text(json.dumps(event) + "\n")
    if fault != "wrong_source_hash":
        record["source_sha256"] = file_sha256(source)
    record["final"] = {k: v for k, v in event.items() if k != "step"}
    manifest.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        refinement_analysis.normalize_final_metrics(runs, values, manifest, population)


def test_default_launcher_does_not_read_cloud_and_rejects_submission_without_release(tmp_path: Path) -> None:
    out = tmp_path / "plan.json"
    launcher.main(["--plan-path", str(out)])
    assert json.loads(out.read_text())["stage"] == "calibration"
    with pytest.raises(ValueError):
        launcher.main(["--plan-path", str(out), "--submit"])


def test_calibration_release_uses_seed_means_and_ignores_matched_advantage(design: dict) -> None:
    plan, _ = launcher.build_plan(design, "calibration")
    values = {
        r["run_name"]: 0.8 if r["arm"] == "matched" else 1.03 if r["batch_size"] == 32 else 1.0 for r in plan["runs"]
    }
    assert not experiment.calibration_summary(plan, values)["batch32_loss_screen_passed"]
    for row in plan["runs"]:
        if row["arm"] == "unmatched" and row["batch_size"] == 32:
            values[row["run_name"]] = 0.99 if row["trainer_seed"] == experiment.TRAINER_SEEDS[0] else 1.02
    assert experiment.calibration_summary(plan, values)["batch32_loss_screen_passed"]


def test_collection_requires_success_runtime_receipt_and_final_endpoint(design: dict, tmp_path: Path) -> None:
    plan, steps = launcher.build_plan(design, "calibration")
    request, step = plan["runs"][0], steps[0]
    path = step.path(str(tmp_path))
    request = {**request, "output_path": path}
    # A historical receipt is readable even when today's checkout has different code/design pins.
    plan = {
        **plan,
        "runs": [request],
        "design_sha256": "historical-design",
        "code_sha256": {"historical_training.py": "historical-source-digest"},
    }
    plan["plan_sha256"] = launcher.canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"})
    write_record(ArtifactRecord(output_path=path, fingerprint=step.fingerprint()))
    runtime = {
        "design_sha256": plan["design_sha256"],
        "versions": plan["runtime_versions"],
        "code_sha256": plan["code_sha256"],
    }
    receipt_path = Path(path) / "verified_runtime.json"
    receipt_path.write_text(json.dumps(runtime))
    metric_path = Path(LevanterCheckpoint(path=path).checkpoint_dir) / "eval_metrics.jsonl"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps({"step": request["total_steps"] - 2, plan["primary_metric"]: 1.2}) + "\n")
    out = tmp_path / "measurements.csv"
    with pytest.raises(ValueError, match="Training incomplete"):
        launcher.collect_results(plan, out)
    StatusFile(path, worker_id="synthetic-collection-test").write_status(STATUS_SUCCESS)
    with pytest.raises(ValueError, match="endpoint"):
        launcher.collect_results(plan, out)
    metric_path.write_text("\n" + json.dumps({"step": request["total_steps"] - 1, plan["primary_metric"]: 1.1}) + "\n")
    values = launcher.collect_results(plan, out)
    assert analysis.verified_measurements(plan, out) == values == {request["run_name"]: 1.1}
    archive = tmp_path / "archived_plan.json"
    archive.write_text(json.dumps(plan))
    original = archive.read_bytes()
    launcher.main(["--plan-path", str(archive), "--collect-results", str(out)])
    assert archive.read_bytes() == original
    assert analysis.verified_measurements(plan, out) == values
    receipt_path.write_text(json.dumps({**runtime, "versions": {"jax": "stale"}}))
    with pytest.raises(ValueError, match="environment"):
        launcher.collect_results(plan, out)


def test_frozen_design_cannot_be_overwritten_with_changed_budgets(tmp_path: Path) -> None:
    path = tmp_path / "design.json"
    initial = {"tokens": 100, "version": "synthetic"}
    experiment.write_design(initial, path)
    original = path.read_bytes()
    experiment.write_design(initial, path)
    with pytest.raises(ValueError, match="version the experiment"):
        experiment.write_design({**initial, "tokens": 200}, path)
    assert path.read_bytes() == original
