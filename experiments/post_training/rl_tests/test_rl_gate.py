# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
import math
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from finestore.rl.rl_progress import Change, ComparisonRole
from finestore.rl.run_records import (
    EvaluationAnswerRow,
    EvaluationTaskRow,
    read_attempt,
    read_evaluations,
    write_attempt,
    write_evaluations,
)
from finestore.rl.strata import (
    MembershipState,
    SelectionDefinition,
    SelectionParent,
    SelectionTaskRow,
    read_selection_strata,
)
from finestore.schema import arrow_schema
from marin.execution.artifact import RECORD_FILENAME, Artifact, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, run
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel, SkyRLPolicyModel, SkyRLRun
from marin.rl.skyrl_records import RecordedLaunch, read_skyrl_metrics
from rigging.provenance import Provenance
from scipy.stats import t
from taskcompendium.grading_result import Outcome
from taskcompendium.identity import canonical_sha256

from experiments.post_training import rl as snowball
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_recipe as recipe
from experiments.post_training import rl_scoring as scoring
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_command import EXIT_CODES
from experiments.post_training.rl_scoring import GATE_INPUTS_FILENAME, GateInputs, read_gate_inputs
from experiments.post_training.rl_tests.fixtures import (
    SELECTION_IDENTITY,
    controls,
    edit_receipt,
    write_metrics,
    write_receipt,
)
from experiments.post_training.rl_tests.fixtures import answer as _answer
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    TASK_MANIFEST_FILENAME,
    EvalLiteRecipe,
    HeldoutSource,
    PanelFormat,
    eval_lite_step,
)

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]
KL = "policy/mismatch/staleness0/kl_k3"


def rescore(preset, *args):
    result = CliRunner().invoke(gate.command(snowball_model.SNOWBALL), ["--preset", preset, *args])
    assert result.exception is None or isinstance(result.exception, SystemExit), (result.output, result.exception)
    payload = json.loads(result.output.splitlines()[-1])
    assert result.exit_code == EXIT_CODES[payload["verdict"]], (
        result.output,
        result.exception,
    )
    return payload


def comparisons(report, role=None):
    return {row["metric"]: row for row in report["comparisons"] if role is None or row["role"] == role}


def edit_metrics(root, edit):
    metrics = read_skyrl_metrics(str(root))
    for values in metrics.values():
        edit(values)
    write_metrics(root, metrics)


def test_control_calibration_and_guard_tolerance():
    noise = gate.calibrate([{"x": value} for value in (1, 2, 3, 4)])["x"]
    assert (noise.mean, noise.sd, noise.df) == pytest.approx((2.5, math.sqrt(5 / 3), 3))
    row = gate._comparison("x", ComparisonRole.OBJECTIVE, [{"x": 4}], noise, 4, 2, higher_is_better=True)
    assert (row.control, row.run, row.delta) == (2.5, 4, 1.5)
    assert row.tolerance == pytest.approx(t.ppf(1 - 0.025 / 2, 3) * math.sqrt(5 / 3) * math.sqrt(1.25))
    assert row.change == Change.SAME
    zero = gate.calibrate([{"x": 2}, {"x": 2}])["x"]
    unavailable = gate._comparison("x", ComparisonRole.OBJECTIVE, [{"x": 4}], zero, 2)
    assert (unavailable.tolerance, unavailable.change) == (None, Change.UNAVAILABLE)


def test_rescore_leave_one_out_writes_the_printed_report(recorded_runs):
    controls(recorded_runs, "speed-medium")
    report = rescore("speed-medium", "--leave-out", "r0")
    assert (report["verdict"], report["label"]) == ("pass", "no change"), report
    artifact = Path(report.pop("report"))
    assert json.loads(artifact.read_text()) == report and artifact.with_suffix(".md").exists()
    assert {row["role"] for row in report["comparisons"]} >= {"objective", "constraint", "reported"}
    assert all(row["comparator"] is not None for row in report["invariants"])
    for name, speed, kl, verdict, label in (
        ("improvement", 90, 0.1, "pass", "improves"),
        ("regression", 150, 0.1, "fail", "regression"),
        ("trade-off", 90, 0.2, "fail", "trade-off"),
    ):
        candidate = "users/test/" + name
        recorded_runs("speed-medium", candidate, speed=speed, metrics={KL: kl})
        result = rescore("speed-medium", "--candidate", candidate)
        assert (result["verdict"], result["label"]) == (verdict, label), result


def test_extra_requirements_cannot_widen_existing_guards(recorded_runs):
    preset = snowball_model.SNOWBALL.by_name["speed-medium"]
    key = "policy/raw_grad_norm"
    pin = gate.control_pin(preset, table=snowball_model.SNOWBALL)
    control_roots = []
    for index in range(pin.runs):
        root = recorded_runs(
            preset.name,
            gate.control_name(pin.preset, pin.commit, index, table=snowball_model.SNOWBALL),
            speed=100 + index,
            metrics={key: 100 + index},
        )
        control_roots.append(root)
    candidates = []
    for index, value in enumerate((0, 0.01)):
        name = f"users/test/requirements-{index}"
        root = recorded_runs(preset.name, name, speed=108, metrics={key: value})
        candidates.append(ArtifactStep.adopt(name, preset.revision, str(root), kind=SkyRLRun))
    report = run(
        gate.gate_step(
            preset,
            tuple(candidates),
            inputs=GateInputs(
                preset_name=preset.name,
                requirements=(
                    key + "<=control",
                    key + "<=control",
                    key + "<=0",
                    "speed/update_seconds<=control",
                ),
            ),
            table=snowball_model.SNOWBALL,
        )
    )[0]
    assert report.verdict == "fail" and report.label == "regression"
    assert next(row for row in report.comparisons if row.metric == "speed/update_seconds").passes is False
    requirement = report.rows(ComparisonRole.REQUIREMENT)
    assert len(requirement) == 1 and requirement[0].passes
    assert requirement[0].tolerance == pytest.approx(t.ppf(0.975, 3) * math.sqrt(5 / 3) * math.sqrt(0.5 + 0.25))
    assert [(row.replicate, row.holds) for row in report.invariants if row.name == "require/" + key] == [
        (0, True),
        (1, False),
    ]

    table = snowball_model.SNOWBALL
    pin = table.controls[preset.controls]
    for root in control_roots:
        shutil.copytree(root, root.with_name("stale"))
    stale_table = replace(table, controls={**table.controls, preset.controls: replace(pin, revision="stale")})
    stale = run(
        gate.gate_step(
            preset,
            tuple(candidates),
            table=stale_table,
            inputs=GateInputs(preset_name=preset.name, requirements=("policy/raw_grad_norm<=11",)),
        )
    )[0]
    assert stale.verdict == "invalid" and "recorded control revision=stale" in stale.reason


def test_mismatch_uses_selected_tokens_over_the_measured_window(recorded_runs, preset_table):
    preset = replace(snowball_model.SNOWBALL.by_name["speed-long"], updates=3, measured=range(2, 4))
    preset_table(preset)
    prefix = KL.rsplit("/", 1)[0]
    for case, expected in (("weighted", 0.05), ("empty", None), ("nonfinite", None), ("pooled", 0.2)):
        metrics = {
            step: {
                "timing/step_wall/group_admission": value,
                "timing/step_wall/policy_training": 2,
                "timing/step_wall/checkpoint_work": 900,
                "timing/step_wall/evaluation": 700,
                KL: kl,
                prefix + "/selected_tokens": count,
                prefix + "/finite_fraction": 1,
                "policy/mismatch/pooled/kl_k3": 0,
            }
            for step, value, kl, count in [(1, 1e9, 1e9, 1000), (2, 5, 0.5, 1), (3, 9, 0, 9)]
        }
        for step in (2, 3):
            if case == "empty":
                metrics[step][prefix + "/selected_tokens"] = 0
            if case == "nonfinite":
                metrics[step][prefix + "/finite_fraction"] = 0
            if case == "pooled":
                metrics[step][KL] = 0.2
        root = recorded_runs(preset.name, "users/test/measured-mismatch-" + case)
        write_metrics(root, {step: {**row, "attempt_id": "recorded-attempt"} for step, row in metrics.items()})
        values = scoring.load_run(str(root), preset).values
        assert values["speed/update_seconds"] == 9
        assert (KL not in values) if expected is None else values[KL] == pytest.approx(expected)


def test_training_invariants_from_recorded_values(recorded_runs, preset_table):
    preset = replace(snowball_model.SNOWBALL.by_name["speed-medium"], updates=2, measured=range(1, 3))
    preset_table(preset)
    for fault in ("updates", "loss", "policy-loss", "union", "missing-counter"):
        metrics = {
            step: {"policy/raw_grad_norm": 1, "policy/policy_loss": 1, "policy/final_loss": 1, "loss/total": 1}
            for step in (1, 2)
        }
        counters = {
            "generate/outcome/" + value: 99 if value == Outcome.GRADED else 0 for value in (*Outcome, "rollout_failure")
        }
        if fault == "updates":
            metrics.pop(2)
        elif fault == "loss":
            metrics[2]["loss/total"] = float("nan")
        elif fault == "policy-loss":
            metrics[2]["policy/policy_loss"] = float("nan")
        elif fault == "union":
            counters.update({"generate/outcome/unavailable": 1, "generate/outcome/rollout_failure": 1})
        else:
            counters.pop("generate/outcome/graded")
        root = recorded_runs(preset.name, "users/test/training-invariant-" + fault)
        write_metrics(
            root, {step: {**row, **counters, "attempt_id": "recorded-attempt"} for step, row in metrics.items()}
        )
        facts = {fact.name: fact for fact in scoring.load_run(str(root), preset).invariants}
        expected = (
            "planned_updates"
            if fault == "updates"
            else "finite_losses_and_gradients" if fault in ("loss", "policy-loss") else "grader_error_fraction"
        )
        assert not facts[expected].holds


@pytest.mark.parametrize("preset", ["speed-short", "learning-medium"])
def test_current_generation_counters_compare_with_stored_controls(recorded_runs, preset):
    controls(recorded_runs, preset)
    root = recorded_runs(preset, "users/test/current-generation", speed=101)
    if preset == "speed-short":
        with edit_receipt(root) as receipt:
            launch = receipt["config"]
            source = launch["inputs"]["evaluation_data"]["eval_lite"][0]
            for callback in launch["skyrl"]["trainer"]["callbacks"]:
                if "eval_lite" in callback.get("additional_evaluations", {}):
                    callback["additional_evaluations"]["eval_lite"]["data_files"] = [
                        str(Path(source["local_path"]) / source["relative_path"])
                    ]
        evaluations = read_evaluations(
            str(root / "exports"), attempt_id="recorded-attempt", evaluation="eval/eval_lite/"
        )
        damaged = evaluations[0].model_copy(
            update={
                "answers": [
                    answer.model_copy(update={"error_treatment": "passthrough"}) for answer in evaluations[0].answers
                ]
            }
        )
        write_evaluations(str(root / "exports"), [damaged])
        facts = {
            fact.name: fact for fact in scoring.load_run(str(root), snowball_model.SNOWBALL.by_name[preset]).invariants
        }
        assert not facts["eval_lite/scored_fraction"].holds and facts["eval_lite/grader_error_fraction"].holds
        write_evaluations(str(root / "exports"), evaluations)
    metrics = read_skyrl_metrics(str(root))
    step = snowball_model.SNOWBALL.by_name[preset].updates
    report = rescore(preset, "--candidate", "users/test/current-generation")
    assert (report["verdict"], report["label"]) == ("pass", "no change"), report
    prefix = "eval/all/outcome/" if preset == "learning-medium" else "eval/eval_lite/all/outcome/"
    metrics[step][prefix + "graded"] = 19
    metrics[step][prefix + "unavailable"] = 1
    write_metrics(root, metrics)
    alias = root.parent.parent / "current-bad-counter" / root.name
    shutil.copytree(root, alias)
    report = rescore(preset, "--candidate", "users/test/current-bad-counter")
    assert (report["verdict"], report["label"]) == ("fail", "no change")
    assert any(row["name"] == "eval_lite/grader_error_fraction" and not row["holds"] for row in report["invariants"])


def test_named_evaluation_protocol_uses_its_recorded_request(recorded_runs):
    root = recorded_runs("speed-short", "users/test/recorded-named")
    launch = json.loads((root / "terminal.json").read_text())["config"]
    launch["skyrl"]["generator"]["eval_sampling_params"]["presence_penalty"] = 0.8
    launch["skyrl"]["generator"]["eval_n_samples_per_prompt"] = 7
    protocol = scoring.evaluation_protocol(RecordedLaunch.from_document(launch, attempt="recorded-attempt"), "eval_lite")
    assert {
        key: protocol["sampling_params"][key] for key in ("temperature", "ignore_eos", "max_generate_length", "logprobs")
    } == {
        "temperature": 0,
        "ignore_eos": False,
        "max_generate_length": 8192,
        "logprobs": None,
    }
    assert protocol["n_samples_per_prompt"] == 1
    assert protocol["context_budget"]["max_new_tokens_per_turn"] == 8192


def test_eval_lite_scores_join_permuted_answers_and_aggregate_tasks(tmp_path, recorded_runs):
    answers = [
        _answer("bounded", 2, low=1, high=5)
        | {"score": 0.8, "output_response": "first\u2028second\u2029third\u0085fourth"},
        _answer("bounded", None, status="error", error="TimeoutError"),
        _answer("mixed", 0),
        _answer("mixed", 1),
        _answer("stem", 0),
        _answer("stem", 0.8) | {"error_treatment": "passthrough"},
        _answer("general", 1),
        {key: value for key, value in _answer("general", 1).items() if key != "verification"},
        _answer("coding", 0.8),
        _answer("coding", 0.4),
        _answer("agents", 1),
        _answer("agents", 1),
    ]
    normalized = [0.25, None, 0, 1, 0, 0.8, 1, None, 0.8, 0.4, 1, 1]
    families = {
        "mixed": ("structural", "format", "hard"),
        "bounded": ("structural", "format", "learnable"),
        "stem": ("stem", "scientific_reasoning", "hard"),
        "general": ("general", "general", "easy"),
        "coding": ("competitive_coding", "coding", "learnable"),
        "agents": ("agents", "agents", "easy"),
    }
    root = recorded_runs("learning-medium", "users/test/score-order")
    native = json.loads((root / "terminal.json").read_text())
    launch = native["config"]
    launch["skyrl"]["generator"]["eval_n_samples_per_prompt"] = 2
    source = launch["inputs"]["validation_data"][0]
    base = json.loads(pq.read_table(Path(source["uri"]) / source["relative_path"])["task_spec"][0].as_py())
    selection = tmp_path / "selection"
    selection.mkdir()
    specs = {task_id: {**base, "id": task_id} for task_id in families}
    manifest = [
        SelectionTaskRow(
            task_id=task_id,
            task_digest=canonical_sha256(specs[task_id]),
            family=family,
            aai_category=category,
            difficulty=difficulty,
            agentic=False,
            state=MembershipState.ACTIVE,
        )
        for task_id, (family, category, difficulty) in families.items()
    ]
    pq.write_table(
        pa.table({"task_spec": [json.dumps(value) for value in specs.values()]}), selection / "validation.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist([row.model_dump(mode="json") for row in manifest], schema=arrow_schema(SelectionTaskRow)),
        selection / TASK_MANIFEST_FILENAME,
    )
    _json(
        selection / "summary.json",
        {
            "selection_strata": (
                SelectionDefinition(
                    name="permuted", version="2026.11.17.1", sampling="all", task_files=("validation.parquet",)
                ).model_dump(mode="json")
            )
        },
    )
    launch["artifacts"]["export_root"] = str(tmp_path / "configured-export")
    source["uri"] = str(selection)
    write_receipt(tmp_path, native)
    evaluation = snowball_model.EVALUATIONS["eval_lite_100"]
    export, attempt_id = launch["artifacts"]["export_root"], launch["run"]["attempt_id"]
    annotations = {row.task_id: row for row in manifest}
    task_rows = []
    for step in (-1, 25):
        for task_id in families:
            task_answers = []
            for ordinal, (answer, score) in enumerate(zip(answers, normalized, strict=True)):
                if answer["task_id"] != task_id:
                    continue
                verification = answer.get("verification")
                task_answers.append(
                    EvaluationAnswerRow(
                        ordinal=ordinal,
                        native_uid=str(ordinal),
                        optimization_reward=answer.get("score", 0.0),
                        verifier_score=score,
                        raw_verifier_score=verification["score"] if verification else None,
                        score_min=verification["score_min"] if verification else None,
                        score_max=verification["score_max"] if verification else None,
                        verification_status=verification["status"] if verification else None,
                        passed=score == 1,
                        outcome="graded" if score is not None else "unavailable",
                        error_type=answer.get("exception_type"),
                        error_treatment=answer.get("error_treatment"),
                    )
                )
            task_rows.append(
                EvaluationTaskRow(
                    attempt_id=attempt_id,
                    step=step,
                    evaluation="eval/",
                    panel_name="permuted",
                    panel_version="2026.11.17.1",
                    task_id=task_id,
                    data_source="fixture",
                    strata=annotations[task_id].strata,
                    task_digest=annotations[task_id].task_digest,
                    score=None,
                    outcome="mixed",
                    error_type=None,
                    answers=task_answers,
                )
            )
    write_evaluations(export, task_rows)
    write_evaluations(
        export,
        [
            row.model_copy(
                update={
                    "attempt_id": "other-attempt",
                    "answers": [answer.model_copy(update={"verifier_score": 0.0}) for answer in row.answers],
                }
            )
            for row in task_rows
        ],
    )
    current = scoring.evaluation_scores(evaluation, str(tmp_path))
    with pytest.raises(scoring.InvalidComparison, match="dump is missing for attempt"):
        scoring.evaluation_answers(str(tmp_path))
    directory = Path(export) / "dumped_evals" / attempt_id / "global_step_25_evals"
    directory.mkdir(parents=True)
    current_answers = [dict(row, attempt_id=attempt_id, step=25, is_last_step=True) for row in answers]
    current_answers.insert(0, dict(current_answers[0], is_last_step=False))
    (directory / "task_set.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in current_answers) + "\n"
    )
    assert [row.answer["task_id"] for row in scoring.evaluation_answers(str(tmp_path))] == [
        row["task_id"] for row in answers
    ]
    statistics = scoring.evaluation_statistics(current.filter(pc.equal(current["step"], 25)))
    expected = {
        "mean": 109 / 180,
        "family_mean": 0.595,
        "mixed_fraction": 1 / 6,
        "all_right_fraction": 1 / 3,
        "all_wrong_fraction": 0.5,
        "scored_fraction": 0.75,
        "scored": 9,
        "attempted": 12,
        "aai_category/format/mean": 0.375,
        "difficulty/hard/mean": 0.25,
        "aai_weighted_mean": 0.72,
        "aai_category_weight_coverage": 1,
        "aai_category/scientific_reasoning/weight": 0.2,
        "aai_category/coding/weight": 0.2,
        "aai_category/general/weight": 0.3,
        "aai_category/agents/weight": 0.3,
    }
    assert {key: statistics["eval_lite_100/" + key] for key in expected} == pytest.approx(expected)
    unavailable = current.set_column(
        current.schema.get_field_index("scored"),
        "scored",
        pa.array([row["scored"] and row["item"] != "stem" for row in current.to_pylist()]),
    ).filter(pc.equal(current["step"], 25))
    assert "eval_lite_100/aai_weighted_mean" not in scoring.evaluation_statistics(unavailable)

    (selection / FULL_VALIDATION_FILENAME).write_bytes((selection / "validation.parquet").read_bytes())
    addition = {
        **base,
        "id": "addition",
        "tags": ["eval"],
        "context": {**base["context"], "events": [{**base["context"]["events"][0], "content": "Return a new answer."}]},
    }
    addition_file, excluded_file = tmp_path / "addition.parquet", tmp_path / "excluded.parquet"
    pq.write_table(pa.table({"task_spec": [json.dumps(addition)]}), addition_file)
    pq.write_table(pa.table({"task_spec": [json.dumps({**base, "id": "excluded"})]}), excluded_file)
    rates_file = tmp_path / "reference.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": task_id,
                    "task_digest": canonical_sha256(document),
                    "model_revision": "reference-model",
                    "attempted": 8,
                    "samples": 8,
                    "passed": 0 if task_id == "bounded" else 4,
                    "mean_score": 0.0 if task_id == "bounded" else 0.5,
                }
                for task_id, document in specs.items()
            ]
        ),
        rates_file,
    )
    version = "2026.11.17.2"
    successor = EvalLiteRecipe(
        name="permuted",
        version=version,
        parent=SelectionParent(root=str(selection), name="permuted", version="2026.11.17.1"),
        seed=3,
        sources=(
            {
                "source_id": "addition",
                "rows": 1,
                "format": PanelFormat.TASK_SPEC,
                "family": "structural",
                "aaii_category": "format",
                "license": "fixture",
                "count": 1,
            },
        ),
        training=(),
        development=HeldoutSource(uri=str(excluded_file), version=version, rows=1, format=PanelFormat.TASK_SPEC),
        fast_counts={"structural": 2, "stem": 1, "general": 1, "competitive_coding": 1, "agents": 1},
        pass_rates={"uri": str(rates_file), "version": version, "rows": len(specs), "model_revision": "reference-model"},
    )
    successor_root = run(
        eval_lite_step(
            {"addition": ArtifactStep.adopt("fixture/version-addition", version, str(addition_file))}, successor
        )
    )[0].path
    child = read_selection_strata(successor_root)
    assert child is not None
    child_manifest_path = Path(successor_root) / TASK_MANIFEST_FILENAME
    child_manifest = pq.read_table(child_manifest_path).to_pylist()
    for annotation in child_manifest:
        if annotation["task_id"] == "mixed":
            annotation["difficulty"] = "easy"
    pq.write_table(pa.Table.from_pylist(child_manifest, schema=arrow_schema(SelectionTaskRow)), child_manifest_path)
    child = read_selection_strata(successor_root)
    assert child.tasks["mixed"].strata.difficulty != annotations["mixed"].strata.difficulty
    updated = [
        row.model_copy(
            update={"panel_name": "permuted", "panel_version": version, "strata": child.tasks[row.task_id].strata}
        )
        for row in task_rows
    ]
    addition_answers = [
        updated[-1].answers[0].model_copy(update={"ordinal": len(answers) + i, "native_uid": f"addition-{i}"})
        for i in range(2)
    ]
    updated.append(
        updated[-1].model_copy(
            update={
                "task_id": "addition",
                "task_digest": child.tasks["addition"].task_digest,
                "strata": child.tasks["addition"].strata,
                "answers": addition_answers,
            }
        )
    )
    # Use the later evaluation's full pool to recover both versions, including the retired task.
    complete = [
        row.model_copy(
            update={
                "answers": [
                    row.answers[0].model_copy(
                        update={
                            "verifier_score": 0.0 if row.task_id == "bounded" else 1.0,
                            "verification_status": "verified",
                            "error_type": None,
                            "error_treatment": None,
                        }
                    )
                ]
            }
        )
        for row in updated
        if row.step == 25
    ]
    version_export = str(tmp_path / "version-scoring")
    write_evaluations(version_export, complete)
    stored = read_evaluations(version_export, attempt_id=attempt_id, evaluation="eval/", step=25)
    parent = read_selection_strata(str(selection))
    assert parent is not None
    assert scoring.score_eval_lite_version(stored, parent) == scoring.EvalLiteVersionScore(5 / 6, 6, (), ())
    assert scoring.score_eval_lite_version(stored, child) == scoring.EvalLiteVersionScore(1.0, 6, (), ())

    damaged = [
        (
            row.model_copy(update={"task_digest": "different-content"})
            if row.task_id == "coding"
            else row.model_copy(update={"answers": []}) if row.task_id == "general" else row
        )
        for row in stored
        if row.task_id != "stem"
    ]
    assert scoring.score_eval_lite_version(damaged, parent) == scoring.EvalLiteVersionScore(
        None, 3, ("general", "stem"), ("coding",)
    )
    with pytest.raises(ValueError, match="select one evaluation, attempt and step"):
        scoring.score_eval_lite_version([*stored, stored[0].model_copy(update={"step": 0})], parent)

    write_evaluations(export, updated)
    source.update(uri=successor_root, relative_path=FULL_VALIDATION_FILENAME)
    write_receipt(tmp_path, native)
    previous = scoring.evaluation_scores(evaluation, str(tmp_path), panel_root=str(selection), step=25)
    assert previous.equals(current.filter(pc.equal(current["step"], 25)))
    target = next(row for row in updated if row.task_id == "bounded" and row.step == 25)
    for length in (0, 1):
        write_evaluations(export, [target.model_copy(update={"answers": target.answers[:length]})])
        with pytest.raises(scoring.InvalidComparison, match="incomplete answer coverage"):
            scoring.evaluation_scores(evaluation, str(tmp_path), panel_root=str(selection), step=25)
    write_evaluations(export, updated)
    measured_file = Path(successor_root) / FULL_VALIDATION_FILENAME
    original = measured_file.read_bytes()
    changed = pq.read_table(measured_file).to_pylist()
    changed_row = next(row for row in changed if json.loads(row["task_spec"])["id"] == "bounded")
    changed_spec = json.loads(changed_row["task_spec"])
    changed_spec["grader"]["parameters"] = {"expected": ["a different answer"]}
    changed_row["task_spec"] = json.dumps(changed_spec)
    pq.write_table(pa.Table.from_pylist(changed), measured_file)
    with pytest.raises(scoring.InvalidComparison, match="task content differs"):
        scoring.evaluation_scores(evaluation, str(tmp_path), panel_root=str(selection), step=25)
    measured_file.write_bytes(original)
    old_launch = RecordedLaunch.from_document(launch, attempt=attempt_id)
    old_receipt = read_attempt(export, attempt_id)
    active_attempt = "active-only"
    launch["run"]["attempt_id"] = native["result"]["attempt_id"] = active_attempt
    write_receipt(tmp_path, native)
    active_record = read_attempt(export, active_attempt)
    assert active_record is not None
    write_attempt(export, active_record.model_copy(update={"evaluation_storage_version": 1}))
    write_evaluations(
        export,
        [
            row.model_copy(update={"attempt_id": active_attempt})
            for row in updated
            if child.tasks[row.task_id].state == MembershipState.ACTIVE
        ],
    )
    with pytest.raises(scoring.InvalidComparison, match="recomputation is incomplete"):
        scoring.evaluation_scores(evaluation, str(tmp_path), panel_root=str(selection), step=25)
    with pytest.raises(scoring.InvalidComparison, match="differs from the selected receipt"):
        scoring.evaluation_scores(evaluation, str(tmp_path), launch=old_launch)
    with pytest.raises(scoring.InvalidComparison, match="differs from the selected receipt"):
        scoring.evaluation_answers(str(tmp_path), old_launch)
    assert scoring.evaluation_scores(
        evaluation, str(tmp_path), launch=old_launch, receipt=old_receipt, panel_root=str(selection), step=25
    ).equals(previous)


@pytest.mark.parametrize(
    "case",
    [
        "initializer",
        "initializer-hf",
        "stale-default",
        "control-commit",
        "unbound-data",
        "batch",
        "allocation",
        "cached-default",
        "zero-spread",
        "missing-guard",
        "omitted-default",
        "omitted-current-nondefault",
        "omitted-schema-nondefault",
        "omitted-candidate-default",
        "schema-timeout",
        "protocol-temperature",
        "malformed-seed",
    ],
)
def test_gate_compares_recorded_configuration_and_defaults(preset_table, recorded_runs, tmp_path, monkeypatch, case):
    preset = snowball_model.SNOWBALL.by_name["speed-medium" if case == "missing-guard" else "learning-medium"]
    setting = "trainer.policy.optimizer_config.lr=2e-6"
    settings = (
        ()
        if case
        in (
            "stale-default",
            "cached-default",
            "zero-spread",
            "omitted-default",
            "omitted-current-nondefault",
            "omitted-schema-nondefault",
            "omitted-candidate-default",
            "schema-timeout",
        )
        else (setting,)
    )
    if case == "batch":
        settings = (
            "trainer.train_batch_size=64",
            "trainer.policy_mini_batch_size=64",
            "generator.n_samples_per_prompt=8",
            "trainer.micro_train_batch_size_per_gpu=2",
        )
    inputs = GateInputs(preset_name=preset.name, settings=settings)
    if case.startswith("initializer"):
        name, version = "tests/custom-init", "2026.11.07"
        root = tmp_path / name / version
        if case == "initializer-hf":
            initializer = ArtifactHfModel(
                ArtifactStep.adopt(name, version, "s3://fixture-initializer", kind=Artifact),
                preset.model.tokenizer_uri,
                preset.model.tokenizer_revision,
            )
            monkeypatch.setattr(
                "s3fs.S3FileSystem.exists",
                lambda self, path, **kwargs: str(path).removeprefix("s3://")
                in {"fixture-initializer", "fixture-initializer/policy/tokenizer_config.json"},
            )
        else:
            model = SkyRLRun(
                path=str(root),
                hf_model_uri="s3://fixture-initializer/policy",
                global_step=25,
                tokenizer_uri=preset.model.tokenizer_uri,
                tokenizer_revision=preset.model.tokenizer_revision,
                checkpoint_root=str(root / "checkpoints"),
                draft_checkpoint_root=None,
                terminal_manifest_uri=str(root / "terminal.json"),
                iris_job_id="/test-user/initializer",
            )
            _json(
                root / RECORD_FILENAME,
                ArtifactRecord(
                    name=name,
                    version=version,
                    fingerprint="initializer",
                    result_type=result_type_name(SkyRLRun),
                    result=model.result_payload(),
                ).model_dump(mode="json"),
            )
            initializer = SkyRLPolicyModel(ArtifactStep.adopt(name, version, str(root), kind=SkyRLRun))
        inputs = inputs.model_copy(update={"init": recipe.stored_initializer(initializer, preset=preset)})
    control_roots = controls(recorded_runs, preset.name)
    if case == "zero-spread":
        pin = gate.control_pin(preset, table=snowball_model.SNOWBALL)
        control_roots = [
            recorded_runs(
                preset.name, gate.control_name(pin.preset, pin.commit, index, table=snowball_model.SNOWBALL), score=0.5
            )
            for index in range(pin.runs)
        ]
    if case in ("stale-default", "omitted-current-nondefault"):
        changed = setting if case == "stale-default" else 'generator.chat_template_kwargs={"enable_thinking":false}'
        preset = replace(preset, recipe=replace(preset.recipe, config=preset.recipe.config.with_settings((changed,))))
        preset_table(preset)
    name = "users/test/config-" + case
    if case == "initializer-hf":
        with monkeypatch.context() as planning:
            planning.setattr(
                skyrl_api, "discover_hf_checkpoints", lambda *_: pytest.fail("gate planning read the initializer export")
            )
            planned = gate.validate_candidate(preset, inputs, table=snowball_model.SNOWBALL)
            assert planned["inputs"]["model"]["uri"].endswith("/<terminal-hf-export>")
        return
    candidate = recorded_runs(preset.name, name, inputs=inputs)
    fault = {
        "stale-default": "control config differs",
        "control-commit": "does not match its pin",
        "unbound-data": "skyrl.data.train_data",
        "allocation": "iris.allocation.num_nodes",
        "zero-spread": "variation is zero",
        "missing-guard": "missing or non-finite guard metric",
        "omitted-current-nondefault": "control config differs",
        "omitted-schema-nondefault": "control config differs",
        "omitted-candidate-default": "skyrl.generator.chat_template_kwargs",
        "schema-timeout": "timed out after 120 s",
        "protocol-temperature": "temperature",
        "malformed-seed": "malformed recorded configuration",
    }.get(case)
    if case.startswith(("omitted-", "schema-")):
        omitted = [candidate] if case == "omitted-candidate-default" else control_roots
        for old in omitted:
            with edit_receipt(old) as receipt:
                if case == "omitted-schema-nondefault":
                    assert receipt["config"]["skyrl"]["trainer"]["policy"]["optimizer_config"].pop("weight_decay") == 0.0
                else:
                    assert receipt["config"]["skyrl"]["generator"].pop("chat_template_kwargs") == {}
                if case == "omitted-default":
                    assert receipt["config"]["inputs"].pop("starting_checkpoint") is None
                    assert receipt["config"]["iris"].pop("role_topology") is None
    if case == "schema-timeout":
        original_subprocess = subprocess.run
        for diagnostic in ("timeout", "empty", "whitespace", "oserror"):
            with monkeypatch.context() as unavailable:
                unavailable.setattr(
                    Provenance,
                    "capture",
                    lambda *_, tag=diagnostic: Provenance(tag, "commit-one", False, "branch", "test"),
                )

                def unavailable_defaults(command, *args, diagnostic=diagnostic, **kwargs):
                    if any("OmegaConf.structured(launch_config.SkyRLLaunchConfig)" in str(value) for value in command):
                        if diagnostic == "timeout":
                            raise subprocess.TimeoutExpired(command, 120)
                        if diagnostic == "oserror":
                            raise FileNotFoundError("native composer executable missing")
                        return subprocess.CompletedProcess(command, 1 if diagnostic == "whitespace" else 0, "", "\n")
                    return original_subprocess(command, *args, **kwargs)

                unavailable.setattr(subprocess, "run", unavailable_defaults)
                report = rescore(preset.name, "--candidate", name)
                assert (
                    report["verdict"] == "invalid" and "native configuration cannot be composed" in report["reason"]
                ), report
                if diagnostic == "timeout":
                    assert "timed out after 120 s" in report["reason"]
        return
    root = control_roots[0] if case == "control-commit" else candidate
    if case in ("control-commit", "unbound-data", "allocation"):
        with edit_receipt(root) as receipt:
            launch = receipt["config"]
            if case == "control-commit":
                receipt["result"]["provenance"]["base_commit"] = "wrong-commit"
            elif case == "unbound-data":
                launch["skyrl"]["data"]["train_data"] = ["/unbound/train.parquet"]
            else:
                launch["iris"]["allocation"]["num_nodes"] += 1
    if case in {"protocol-temperature", "malformed-seed"}:
        with edit_receipt(candidate) as receipt:
            if case == "malformed-seed":
                receipt["config"]["skyrl"]["trainer"]["seed"] = None
            else:
                receipt["config"]["skyrl"]["generator"]["eval_sampling_params"]["temperature"] = 0.5
    if case == "missing-guard":
        edit_metrics(candidate, lambda row: row.update({"policy/mismatch/staleness0/selected_tokens": 0}))
    report = rescore(preset.name, "--candidate", name)
    if case == "protocol-temperature":
        assert report["verdict"] == "invalid" and "evaluation protocol" in report["reason"], report
    elif fault:
        assert report["verdict"] == "invalid" and fault in report["reason"], report
        if case == "malformed-seed":
            assert "seed" in report["reason"]
    else:
        assert (report["verdict"], report["label"]) == ("pass", "no change"), report
        if case.startswith("initializer") or case == "batch":
            assert (
                "inputs.model.uri" if case.startswith("initializer") else "skyrl.trainer.train_batch_size"
            ) in report["run"]["config_changes"]
    if case == "cached-default":
        promoted = replace(
            preset,
            recipe=replace(
                preset.recipe,
                config=preset.recipe.config.with_settings(("trainer.policy.optimizer_config.weight_decay=0.1",)),
            ),
        )
        preset_table(promoted)
        updated = rescore(preset.name, "--candidate", name)
        assert updated["verdict"] == "invalid" and "weight_decay" in updated["reason"]
        assert updated["report"] != report["report"]


def test_task_set_content_hash_is_order_independent_and_checked(recorded_runs):
    preset = "learning-medium"
    controls(recorded_runs, preset)
    source = recorded_runs(preset, "users/test/task-content")
    receipt = json.loads((source / "terminal.json").read_text())
    descriptor = receipt["config"]["inputs"]["validation_data"][0]
    original = Path(descriptor["uri"]) / descriptor["relative_path"]
    rows = pq.read_table(original).to_pylist()
    inactive = json.loads(rows[0]["task_spec"])
    inactive["id"] = "retained-inactive"
    panel_root = original.parent
    pool_filename = "retained_pool.parquet"
    manifest_path = panel_root / TASK_MANIFEST_FILENAME
    annotations = pq.read_table(manifest_path).to_pylist()
    summary_path = panel_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    pool = pq.read_table(panel_root / summary["selection_strata"]["task_files"][0]).to_pylist()
    annotations.append(
        SelectionTaskRow(
            task_id=inactive["id"],
            task_digest=canonical_sha256(inactive),
            state=MembershipState.RETIRED,
        ).model_dump(mode="json")
    )
    pq.write_table(pa.Table.from_pylist(annotations, schema=arrow_schema(SelectionTaskRow)), manifest_path)
    pq.write_table(pa.Table.from_pylist([*pool, {"task_spec": json.dumps(inactive)}]), panel_root / pool_filename)
    summary["selection_strata"]["task_files"] = [pool_filename]
    summary_path.write_text(json.dumps(summary))
    task_set = scoring.read_recorded_task_set(
        RecordedLaunch.from_document(receipt["config"], attempt=receipt["config"]["run"]["attempt_id"]), "eval_lite_100"
    )
    assert set(task_set.task_ids) == {json.loads(row["task_spec"])["id"] for row in rows}
    assert inactive["id"] not in task_set.task_ids
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), original)
    report = rescore(preset, "--candidate", "users/test/task-content")
    assert report["verdict"] == "pass", report
    changed = recorded_runs(preset, "users/test/task-content-edited")
    with edit_receipt(changed) as receipt:
        descriptor = receipt["config"]["inputs"]["validation_data"][0]
        copied = changed / "task-set-copy"
        shutil.copytree(descriptor["uri"], copied)
        descriptor["uri"] = str(copied)
        receipt["config"]["skyrl"]["data"]["val_data"] = [dict(descriptor)]
        path = copied / descriptor["relative_path"]
        rows = pq.read_table(path).to_pylist()
        task = json.loads(rows[0]["task_spec"])
        task["context"]["events"][0]["content"] = "Return dog."
        rows[0]["task_spec"] = json.dumps(task)
        pq.write_table(pa.Table.from_pylist(rows), path)
        annotations = pq.read_table(copied / TASK_MANIFEST_FILENAME).to_pylist()
        for annotation in annotations:
            if annotation["task_id"] == task["id"]:
                annotation["task_digest"] = canonical_sha256(task)
        pq.write_table(
            pa.Table.from_pylist(annotations, schema=arrow_schema(SelectionTaskRow)), copied / TASK_MANIFEST_FILENAME
        )
        for filename in {pool_filename, *summary["selection_strata"]["active_files"]}:
            pool = pq.read_table(copied / filename).to_pylist()
            for row in pool:
                if json.loads(row["task_spec"])["id"] == task["id"]:
                    row["task_spec"] = json.dumps(task)
            pq.write_table(pa.Table.from_pylist(pool), copied / filename)
    report = rescore(preset, "--candidate", "users/test/task-content-edited")
    assert report["verdict"] == "invalid" and "TaskSpec content differs" in report["reason"], report


@pytest.mark.parametrize("preset_name", ["learning-medium", "data-medium"])
def test_python_run_saves_typed_inputs_for_rescoring(
    preset_table, recorded_runs, selection, tmp_path, monkeypatch, preset_name
):
    preset = snowball_model.SNOWBALL.by_name[preset_name]
    model = tmp_path / "model"
    model.mkdir()
    model_uri = f"s3://marin-test/models/{tmp_path.name}"

    def local_model_path(path):
        return model / str(path).removeprefix("s3://").removeprefix(model_uri.removeprefix("s3://")).lstrip("/")

    monkeypatch.setattr(
        "s3fs.S3FileSystem.glob",
        lambda self, path, **kwargs: [f"{model_uri}/{value.relative_to(model)}" for value in model.rglob("config.json")],
    )
    monkeypatch.setattr(
        "s3fs.S3FileSystem.modified",
        lambda self, path: datetime.fromtimestamp(local_model_path(path).stat().st_mtime, UTC),
    )
    monkeypatch.setattr("s3fs.S3FileSystem.exists", lambda self, path, **kwargs: local_model_path(path).exists())
    local = replace(
        preset.model,
        step=ArtifactStep.adopt(
            preset.model.step.name,
            preset.model.step.version,
            model_uri,
            kind=preset.model.step.artifact_type,
        ),
    )
    preset = replace(preset, model=local)
    preset_table(preset)
    controls(recorded_runs, preset_name)
    tasks = selection if preset_name == "data-medium" else None
    setting_text = (
        ("trainer.policy.optimizer_config.lr=2.0e-6", 'extra_env.VLLM_PROBE="1"', "generator.max_num_seqs=1024")
        if tasks is None
        else ()
    )
    settings = setting_text
    export = model / "selected-hf"
    export.mkdir()
    for filename in ("config.json", "tokenizer_config.json"):
        (export / filename).write_text("{}")
    initializer = (
        ArtifactHfModel(
            ArtifactStep.adopt(
                "tests/noncanonical-initializer",
                "2026.11.17",
                model_uri,
                kind=local.step.artifact_type,
                config={"tokenizer_uri": "tests/own-tokenizer"},
            ),
            "tests/own-tokenizer",
            "own-tokenizer-revision",
        )
        if preset.accepts_init
        else None
    )
    built = snowball.build_run(
        preset,
        settings=settings,
        tasks=tasks,
        init=initializer,
        requirements=("policy/raw_grad_norm<=10",),
        table=snowball_model.SNOWBALL,
    )
    task_name = SELECTION_IDENTITY if tasks else None
    source = recorded_runs(
        preset.name,
        built.runs[0].name,
        fingerprint=built.runs[0].fingerprint(),
        inputs=built.inputs,
    )
    staged = source.with_name("native-staging")
    source.rename(staged)
    (staged / GATE_INPUTS_FILENAME).unlink()

    def native(config, *, forward_env=()):
        shutil.copytree(staged, source, dirs_exist_ok=True)
        return SkyRLRun.raw_load(str(source))

    monkeypatch.setattr(snowball, "run_skyrl", native)
    monkeypatch.setattr(sys, "argv", ["unrelated-command", "--preset", "train-long", "--require", "unrelated"])
    run(*built.runs)
    recorded = read_gate_inputs(str(source))
    assert recorded.preset_name == preset_name and recorded.requirements == ("policy/raw_grad_norm<=10",)
    decoded = dict(item.split("=", 1) for item in recorded.settings)
    assert {key: json.loads(value) for key, value in decoded.items()} == (
        {}
        if tasks
        else {
            "trainer": {"policy": {"optimizer_config": {"lr": 2e-6}}},
            "extra_env": {"VLLM_PROBE": "1"},
            "generator": {"max_num_seqs": 1024},
        }
    )
    assert recorded.tasks == task_name
    if initializer is not None:
        assert (
            recorded.init.artifact_root,
            recorded.init.relative_path,
            recorded.init.tokenizer_uri,
            recorded.init.tokenizer_revision,
        ) == (model_uri, None, "tests/own-tokenizer", "own-tokenizer-revision")
        assert json.loads(recorded.init.fingerprint_payload)["config"]["tokenizer_uri"] == "tests/own-tokenizer"
        newer = model / "newer-hf"
        newer.mkdir()
        for filename in ("config.json", "tokenizer_config.json"):
            (newer / filename).write_text("{}")
    report = rescore(preset_name, "--candidate", built.runs[0].name)
    assert report["verdict"] == "pass", report
    assert report["run"]["requirements"] == list(recorded.requirements)
    assert report["run"]["settings"] == list(recorded.settings)
    assert report["run"]["tasks"] == recorded.tasks
    assert report["run"]["attempt_ids"] == ["recorded-attempt"]
    assert comparisons(report)["eval_lite/family_mean/gain_per_h100_hour"]["run"] == pytest.approx(0.005)
    if tasks is None:
        assert {"inputs.model.uri", "skyrl.trainer.policy.optimizer_config.lr", "skyrl.generator.max_num_seqs"} <= set(
            report["run"]["config_changes"]
        )
