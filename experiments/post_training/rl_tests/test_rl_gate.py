# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
import math
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from finestore.rl.rl_progress import (
    EVALUATION_PROTOCOL_DIFFERS,
    SHARED_TASK_FRACTION_INVARIANT,
    SHARED_TASKS_INVARIANT,
    ComparisonRole,
)
from finestore.rl.run_records import (
    read_attempt,
    read_evaluations,
    write_evaluations,
)
from finestore.rl.strata import (
    MembershipState,
    SelectionDefinition,
    SelectionParent,
    SelectionTaskRow,
    read_selection_manifest,
)
from finestore.schema import arrow_schema
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel, SkyRLPolicyModel, SkyRLRun
from marin.rl.skyrl_records import RecordedLaunch, read_skyrl_metrics
from rigging.provenance import Provenance
from scipy.stats import t
from skyrlrecipe.documents import get_path, set_path
from taskcompendium.grading_result import Outcome
from taskcompendium.identity import canonical_sha256

from experiments.post_training import rl as snowball
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_measurements as measurements
from experiments.post_training import rl_scoring as scoring
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_command import EXIT_CODES
from experiments.post_training.rl_measurements import RUN_INPUTS_FILENAME, RunInputs, RunOptions, read_run_inputs
from experiments.post_training.rl_tests.fixtures import (
    SELECTION_IDENTITY,
    cache_native_compositions,
    controls,
    edit_receipt,
    evaluation_task_rows,
    fake_s3,
    fake_s3_checkpoint,
    write_metrics,
    write_receipt,
    write_skyrl_run_record,
)
from experiments.post_training.rl_tests.fixtures import answer as _answer
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    TASK_MANIFEST_FILENAME,
    VALIDATION_FILENAME,
)

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]
KL = "policy/mismatch/staleness0/kl_k3"


def compare(preset, *args):
    result = CliRunner().invoke(gate.command(snowball_model.SNOWBALL), ["--preset", preset, *args])
    assert result.exception is None or isinstance(result.exception, SystemExit), (result.output, result.exception)
    payload = json.loads(result.stdout.splitlines()[-1])
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


def test_compare_leave_one_out_writes_the_printed_report(recorded_runs):
    controls(recorded_runs, "speed-medium")
    report = compare("speed-medium", "--leave-out", "r0")
    assert (report["verdict"], report["effect"]) == ("pass", "no change"), report
    artifact = Path(report.pop("report"))
    assert json.loads(artifact.read_text()) == report and artifact.with_suffix(".md").exists()
    assert {row["role"] for row in report["comparisons"]} >= {"objective", "guard", "reported"}
    assert all(row["comparator"] is not None for row in report["invariants"])
    for name, speed, kl, verdict, effect in (
        ("improvement", 90, 0.1, "pass", "improves"),
        ("regression", 150, 0.1, "fail", "regression"),
        ("trade-off", 90, 0.2, "fail", "trade-off"),
    ):
        candidate = "users/test/" + name
        recorded_runs("speed-medium", candidate, speed=speed, metrics={KL: kl})
        result = compare("speed-medium", "--candidate", candidate)
        assert (result["verdict"], result["effect"]) == (verdict, effect), result


def test_extra_requirements_cannot_widen_existing_guards(recorded_runs):
    preset = snowball_model.SNOWBALL.by_name["speed-medium"]
    key = "policy/raw_grad_norm"
    pin = gate.control_pin(preset, table=snowball_model.SNOWBALL)
    control_roots = []
    for index in range(pin.replicates):
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
            inputs=RunInputs(
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
    assert report.verdict == "fail" and report.effect == "regression"
    speed_guard = next(row for row in report.comparisons if row.metric == "speed/update_seconds")
    assert speed_guard.passes is False
    assert speed_guard.tolerance == pytest.approx(t.ppf(1 - 0.025 / 2, 3) * math.sqrt(5 / 3) * math.sqrt(0.75))
    requirement = report.rows(ComparisonRole.REQUIREMENT)
    assert len(requirement) == 1 and requirement[0].passes
    assert requirement[0].tolerance == pytest.approx(t.ppf(0.975, 3) * math.sqrt(5 / 3) * math.sqrt(0.5 + 0.25))
    assert [(row.replicate, row.holds) for row in report.invariants if row.name == "require/" + key] == [
        (0, True),
        (1, False),
    ]

    table = snowball_model.SNOWBALL
    pin = table.control_pins[preset.control_preset]
    for root in control_roots:
        shutil.copytree(root, root.with_name("stale"))
    stale_table = replace(
        table, control_pins={**table.control_pins, preset.control_preset: replace(pin, revision="stale")}
    )
    stale = run(
        gate.gate_step(
            preset,
            tuple(candidates),
            table=stale_table,
            inputs=RunInputs(preset_name=preset.name, requirements=("policy/raw_grad_norm<=11",)),
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
        values = measurements.read_measurements(str(root), preset).values
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
        facts = {fact.name: fact for fact in measurements.read_measurements(str(root), preset).invariants}
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
            fact.name: fact
            for fact in measurements.read_measurements(str(root), snowball_model.SNOWBALL.by_name[preset]).invariants
        }
        assert not facts["eval_lite/task_coverage"].holds and facts["eval_lite/grader_error_fraction"].holds
        write_evaluations(str(root / "exports"), evaluations)
        export = str(root / "exports")
        profile = snowball_model.SNOWBALL.by_name[preset]
        primary = read_evaluations(export, attempt_id="recorded-attempt", evaluation="eval/")

        def counts(row, value):
            return row.model_copy(
                update={"answers": [answer.model_copy(update={"response_token_count": value}) for answer in row.answers]}
            )

        def fixed_reply_holds():
            return next(
                fact.holds
                for fact in measurements.read_measurements(str(root), profile).invariants
                if fact.name == "fixed_reply_tokens"
            )

        write_evaluations(export, [counts(primary[0], 4095)])
        assert not fixed_reply_holds()
        write_evaluations(export, [counts(row, None) for row in primary])
        legacy_metrics = read_skyrl_metrics(str(root))
        tokens = 4096 * sum(len(row.answers) for row in primary)
        legacy_metrics[0].update({"eval/all/response_tokens_max": 4096, "eval/all/response_tokens": tokens - 1})
        write_metrics(root, legacy_metrics)
        assert not fixed_reply_holds()
        legacy_metrics[0]["eval/all/response_tokens"] = tokens
        write_metrics(root, legacy_metrics)
        assert fixed_reply_holds()
        write_evaluations(export, primary)
    metrics = read_skyrl_metrics(str(root))
    step = snowball_model.SNOWBALL.by_name[preset].updates
    report = compare(preset, "--candidate", "users/test/current-generation")
    assert (report["verdict"], report["effect"]) == ("pass", "no change"), report
    if preset == "learning-medium":
        unscored = recorded_runs(preset, "users/test/no-eligible-evaluation", speed=101)
        export = str(unscored / "exports")
        assert not (Path(export) / "best_checkpoint.json").exists()
        with edit_receipt(unscored) as attempt:
            attempt["result"]["model"] = None
        evaluations = read_evaluations(export, attempt_id="recorded-attempt", evaluation="eval/")
        write_evaluations(
            export,
            [
                task.model_copy(
                    update={
                        "answers": [
                            answer.model_copy(update={"error_treatment": "passthrough"}) for answer in task.answers
                        ]
                    }
                )
                for task in evaluations
            ],
        )
        report = compare(preset, "--candidate", "users/test/no-eligible-evaluation")
        assert report["verdict"] == "fail", report
        assert any(row["name"] == "eval_lite/task_coverage" and not row["holds"] for row in report["invariants"])
    prefix = "eval/all/outcome/" if preset == "learning-medium" else "eval/eval_lite/all/outcome/"
    metrics[step][prefix + "graded"] = 19
    metrics[step][prefix + "unavailable"] = 1
    write_metrics(root, metrics)
    alias = root.parent.parent / "current-bad-counter" / root.name
    shutil.copytree(root, alias)
    report = compare(preset, "--candidate", "users/test/current-bad-counter")
    assert (report["verdict"], report["effect"]) == ("fail", "no change")
    assert any(row["name"] == "eval_lite/grader_error_fraction" and not row["holds"] for row in report["invariants"])


def test_training_evaluation_checks_held_out_grading_with_either_checkpoint_policy(recorded_runs, preset_table):
    preset = snowball_model.SNOWBALL.by_name["train-long"]
    evaluation = replace(preset.evaluations[0], every=2)
    for retention in ("latest", "best"):
        candidate = replace(
            preset.with_updates(2), checkpoints=retention, export="none", checkpoint_every=2, evaluations=(evaluation,)
        )
        preset_table(candidate)
        root = recorded_runs(candidate.name, f"users/test/checkpoint-grading-{retention}")
        facts = {fact.name for fact in measurements.read_measurements(str(root), candidate).invariants}
        assert {
            f"{evaluation.name}/{check}" for check in ("answers_complete", "task_coverage", "grader_error_fraction")
        } <= facts


def test_additional_evaluation_protocol_uses_its_recorded_request(recorded_runs):
    root = recorded_runs("speed-short", "users/test/recorded-named")
    launch = json.loads((root / "terminal.json").read_text())["config"]
    launch["skyrl"]["generator"]["eval_sampling_params"]["presence_penalty"] = 0.8
    launch["skyrl"]["generator"]["eval_n_samples_per_prompt"] = 7
    protocol = measurements.evaluation_protocol(
        RecordedLaunch.from_document(launch, attempt="recorded-attempt"), "eval_lite"
    )
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

    def write_panel(root, version, rows, *, parent=None):
        root.mkdir()
        for filename, population in (
            (FULL_VALIDATION_FILENAME, rows),
            ("validation.parquet", [row for row in rows if row.state == MembershipState.ACTIVE]),
        ):
            pq.write_table(
                pa.table({"task_spec": [json.dumps(specs[row.task_id]) for row in population]}), root / filename
            )
        pq.write_table(
            pa.Table.from_pylist([row.model_dump(mode="json") for row in rows], schema=arrow_schema(SelectionTaskRow)),
            root / TASK_MANIFEST_FILENAME,
        )
        definition = SelectionDefinition(name="permuted", version=version, sampling="all", parent=parent)
        _json(root / "summary.json", {"selection_strata": definition.model_dump(mode="json")})
        return read_selection_manifest(str(root))

    parent = write_panel(selection, "2026.11.17.1", manifest)
    launch["artifacts"]["export_root"] = str(tmp_path / "configured-export")
    source.update(uri=str(selection), relative_path=FULL_VALIDATION_FILENAME)
    write_receipt(tmp_path, native)
    evaluation = snowball_model.EVALUATIONS["eval_lite_100"]
    export, attempt_id = launch["artifacts"]["export_root"], launch["run"]["attempt_id"]
    task_rows = [
        row
        for step in (-1, 25)
        for row in evaluation_task_rows(
            answers, attempt_id=attempt_id, step=step, evaluation="eval/", selection=parent, normalized=normalized
        )
    ]
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
    current = measurements.read_answers(evaluation, str(tmp_path))
    statistics = scoring.score_answers(current.filter(pc.equal(current["step"], 25)))
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
        pa.array([row["scored"] and row["task_id"] != "stem" for row in current.to_pylist()]),
    ).filter(pc.equal(current["step"], 25))
    assert "eval_lite_100/aai_weighted_mean" not in scoring.score_answers(unavailable)

    specs["addition"] = {
        **base,
        "id": "addition",
        "context": {**base["context"], "events": [{**base["context"]["events"][0], "content": "Return a new answer."}]},
    }
    changed = {"bounded": {"state": MembershipState.RETIRED}, "mixed": {"difficulty": "easy"}}
    child_rows = [
        *(row.model_copy(update=changed.get(row.task_id, {})) for row in manifest),
        SelectionTaskRow(
            task_id="addition",
            task_digest=canonical_sha256(specs["addition"]),
            family="structural",
            aai_category="format",
            state=MembershipState.ACTIVE,
        ),
    ]
    child_root = tmp_path / "child"
    child = write_panel(
        child_root,
        "2026.11.17.2",
        child_rows,
        parent=SelectionParent(root=str(selection), name="permuted", version="2026.11.17.1"),
    )
    assert child.tasks["mixed"].strata.difficulty == "easy"
    version_answers = [_answer(item, float(item != "bounded")) for item in specs for _ in range(2)]
    complete = evaluation_task_rows(version_answers, attempt_id=attempt_id, step=25, evaluation="eval/", selection=child)
    write_evaluations(export, complete)
    stored = read_evaluations(export, attempt_id=attempt_id, evaluation="eval/", step=25)
    assert scoring.score_eval_lite_version(stored, parent, expected_answers=2) == scoring.EvalLiteVersionScore(
        5 / 6, 6, (), ()
    )
    assert scoring.score_eval_lite_version(stored, child, expected_answers=2) == scoring.EvalLiteVersionScore(
        1.0, 6, (), ()
    )

    repairs = {
        "coding": {"task_digest": "different-content"},
        "general": {"answers": []},
        "mixed": {"answers": next(row.answers[:1] for row in stored if row.task_id == "mixed")},
    }
    damaged = [row.model_copy(update=repairs.get(row.task_id, {})) for row in stored if row.task_id != "stem"]
    assert scoring.score_eval_lite_version(damaged, parent, expected_answers=2) == scoring.EvalLiteVersionScore(
        None, 2, ("general", "mixed", "stem"), ("coding",)
    )
    with pytest.raises(ValueError, match="select one evaluation, attempt and step"):
        scoring.score_eval_lite_version([*stored, stored[0].model_copy(update={"step": 0})], parent)

    old_launch = RecordedLaunch.from_document(launch, attempt=attempt_id)
    old_receipt = read_attempt(export, attempt_id)
    previous = measurements.read_answers(evaluation, str(tmp_path), launch=old_launch, attempt=old_receipt)
    active_attempt = "active-only"
    launch["run"]["attempt_id"] = native["result"]["attempt_id"] = active_attempt
    source["uri"] = str(child_root)
    write_receipt(tmp_path, native)
    write_evaluations(
        export,
        [
            row.model_copy(update={"attempt_id": active_attempt})
            for row in stored
            if child.tasks[row.task_id].state == MembershipState.ACTIVE
        ],
    )
    active = read_evaluations(export, attempt_id=active_attempt, evaluation="eval/", step=25)
    assert scoring.score_eval_lite_version(active, parent, expected_answers=2).missing == ("bounded",)
    with pytest.raises(measurements.InvalidComparison, match="differs from the selected attempt record"):
        measurements.read_answers(evaluation, str(tmp_path), launch=old_launch)
    assert measurements.read_answers(evaluation, str(tmp_path), launch=old_launch, attempt=old_receipt).equals(previous)


@pytest.mark.parametrize(
    "case",
    [
        "initializer",
        "initializer-hf",
        "priority",
        "stale-default",
        "control-commit",
        "unbound-data",
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
        "evaluation-cadence",
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
    inputs = RunInputs(preset_name=preset.name, settings=settings)
    if case.startswith("initializer"):
        name, version = "tests/custom-init", "2026.11.07"
        root = tmp_path / name / version
        if case == "initializer-hf":
            initializer = ArtifactHfModel(
                ArtifactStep.adopt(name, version, "s3://fixture-initializer", kind=Artifact),
                preset.model.tokenizer_uri,
                preset.model.tokenizer_revision,
            )
            fake_s3(monkeypatch, "fixture-initializer", "fixture-initializer/policy/tokenizer_config.json")
        else:
            write_skyrl_run_record(
                root,
                name,
                version,
                fingerprint="initializer",
                hf_model_uri="s3://fixture-initializer/policy",
                global_step=25,
                tokenizer_uri=preset.model.tokenizer_uri,
                tokenizer_revision=preset.model.tokenizer_revision,
                iris_job_id="/test-user/initializer",
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
            for index in range(pin.replicates)
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
    if case == "priority":
        override = replace(
            preset,
            recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, priority="production")),
        )
        inputs = inputs.model_copy(update={"options": RunOptions.from_preset(override)})
        gate.validate_candidate(override, inputs, table=snowball_model.SNOWBALL)
        table = snowball_model.SNOWBALL
        with monkeypatch.context() as altered:
            altered.setattr(snowball_model, "SNOWBALL", replace(table, presets=(override,)))
            candidate = recorded_runs(preset.name, name, inputs=inputs)
        assert json.loads((candidate / "terminal.json").read_text())["config"]["iris"]["priority"] == "production"
    else:
        candidate = recorded_runs(preset.name, name, inputs=inputs)
    omissions = {
        "omitted-default": {
            "config.skyrl.generator.chat_template_kwargs": {},
            "config.inputs.starting_checkpoint": None,
            "config.iris.role_topology": None,
        },
        "omitted-current-nondefault": {"config.skyrl.generator.chat_template_kwargs": {}},
        "omitted-schema-nondefault": {"config.skyrl.trainer.policy.optimizer_config.weight_decay": 0.0},
        "omitted-candidate-default": {"config.skyrl.generator.chat_template_kwargs": {}},
        "schema-timeout": {"config.skyrl.generator.chat_template_kwargs": {}},
    }.get(case, {})
    if omissions:
        omitted = [candidate] if case == "omitted-candidate-default" else control_roots
        for old in omitted:
            with edit_receipt(old) as receipt:
                for path, original in omissions.items():
                    parent, key = path.rsplit(".", 1)
                    assert get_path(receipt, parent).pop(key) == original
    if case == "schema-timeout":
        cache_native_compositions(monkeypatch)
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
                report = compare(preset.name, "--candidate", name)
                assert (
                    report["verdict"] == "invalid" and "native configuration cannot be composed" in report["reason"]
                ), report
                if diagnostic == "timeout":
                    assert "timed out after 120 s" in report["reason"]
        return
    reasons = {
        "stale-default": ("control config differs",),
        "zero-spread": ("variation is zero",),
        "missing-guard": ("missing or non-finite guard metric",),
        "omitted-current-nondefault": ("control config differs",),
        "omitted-schema-nondefault": ("control config differs",),
        "omitted-candidate-default": ("skyrl.generator.chat_template_kwargs",),
    }.get(case, ())
    mutation = {
        "control-commit": (control_roots[0], "result.provenance.base_commit", "wrong-commit", "does not match its pin"),
        "unbound-data": (candidate, "config.skyrl.data.train_data", ["/unbound/train.parquet"], "skyrl.data.train_data"),
        "allocation": (
            candidate,
            "config.iris.allocation.num_nodes",
            preset.recipe.layout.workload.plan.num_nodes + 1,
            "iris.allocation.num_nodes",
        ),
        "protocol-temperature": (
            candidate,
            "config.skyrl.generator.eval_sampling_params.temperature",
            0.5,
            "evaluation protocol",
        ),
        "malformed-seed": (candidate, "config.skyrl.trainer.seed", None, "malformed recorded configuration", "seed"),
    }.get(case)
    if mutation is not None:
        root, path, value, *reasons = mutation
        with edit_receipt(root) as receipt:
            set_path(receipt, path, value)
    if case == "missing-guard":
        edit_metrics(candidate, lambda row: row.update({"policy/mismatch/staleness0/selected_tokens": 0}))
    if case == "evaluation-cadence":
        with edit_receipt(candidate) as receipt:
            evaluation = next(
                callback
                for callback in receipt["config"]["skyrl"]["trainer"]["callbacks"]
                if callback["type"] == "evaluation"
            )
            evaluation["eval_steps"] = 17
        reasons = ("eval_steps",)
    report = compare(preset.name, "--candidate", name)
    if reasons:
        assert report["verdict"] == "invalid" and all(reason in report["reason"] for reason in reasons), report
    else:
        assert (report["verdict"], report["effect"]) == ("pass", "no change"), report
        if case.startswith("initializer"):
            assert "inputs.model.uri" in report["run"]["config_changes"]
    if case == "cached-default":
        promoted = replace(
            preset,
            recipe=replace(
                preset.recipe,
                config=preset.recipe.config.with_settings(("trainer.policy.optimizer_config.weight_decay=0.1",)),
            ),
        )
        preset_table(promoted)
        updated = compare(preset.name, "--candidate", name)
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
    task_set = measurements.read_recorded_task_set(
        RecordedLaunch.from_document(receipt["config"], attempt=receipt["config"]["run"]["attempt_id"]), "eval_lite_100"
    )
    assert set(task_set.task_ids) == {json.loads(row["task_spec"])["id"] for row in rows}
    assert inactive["id"] not in task_set.task_ids
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), original)
    report = compare(preset, "--candidate", "users/test/task-content")
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
    report = compare(preset, "--candidate", "users/test/task-content-edited")
    assert report["verdict"] == "invalid" and "TaskSpec content differs" in report["reason"], report


@pytest.mark.parametrize(
    "case, replaced, extra, shared, reason",
    (
        ("different-content", (19,), 0, 19, None),
        ("additional-tasks", (), 1, 20, None),
        ("relabeled-family", (), 0, 20, None),
        ("too-few-common", range(10, 20), 0, 10, "10 of 20 control tasks shared"),
        ("missing-family", (0,), 0, 19, "family/small: 0 of 1 control tasks shared"),
    ),
)
def test_control_link_scores_shared_evaluation(
    recorded_runs, preset_table, monkeypatch, tmp_path, case, replaced, extra, shared, reason
):
    cache_native_compositions(monkeypatch)
    preset = replace(
        snowball_model.SNOWBALL.by_name["learning-medium"], updates=2, measured=range(1, 3), checkpoint_every=3
    )
    table = preset_table(preset)
    previous = replace(gate.control_pin(preset, table=table), replicates=2)
    table = replace(table, control_pins={**table.control_pins, preset.name: previous})
    monkeypatch.setattr(snowball_model, "SNOWBALL", table)
    baseline_roots = controls(recorded_runs, preset.name)
    if case in ("different-content", "relabeled-family"):
        for control, score in zip(baseline_roots, (0.1, 0.9), strict=True):
            export = str(control / "exports")
            rows = read_evaluations(export, attempt_id="recorded-attempt", evaluation="eval/")
            selected = "task-19" if case == "different-content" else "task-0"
            write_evaluations(
                export,
                [
                    row.model_copy(
                        update={
                            "answers": [answer.model_copy(update={"verifier_score": score}) for answer in row.answers],
                        }
                    )
                    for row in rows
                    if row.task_id == selected
                ],
            )
    old_binding = preset.held_out_tasks
    assert old_binding is not None
    new_binding = old_binding.split("@")[0] + "@2026.11.24.1"
    old_panel = Path(recipe.selection_artifact(old_binding).path())
    panel = Path(recipe.selection_artifact(new_binding).path())
    panel.mkdir(parents=True)
    old_tasks = pq.read_table(old_panel / FULL_VALIDATION_FILENAME).to_pylist()
    count = len(old_tasks)
    assert count == 20
    annotations = pq.read_table(old_panel / TASK_MANIFEST_FILENAME).to_pylist()[:count]
    if case == "relabeled-family":
        annotations[0]["family"] = "large"
    for index in replaced:
        task = json.loads(old_tasks[index]["task_spec"])
        task["context"]["events"][0]["content"] = f"Updated question {index}"
        old_tasks[index]["task_spec"] = json.dumps(task)
        annotations[index]["task_digest"] = canonical_sha256(task)
    for index in range(count, count + extra):
        task = json.loads(old_tasks[-1]["task_spec"])
        task["id"] = f"task-{index}"
        old_tasks.append({"task_spec": json.dumps(task)})
        annotations.append({**annotations[-1], "task_id": task["id"], "task_digest": canonical_sha256(task)})
    for name in (VALIDATION_FILENAME, FULL_VALIDATION_FILENAME):
        pq.write_table(pa.Table.from_pylist(old_tasks), panel / name)
    pq.write_table(
        pa.Table.from_pylist(annotations, schema=arrow_schema(SelectionTaskRow)), panel / TASK_MANIFEST_FILENAME
    )
    summary = json.loads((old_panel / "summary.json").read_text())
    summary.update(validation_rows=count + extra, full_validation_rows=count + extra)
    summary["selection_strata"]["version"] = "2026.11.24.1"
    _json(panel / "summary.json", summary)

    fresh = replace(previous, commit="next-main", revision="2026.11.24.1")
    preset = replace(preset, revision=fresh.revision, held_out_tasks=new_binding)
    table = preset_table(preset)
    table = replace(table, control_pins={**table.control_pins, preset.name: fresh})
    monkeypatch.setattr(snowball_model, "SNOWBALL", table)
    changed_ids = {f"task-{index}" for index in (*replaced, *range(count, count + extra))}
    fresh_roots = []
    for index in range(fresh.replicates):
        candidate = recorded_runs(preset.name, gate.control_name(preset.name, fresh.commit, index, table=table))
        fresh_roots.append(candidate)
        if changed_ids and reason is None:
            export = str(candidate / "exports")
            rows = read_evaluations(export, attempt_id="recorded-attempt", evaluation="eval/", step=preset.updates)
            write_evaluations(
                export,
                [
                    row.model_copy(
                        update={"answers": [answer.model_copy(update={"verifier_score": 1.0}) for answer in row.answers]}
                    )
                    for row in rows
                    if row.task_id in changed_ids
                ],
            )
        elif case == "relabeled-family":
            with edit_receipt(candidate) as receipt:
                receipt["result"].update(started_at="2026-11-06T18:41:00+00:00", finished_at="2026-11-06T19:41:00+00:00")
    [link] = run(gate.control_link_step(preset, fresh, previous, table=table))
    report = link.model_dump(mode="json")
    facts = {fact["name"]: fact for fact in report["invariants"]}
    assert facts[f"eval_lite/{SHARED_TASKS_INVARIANT}"]["value"] == shared
    assert facts[f"eval_lite/{SHARED_TASK_FRACTION_INVARIANT}"]["value"] == shared / count
    if reason is not None:
        assert report["verdict"] == "invalid" and reason in report["reason"], report
        return
    assert (report["verdict"], report["effect"]) == ("pass", "no change"), report
    assert report["control"]["held_out_tasks"] == old_binding
    assert report["run"]["held_out_tasks"] == new_binding
    results = comparisons(report)
    family_mean = results["eval_lite/family_mean"]
    assert family_mean["tasks"] == shared and family_mean["run"] == pytest.approx(0.5)
    assert family_mean["control"] == pytest.approx(0.49525 if case == "relabeled-family" else 0.495)
    if case == "different-content":
        assert family_mean["sd"] == pytest.approx(0.01 / math.sqrt(2))
        gate_report = gate._write_gate(
            preset,
            (str(fresh_roots[0]),),
            tuple(map(str, baseline_roots)),
            replace(previous, revision=fresh.revision),
            RunInputs(preset_name=preset.name),
            str(tmp_path / "candidate-gate"),
        )
        assert gate_report.verdict == "invalid"
        assert gate_report.reason == "evaluation protocol or held-out TaskSpec content differs"

        with pytest.raises(ValueError, match="distinct cohorts"):
            gate.control_link_step(preset, fresh, fresh, table=table)
        with pytest.raises(ValueError, match="current control pin must match"):
            gate.control_link_step(preset, previous, fresh, table=table)

        def rejected_link(expected_reason):
            report = gate._write_control_link(
                preset,
                fresh,
                previous,
                tuple(map(str, fresh_roots)),
                tuple(map(str, baseline_roots)),
                str(tmp_path / "incompatible-link"),
            )
            assert report.verdict == "invalid" and report.reason == expected_reason

        baseline = baseline_roots[1]
        with edit_receipt(baseline) as receipt:
            native = deepcopy(receipt["config"])
        validation = native["inputs"]["validation_data"]
        mutations = (
            ("inputs.train_data", [], "recorded control workload differs from the link preset"),
            ("inputs.evaluation_data", {"retired_evaluation": validation}, EVALUATION_PROTOCOL_DIFFERS),
            (
                "inputs.validation_data",
                [{**validation[0], "identity": "selection/other-panel@2026.11.24.2:fixture"}],
                "control cohort contains different held-out selections or TaskSpec sets",
            ),
        )
        for key, value, expected_reason in mutations:
            with edit_receipt(baseline) as receipt:
                receipt["config"] = deepcopy(native)
                set_path(receipt["config"], key, value)
            rejected_link(expected_reason)
        with edit_receipt(baseline) as receipt:
            receipt["config"] = native

        with edit_receipt(fresh_roots[0]) as receipt:
            receipt["config"]["skyrl"]["generator"]["eval_sampling_params"]["temperature"] = 0.5
        rejected_link(EVALUATION_PROTOCOL_DIFFERS)
    if changed_ids:
        unshared = results["eval_lite/unshared/mean"]
        assert (unshared["role"], unshared["tasks"], unshared["run"]) == ("reported", len(changed_ids), 1.0)
    else:
        assert "eval_lite/unshared/mean" not in results
        assert "eval_lite/family/small/mean" not in results
        assert results["eval_lite/family/large/mean"]["tasks"] == 20
        assert results["eval_lite/family/large/mean/initial"]["control"] == pytest.approx(0.31)
        assert results["eval_lite/family/large/mean/gain"]["control"] == pytest.approx(0.18525)
    gain = "eval_lite/family_mean/gain_per_gpu_hour"
    if extra:
        assert results[gain]["run"] is results[gain]["control"] is None
    else:
        assert results[gain]["run"] == pytest.approx(0.2 / 40)
        expected_control_gain = 0.18525 if case == "relabeled-family" else 0.195
        assert results[gain]["control"] == pytest.approx(expected_control_gain / 40)


@pytest.mark.parametrize("preset_name", ["learning-medium", "data-medium"])
def test_python_run_saves_typed_inputs_for_comparison(
    preset_table, recorded_runs, selection, tmp_path, monkeypatch, preset_name
):
    preset = snowball_model.SNOWBALL.by_name[preset_name]
    gain_key = "eval_lite/family_mean/gain_per_gpu_hour"
    model = tmp_path / "model"
    model.mkdir()
    model_uri = f"s3://marin-test/models/{tmp_path.name}"

    fake_s3_checkpoint(monkeypatch, model_uri, model)
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
    settings = (
        (
            "trainer.policy.optimizer_config.lr=2.0e-6",
            "trainer.train_batch_size=64",
            "trainer.policy_mini_batch_size=64",
            "trainer.micro_train_batch_size_per_gpu=2",
            "generator.n_samples_per_prompt=8",
            'extra_env.VLLM_PROBE="1"',
            "generator.max_num_seqs=1024",
        )
        if tasks is None
        else ()
    )
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
    built = snowball.build_experiment(
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
    (staged / RUN_INPUTS_FILENAME).unlink()

    def native(config, *, forward_env=()):
        shutil.copytree(staged, source, dirs_exist_ok=True)
        return SkyRLRun.raw_load(str(source))

    monkeypatch.setattr(snowball, "run_skyrl", native)
    monkeypatch.setattr(sys, "argv", ["unrelated-command", "--preset", "train-long", "--require", "unrelated"])
    run(*built.runs)
    recorded = read_run_inputs(str(source))
    assert recorded.preset_name == preset_name and recorded.requirements == ("policy/raw_grad_norm<=10",)
    decoded = dict(item.split("=", 1) for item in recorded.settings)
    assert {key: json.loads(value) for key, value in decoded.items()} == (
        {}
        if tasks
        else {
            "trainer": {
                "policy": {"optimizer_config": {"lr": 2e-6}},
                "train_batch_size": 64,
                "policy_mini_batch_size": 64,
                "micro_train_batch_size_per_gpu": 2,
            },
            "extra_env": {"VLLM_PROBE": "1"},
            "generator": {"max_num_seqs": 1024, "n_samples_per_prompt": 8},
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
    report = compare(preset_name, "--candidate", built.runs[0].name)
    assert report["verdict"] == "pass", report
    assert report["run"]["requirements"] == list(recorded.requirements)
    assert report["run"]["settings"] == list(recorded.settings)
    assert report["run"]["tasks"] == recorded.tasks
    assert report["run"]["attempt_ids"] == ["recorded-attempt"]
    assert comparisons(report)[gain_key]["run"] == pytest.approx(0.005)
    _json(source / "exports/best_checkpoint.json", {"step": preset.updates})
    [status] = snowball._status(built, preset, stored_only=True)["runs"]
    assert status["best_checkpoint_step"] == preset.updates
    if tasks is None:
        assert {
            "inputs.model.uri",
            "skyrl.trainer.policy.optimizer_config.lr",
            "skyrl.generator.max_num_seqs",
            "skyrl.trainer.train_batch_size",
            "skyrl.trainer.micro_forward_batch_size_per_gpu",
            "skyrl.trainer.rollout_buffer.max_in_flight",
        } <= set(report["run"]["config_changes"])
        trainer = read_attempt(str(source / "exports"), "recorded-attempt").native.config["skyrl"]["trainer"]
        assert (trainer["micro_forward_batch_size_per_gpu"], trainer["rollout_buffer"]["max_in_flight"]) == (2, 64)
