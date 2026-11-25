# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from finestore.rl.rl_progress import ComparisonRole
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
from marin.execution.lazy import ArtifactStep, run
from marin.rl.skyrl import SkyRLRun
from marin.rl.skyrl_records import RecordedLaunch, read_skyrl_metrics
from scipy.stats import t
from skyrlrecipe.model_staging import DEFAULT_TOKENIZER_REVISION, model_metadata_relative_path
from taskcompendium.grading_result import Outcome
from taskcompendium.identity import canonical_sha256

from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_measurements as measurements
from experiments.post_training import rl_scoring as scoring
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_measurements import RunInputs
from experiments.post_training.rl_tests.fixtures import (
    KL,
    RL_TEST_MARKS,
    compare,
    controls,
    edit_receipt,
    evaluation_task_rows,
    write_metrics,
    write_receipt,
)
from experiments.post_training.rl_tests.fixtures import answer as _answer
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    TASK_MANIFEST_FILENAME,
)

pytestmark = RL_TEST_MARKS


def test_compare_validates_staged_model_and_tokenizer_bindings(recorded_runs):
    preset = "speed-short"
    control_roots = controls(recorded_runs, preset)
    candidate = recorded_runs(preset, "users/test/staged-metadata", speed=102)
    wrong_tokenizer = recorded_runs(preset, "users/test/wrong-staged-tokenizer", speed=102)
    for root in (*control_roots, candidate, wrong_tokenizer):
        with edit_receipt(root) as receipt:
            launch = receipt["config"]
            model = launch["inputs"]["model"]
            staging_root = Path("/tmp/skyrl-worker")
            model_path = str(staging_root / model_metadata_relative_path(model["uri"], model["identity"]))
            tokenizer_revision = (
                "different" if root == wrong_tokenizer else model["tokenizer_revision"] or DEFAULT_TOKENIZER_REVISION
            )
            tokenizer_path = str(staging_root / model_metadata_relative_path(model["tokenizer_uri"], tokenizer_revision))
            for role in ("policy", "ref"):
                launch["skyrl"]["trainer"][role]["model"].update(
                    path=model_path, tokenizer_path=tokenizer_path, tokenizer_revision=None
                )

    assert compare(preset, "--candidate", "users/test/staged-metadata")["verdict"] == "pass"
    invalid = compare(preset, "--candidate", "users/test/wrong-staged-tokenizer")
    assert invalid["verdict"] == "invalid" and "skyrl.trainer.policy.model.tokenizer_path" in invalid["reason"]


def test_compare_checks_hf_draft_mirror_against_the_requested_revision(recorded_runs):
    preset = "speed-short"
    control_roots = controls(recorded_runs, preset)
    candidate = recorded_runs(preset, "users/test/mirrored-draft", speed=102)
    wrong_mirror = recorded_runs(preset, "users/test/wrong-draft-mirror", speed=102)
    for root in (*control_roots, candidate, wrong_mirror):
        with edit_receipt(root) as receipt:
            launch = receipt["config"]
            draft = launch["skyrl"]["generator"]["speculative_decoding"]["model"]
            commit = draft["source_identity"]
            uri, identity = "s3://test-bucket/draft-mirror", "sha256:" + "f" * 64
            launch["inputs"]["hugging_face_models"] = [
                {
                    "model_id": draft["source_uri"].removeprefix("hf://"),
                    "revision": commit,
                    "resolved_revision": "a" * 40 if root == wrong_mirror else commit,
                    "uri": uri,
                    "identity": identity,
                }
            ]
            draft.update(source_uri=uri, source_identity=identity)

    assert compare(preset, "--candidate", "users/test/mirrored-draft")["verdict"] == "pass"
    invalid = compare(preset, "--candidate", "users/test/wrong-draft-mirror")
    assert invalid["verdict"] == "invalid" and "skyrl.generator.speculative_decoding.model" in invalid["reason"]


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
