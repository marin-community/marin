# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import fcntl
import json
import math
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import monotonic

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from finestore.rl.rl_progress import (
    ComparisonRole,
)
from finestore.rl.run_records import (
    read_evaluations,
    write_evaluations,
)
from finestore.rl.strata import (
    MembershipState,
    SelectionDefinition,
    SelectionTaskRow,
    read_selection_manifest,
)
from finestore.schema import arrow_schema
from marin.execution.lazy import ArtifactStep, run
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel, SkyRLRun, run_skyrl_cpu
from marin.rl.skyrl_records import RecordedLaunch, read_stored_skyrl_attempts
from scipy.stats import t
from skyrlrecipe.model_staging import DEFAULT_TOKENIZER_REVISION, model_metadata_relative_path
from taskcompendium.identity import canonical_sha256

from experiments.post_training import rl as snowball
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_index
from experiments.post_training import rl_measurements as measurements
from experiments.post_training import rl_scoring as scoring
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_command import EXIT_CODES
from experiments.post_training.rl_measurements import (
    RUN_INPUTS_FILENAME,
    RunInputs,
    read_run_inputs,
    run_inputs,
)
from experiments.post_training.rl_tests.fixtures import (
    SELECTION_IDENTITY,
    cohort_from_fixtures,
    controls,
    edit_receipt,
    evaluation_task_rows,
    fake_s3_checkpoint,
    write_hf_export,
    write_metrics,
    write_receipt,
)
from experiments.post_training.rl_tests.fixtures import answer as _answer
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.skyrl_test_support import read_skyrl_metrics
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    TASK_MANIFEST_FILENAME,
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


def test_compare_writes_the_report_for_the_five_newest_base_version_controls(
    recorded_runs, cache_native_compositions, monkeypatch
):
    preset = "speed-medium"
    for index, minute in enumerate((5, 2, 4, 3, 6, 1)):
        recorded_runs(
            preset,
            f"controls/{preset}/r{index}",
            control=True,
            attempt_id=f"control-r{index}",
            cohort_run_id="nightly-batch",
            completed_at=f"2026-11-06T19:{minute:02d}:00+00:00",
            speed=100 + index,
        )
    inputs = RunInputs(preset_name=preset, settings=("trainer.policy.optimizer_config.lr=2e-6",))
    candidate = recorded_runs(preset, "users/test/baseline", speed=102, inputs=inputs)
    index = read_stored_skyrl_attempts(str(candidate))[1]["recorded-attempt"].index
    assert index.base_preset_version != index.preset_version
    indexed = rl_index.list_attempts()
    different = indexed[0].model_copy(
        update={
            "native_run_id": "wrong",
            "attempt_id": "wrong",
            "preset_version": index.preset_version,
            "completed_at": "2026-11-06T20:00:00+00:00",
        }
    )

    monkeypatch.setattr(rl_index, "list_attempts", lambda _service=None: [*indexed, different])
    report = compare(preset, "--candidate", "users/test/baseline")
    assert (report["verdict"], report["effect"]) == ("pass", "no change"), report
    assert report["run"]["attempt_ids"] == ["recorded-attempt"]
    assert report["control"]["version"] == index.base_preset_version
    members = report["control"]["attempts"]
    assert [row["attempt_id"] for row in members] == [f"control-r{i}" for i in (4, 0, 2, 3, 1)]
    assert {row["run_id"] for row in members} == {"nightly-batch"}
    artifact = Path(report.pop("report"))
    assert json.loads(artifact.read_text()) == report and artifact.with_suffix(".md").exists()
    assert {row["role"] for row in report["comparisons"]} >= {"objective", "guard", "reported"}


def test_extra_requirements_cannot_widen_existing_guards(recorded_runs):
    preset = snowball_model.SNOWBALL.by_name["speed-medium"]
    key = "policy/raw_grad_norm"
    for index, root in enumerate(controls(recorded_runs, preset.name)):
        edit_metrics(root, lambda row, i=index: row.update({key: 100 + i}))
    names = ("users/test/requirements-r0", "users/test/requirements-r1")
    roots = [
        recorded_runs(preset.name, name, speed=108, metrics={key: value})
        for name, value in zip(names, (0, 0.01), strict=True)
    ]
    candidates = tuple(
        ArtifactStep.adopt(name, recipe.RUN_ARTIFACT_VERSION, str(root), kind=SkyRLRun)
        for name, root in zip(names, roots, strict=True)
    )
    inputs = run_inputs(
        preset.name,
        preset=preset,
        requirements=(f"{key}<=control", f"{key}<=control", f"{key}<=0", "speed/update_seconds<=control"),
    )

    def gate_for_candidates():
        return gate.gate_step(
            preset,
            candidates,
            inputs=inputs,
            cohort=cohort_from_fixtures(roots[0]),
            group_name="users/test/requirements",
        )

    def required_outcomes(report):
        return [fact.holds for fact in report.invariants if fact.name == "require/" + key]

    report = run(gate_for_candidates())[0]
    assert report.verdict == "fail" and report.effect == "regression"
    speed_guard = next(row for row in report.comparisons if row.metric == "speed/update_seconds")
    assert speed_guard.passes is False
    assert speed_guard.tolerance == pytest.approx(t.ppf(1 - 0.025 / 2, 4) * math.sqrt(2.5) * math.sqrt(0.7))
    requirement = report.rows(ComparisonRole.REQUIREMENT)
    assert len(requirement) == 1 and requirement[0].passes
    assert requirement[0].tolerance == pytest.approx(t.ppf(0.975, 4) * math.sqrt(2.5) * math.sqrt(0.7))
    assert required_outcomes(report) == [True, False]
    edit_metrics(roots[1], lambda row: row.update({key: 0.0}))
    updated = run(gate_for_candidates())[0]
    assert updated.path != report.path and required_outcomes(updated) == [True, True]

    other = cohort_from_fixtures(roots[0]).attempts[1]
    with edit_receipt(Path(other.path)) as receipt:
        receipt["config"]["skyrl"]["trainer"]["policy"]["optimizer_config"]["lr"] *= 2
    stale = run(gate_for_candidates())[0]
    assert stale.path != updated.path
    assert stale.verdict == "invalid" and "stored control config version" in stale.reason


def test_measurements_use_selected_tokens_and_check_training_health(recorded_runs, preset_table, subtests):
    preset = replace(snowball_model.SNOWBALL.by_name["speed-long"], updates=3, measured=range(2, 4))
    preset_table(preset)
    root = recorded_runs(preset.name, "users/test/training-measurements")
    metrics = read_skyrl_metrics(str(root))
    prefix = KL.rsplit("/", 1)[0]
    for step, admission, kl, count in ((1, 1e9, 1e9, 1000), (2, 5, 0.5, 1), (3, 9, 0, 9)):
        metrics[step].update(
            {
                "timing/step_wall/group_admission": admission,
                "timing/step_wall/policy_training": 2,
                "timing/step_wall/checkpoint_work": 900,
                "timing/step_wall/evaluation": 700,
                KL: kl,
                prefix + "/selected_tokens": count,
                prefix + "/finite_fraction": 1,
                "policy/mismatch/pooled/kl_k3": 0,
            }
        )
    for label, field, value, kl, bad in (
        ("weighted", None, None, 0.05, None),
        ("empty", prefix + "/selected_tokens", 0, None, None),
        ("nonfinite-kl", prefix + "/finite_fraction", 0, None, None),
        ("no-pooling", KL, 0.2, 0.2, None),
        ("loss", "loss/total", float("nan"), 0.05, "finite_losses_and_gradients"),
        ("policy-loss", "policy/policy_loss", float("nan"), 0.05, "finite_losses_and_gradients"),
        ("unavailable-and-overlap", "outcome-union", None, 0.05, "grader_error_fraction"),
        ("missing-counter", "missing-graded", None, 0.05, "grader_error_fraction"),
        ("updates", "missing-step", None, 0.5, "planned_updates"),
    ):
        with subtests.test(label=label):
            measured = deepcopy(metrics)
            if field == "missing-step":
                measured.pop(3)
            elif field == "outcome-union":
                for row in measured.values():
                    row.update(
                        {
                            "generate/outcome/graded": 99,
                            "generate/outcome/unavailable": 1,
                            "generate/outcome/rollout_failure": 1,
                        }
                    )
            elif field == "missing-graded":
                measured[3].pop("generate/outcome/graded")
            elif field:
                for step in (2, 3):
                    measured[step][field] = value
            write_metrics(root, measured)
            result = measurements.read_measurements(str(root), preset)
            if bad:
                assert not next(fact.holds for fact in result.invariants if fact.name == bad)
            else:
                assert result.values["speed/update_seconds"] == 9
                assert result.values.get(KL) == (None if kl is None else pytest.approx(kl))


def _assert_speed_short_measurements(root, preset):
    export = str(root / "exports")
    rows = read_evaluations(export, attempt_id="recorded-attempt", evaluation="eval/")

    def facts():
        return {fact.name: fact.holds for fact in measurements.read_measurements(str(root), preset).invariants}

    def changed(row, **fields):
        return row.model_copy(update={"answers": [answer.model_copy(update=fields) for answer in row.answers]})

    held_out = read_evaluations(export, attempt_id="recorded-attempt", evaluation="eval/eval_lite/")[0]
    write_evaluations(export, [changed(held_out, error_treatment="passthrough")])
    with edit_receipt(root) as receipt:
        launch = receipt["config"]
        source = launch["inputs"]["evaluation_data"]["eval_lite"][0]
        for callback in launch["skyrl"]["trainer"]["callbacks"]:
            if named := callback.get("additional_evaluations", {}).get("eval_lite"):
                named["data_files"] = [str(Path(source["local_path"]) / source["relative_path"])]
    launch["skyrl"]["generator"]["eval_n_samples_per_prompt"] = 7
    protocol = measurements.evaluation_protocol(
        RecordedLaunch.from_document(launch, attempt="recorded-attempt"), "eval_lite"
    )
    assert (protocol["n_samples_per_prompt"], protocol["sampling_params"]["max_generate_length"]) == (1, 8192)
    assert not facts()["eval_lite/task_coverage"] and facts()["eval_lite/grader_error_fraction"]
    write_evaluations(export, [changed(rows[0], response_token_count=4095)])
    assert not facts()["fixed_reply_tokens"]
    write_evaluations(export, [changed(row, response_token_count=None) for row in rows])
    metrics = read_skyrl_metrics(str(root))
    tokens = 4096 * sum(len(row.answers) for row in rows)
    for delta in (-1, 0):
        metrics[0].update({"eval/all/response_tokens_max": 4096, "eval/all/response_tokens": tokens + delta})
        write_metrics(root, metrics)
        assert facts()["fixed_reply_tokens"] == (delta == 0)
    write_evaluations(export, rows)


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

    retired = {**base, "id": "retained-inactive"}
    manifest.append(
        SelectionTaskRow(task_id=retired["id"], task_digest=canonical_sha256(retired), state=MembershipState.RETIRED)
    )
    selection.mkdir()
    validation = pa.table({"task_spec": [json.dumps(specs[key]) for key in families]})
    for filename in (FULL_VALIDATION_FILENAME, "validation.parquet"):
        pq.write_table(validation, selection / filename)
    pq.write_table(
        pa.Table.from_pylist([row.model_dump(mode="json") for row in manifest], schema=arrow_schema(SelectionTaskRow)),
        selection / TASK_MANIFEST_FILENAME,
    )
    pq.write_table(
        pa.table({"task_spec": [json.dumps(task) for task in (*specs.values(), retired)]}), selection / "pool.parquet"
    )
    definition = SelectionDefinition(
        name="permuted", version="2026.11.17.1", sampling="all", task_files=("pool.parquet",)
    )
    _json(selection / "summary.json", {"selection_strata": definition.model_dump(mode="json")})
    strata = read_selection_manifest(str(selection))
    launch["artifacts"]["export_root"] = str(tmp_path / "configured-export")
    source.update(uri=str(selection), relative_path=FULL_VALIDATION_FILENAME)
    write_receipt(tmp_path, native)
    evaluation = snowball_model.EVALUATIONS["eval_lite_100"]
    export, attempt_id = launch["artifacts"]["export_root"], launch["run"]["attempt_id"]
    task_rows = [
        row
        for step in (-1, 25)
        for row in evaluation_task_rows(
            answers, attempt_id=attempt_id, step=step, evaluation="eval/", selection=strata, normalized=normalized
        )
    ]
    write_evaluations(
        export,
        [
            *task_rows,
            *(
                row.model_copy(
                    update={
                        "attempt_id": "other-attempt",
                        "answers": [answer.model_copy(update={"verifier_score": 0.0}) for answer in row.answers],
                    }
                )
                for row in task_rows
            ),
        ],
    )
    current = measurements.read_answers(evaluation, str(tmp_path))
    assert set(current["task_id"].to_pylist()) == set(specs)
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


@pytest.mark.timeout(15)
@pytest.mark.parametrize("exit_early", [False, True], ids=["timeout", "nonzero-exit"])
def test_native_failure_stops_build_child(tmp_path, exit_early):
    lock = tmp_path / "build.lock"
    lock.touch()
    worker = Path(__file__).with_name("native_build_child.py")
    deadline = monotonic() + 6
    command = [sys.executable, "-I", str(worker), "exit" if exit_early else "wait", str(lock)]
    if exit_early:
        result = run_skyrl_cpu(command, timeout=2, stream_stderr=True)
        assert result.returncode == 7
        assert "build child ready" in result.stderr
    else:
        with pytest.raises(subprocess.TimeoutExpired, match="timed out after 2") as error:
            run_skyrl_cpu(command, timeout=2)
        assert b"build child ready" in error.value.stderr
    with lock.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
    assert monotonic() < deadline, "native failure waited for the build child to finish"


@pytest.mark.parametrize(
    "case, expected",
    [
        ("redacted-timeout", "[REDACTED]"),
        ("invalid-json", "returned invalid JSON"),
        ("wrong-shape", "expected 1 configuration objects"),
        ("prepare-timeout", "Prepare SkyRL CPU environment"),
    ],
)
def test_native_composition_reports_failure(monkeypatch, case, expected):
    def prepare(timeout, **_):
        if case == "prepare-timeout":
            raise subprocess.TimeoutExpired("prepare", timeout)
        return sys.executable

    def compose(command, *, timeout):
        Path(command[command.index("--progress-file") + 1]).write_text("Compose document 1/1")
        if case == "redacted-timeout":
            raise subprocess.TimeoutExpired("compose", timeout, stderr="sk\x1b-fixture-abcdefghijklmnopqrstuvwxyz")
        return subprocess.CompletedProcess([], 0, "{" if case == "invalid-json" else "[null]", "")

    monkeypatch.setattr(gate, "prepare_skyrl_cpu_python", prepare)
    monkeypatch.setattr(gate, "run_skyrl_cpu", compose)
    with pytest.raises(gate.InvalidComparison, match=re.escape(expected)) as failure:
        gate.compose_configs(["launch.yaml"])
    assert "fixture-abcdefghijklmnopqrstuvwxyz" not in str(failure.value)


@pytest.mark.timeout(15)
def test_streaming_launcher_can_cancel_its_job_on_interrupt(tmp_path):
    marker = tmp_path / "cancellation"
    worker = Path(__file__).with_name("native_build_child.py")
    with pytest.raises(KeyboardInterrupt):
        run_skyrl_cpu([sys.executable, "-I", str(worker), "interrupt", str(marker)], stream_stderr=True)
    assert marker.read_text() == "cancelled"


@pytest.mark.parametrize("preset_name", ["speed-short", "learning-medium", "speed-medium"])
def test_compare_classifies_candidate_variants(
    recorded_runs,
    preset_table,
    monkeypatch,
    tmp_path,
    cache_native_compositions,
    subtests,
    preset_name,
):
    preset = snowball_model.SNOWBALL.by_name[preset_name]
    members = controls(recorded_runs, preset_name)

    def candidate(name, *, verdict="pass", effect="no change", reason="", **options):
        path = recorded_runs(preset_name, f"users/test/{name}", **options)
        assert_verdict(name, verdict, effect, reason)
        return path

    def assert_verdict(name, verdict="pass", effect="no change", reason=""):
        report = compare(preset_name, "--candidate", f"users/test/{name}")
        assert report["verdict"] == verdict, report
        if reason:
            assert reason in report["reason"], report
        else:
            assert report["effect"] == effect, report
        return report

    if preset_name == "speed-medium":
        candidate(
            "missing-guard",
            verdict="invalid",
            reason="missing or non-finite guard metric",
            metrics={"policy/mismatch/staleness0/selected_tokens": 0},
        )
        return
    root = candidate("current-generation")
    if preset_name == "speed-short":
        _assert_speed_short_measurements(root, preset)
    prefix = "eval/all/outcome/" if preset_name == "learning-medium" else "eval/eval_lite/all/outcome/"
    metrics = read_skyrl_metrics(str(root))
    metrics[preset.updates].update({prefix + "graded": 19, prefix + "unavailable": 1})
    write_metrics(root, metrics)
    report = assert_verdict("current-generation", "fail")
    assert any(row["name"] == "eval_lite/grader_error_fraction" and not row["holds"] for row in report["invariants"])
    if preset_name == "speed-short":
        _assert_staged_model_and_draft(recorded_runs, members, assert_verdict)
        return

    edits = (
        ("unbound-data", "config.skyrl.data.train_data", ["/unbound/train.parquet"], "skyrl.data.train_data"),
        (
            "allocation",
            "config.iris.allocation.num_nodes",
            preset.recipe.layout.workload.plan.num_nodes + 1,
            "iris.allocation.num_nodes",
        ),
        ("protocol", "config.skyrl.generator.eval_sampling_params.temperature", 0.5, "evaluation protocol"),
        ("seed", "config.skyrl.trainer.seed", None, "malformed recorded configuration"),
    )
    for name, key, value, reason in edits:
        with subtests.test(case=name):
            candidate(name, verdict="invalid", reason=reason, receipt={key: value})
    candidate("baseline")
    candidate("improves", score=0.8, effect="improves")
    candidate("trade-off", score=0.65, completed_at="2026-11-06T23:40:00+00:00", effect="trade-off")
    omitted = recorded_runs(preset_name, "users/test/omitted-default")
    with edit_receipt(omitted) as receipt:
        receipt["config"]["skyrl"]["generator"].pop("chat_template_kwargs")
    assert_verdict("omitted-default", "invalid", reason="skyrl.generator.chat_template_kwargs")

    source = recorded_runs(preset_name, "users/test/new-groups")
    export = str(source / "exports")
    rows = read_evaluations(export, attempt_id="recorded-attempt", evaluation="eval/")
    for step in (0, preset.updates):
        row = next(row for row in rows if row.step == step)
        write_evaluations(export, [row.model_copy(update={"strata": row.strata.model_copy(update={"family": "other"})})])
        assert_verdict("new-groups", "invalid", reason="scoring groups differ")
        write_evaluations(export, [row])
    with edit_receipt(source) as receipt:
        receipt["result"]["model"] = None
    write_evaluations(
        export,
        [
            row.model_copy(
                update={
                    "answers": [answer.model_copy(update={"error_treatment": "passthrough"}) for answer in row.answers]
                }
            )
            for row in rows
        ],
    )
    report = assert_verdict("new-groups", "fail")
    assert any(row["name"] == "eval_lite/task_coverage" and not row["holds"] for row in report["invariants"])

    options = replace(preset, recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, priority="production")))
    candidate("priority", inputs=run_inputs(preset_name, preset=options), table=preset_table(options))
    preset_table(preset)
    baseline = assert_verdict("baseline")
    promoted = replace(
        preset,
        recipe=replace(
            preset.recipe,
            config=preset.recipe.config.with_settings(("trainer.policy.optimizer_config.weight_decay=0.1",)),
        ),
    )
    preset_table(promoted)
    changed = assert_verdict("baseline", "invalid", reason="weight_decay")
    assert changed["report"] != baseline["report"]
    preset_table(preset)
    _assert_panel_identity(recorded_runs)
    controls(recorded_runs, preset_name, score=0.5)
    candidate("zero-spread", verdict="invalid", reason="variation is zero")


def _assert_staged_model_and_draft(recorded_runs, members, assert_verdict):
    def stage(path, *, wrong_tokenizer=False, wrong_mirror=False):
        with edit_receipt(path) as receipt:
            launch = receipt["config"]
            model = launch["inputs"]["model"]
            stage_root = Path("/tmp/skyrl-worker")
            model_path = stage_root / model_metadata_relative_path(model["uri"], model["identity"])
            revision = "different" if wrong_tokenizer else model["tokenizer_revision"] or DEFAULT_TOKENIZER_REVISION
            tokenizer_path = stage_root / model_metadata_relative_path(model["tokenizer_uri"], revision)
            for role in ("policy", "ref"):
                launch["skyrl"]["trainer"][role]["model"].update(
                    path=str(model_path),
                    tokenizer_path=str(tokenizer_path),
                    tokenizer_revision=None,
                )
            draft = launch["skyrl"]["generator"]["speculative_decoding"]["model"]
            commit = draft["source_identity"]
            uri, identity = "s3://test-bucket/draft-mirror", "sha256:" + "f" * 64
            launch["inputs"]["hugging_face_models"] = [
                {
                    "model_id": draft["source_uri"].removeprefix("hf://"),
                    "revision": commit,
                    "resolved_revision": "a" * 40 if wrong_mirror else commit,
                    "uri": uri,
                    "identity": identity,
                }
            ]
            draft.update(source_uri=uri, source_identity=identity)

    for path in members:
        stage(path)
    for case, flags, reason in (
        ("staged", {}, ""),
        ("wrong-tokenizer", {"wrong_tokenizer": True}, "skyrl.trainer.policy.model.tokenizer_path"),
        ("wrong-mirror", {"wrong_mirror": True}, "skyrl.generator.speculative_decoding.model"),
    ):
        stage(recorded_runs("speed-short", f"users/test/{case}", speed=102), **flags)
        assert_verdict(case, "invalid" if reason else "pass", reason=reason)


def _assert_panel_identity(recorded_runs):
    preset = "learning-medium"
    source = recorded_runs(preset, "users/test/task-content")
    receipt = json.loads((source / "terminal.json").read_text())
    descriptor = receipt["config"]["inputs"]["validation_data"][0]
    original = Path(descriptor["uri"]) / descriptor["relative_path"]
    rows = pq.read_table(original).to_pylist()
    summary = json.loads((original.parent / "summary.json").read_text())["selection_strata"]
    annotations = pq.read_table(original.parent / TASK_MANIFEST_FILENAME).to_pylist()
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), original)
    assert compare(preset, "--candidate", "users/test/task-content")["verdict"] == "pass"
    changed = recorded_runs(preset, "users/test/task-content-edited")
    with edit_receipt(changed) as receipt:
        launch = receipt["config"]
        descriptor = launch["inputs"]["validation_data"][0]
        copied = changed / "task-set-copy"
        shutil.copytree(descriptor["uri"], copied)
        descriptor["uri"] = str(copied)
        launch["skyrl"]["data"]["val_data"] = [dict(descriptor)]
        changed_task = json.loads(rows[0]["task_spec"])
        changed_task["context"]["events"][0]["content"] = "Return dog."
        files = {descriptor["relative_path"], *summary["task_files"], *summary["active_files"]}
        for filename in files:
            pool = pq.read_table(copied / filename).to_pylist()
            for row in pool:
                if json.loads(row["task_spec"])["id"] == changed_task["id"]:
                    row["task_spec"] = json.dumps(changed_task)
            pq.write_table(pa.Table.from_pylist(pool), copied / filename)
        for annotation in annotations:
            if annotation["task_id"] == changed_task["id"]:
                annotation["task_digest"] = canonical_sha256(changed_task)
        pq.write_table(
            pa.Table.from_pylist(annotations, schema=arrow_schema(SelectionTaskRow)), copied / TASK_MANIFEST_FILENAME
        )
    report = compare(preset, "--candidate", "users/test/task-content-edited")
    assert report["verdict"] == "invalid" and "TaskSpec content differs" in report["reason"], report


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
    model_step = replace(preset.model.step, adopt_source=model_uri)
    preset = replace(
        preset,
        model=replace(preset.model, step=model_step),
        guards=snowball_model.SNOWBALL.by_name["learning-medium"].guards,
    )
    preset_table(preset)
    controls(recorded_runs, preset_name)
    tasks = selection if preset_name == "data-medium" else None
    settings = ("trainer.policy.optimizer_config.lr=2.0e-6",) if tasks is None else ()
    step = ArtifactStep.adopt(
        "tests/noncanonical-initializer",
        "2026.11.17",
        model_uri,
        kind=model_step.artifact_type,
        config={"tokenizer_uri": "tests/own-tokenizer"},
    )
    initializer = ArtifactHfModel(step, "tests/own-tokenizer", "own-tokenizer-revision") if preset.accepts_init else None
    if initializer is not None:
        inputs = run_inputs(preset.name, preset=preset, init=recipe.stored_initializer(initializer, preset=preset))
        with monkeypatch.context() as planning:
            planning.setattr(skyrl_api, "discover_hf_checkpoints", lambda _: pytest.fail("read unbuilt HF export"))
            assert gate.compose_preset(preset, inputs)["inputs"]["model"]["uri"].endswith("/<terminal-hf-export>")
    write_hf_export(model / "selected-hf")
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
        receipt={"result.started_at": "2026-11-06T17:40:00+00:00"},
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
    assert recorded.tasks == task_name
    if initializer is not None:
        assert (
            recorded.init.artifact_root,
            recorded.init.relative_path,
            recorded.init.tokenizer_uri,
            recorded.init.tokenizer_revision,
        ) == (model_uri, None, "tests/own-tokenizer", "own-tokenizer-revision")
        assert json.loads(recorded.init.fingerprint_payload)["config"]["tokenizer_uri"] == "tests/own-tokenizer"
        write_hf_export(model / "newer-hf")
    report = compare(preset_name, "--candidate", built.runs[0].name)
    assert report["verdict"] == "pass", report
    assert report["run"]["attempt_ids"] == ["recorded-attempt"]
    assert comparisons(report)[gain_key]["run"] == pytest.approx(0.005)
    _json(source / "exports/best_checkpoint.json", {"step": preset.updates})
    [status] = snowball._status(built, preset, stored_only=True)["runs"]
    assert status["best_checkpoint_step"] == preset.updates
    if tasks is None:
        assert {"inputs.model.uri", "skyrl.trainer.policy.optimizer_config.lr"} <= set(report["run"]["config_changes"])
