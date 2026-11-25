# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import runpy
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from finestore.rl.run_records import AttemptRecord, read_attempt, read_evaluations, write_attempt
from iris.client.client import IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel, SkyRLRun

from experiments.post_training import rl as snowball
from experiments.post_training import rl_gate as gate
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_tests.fixtures import (
    RL_TEST_MARKS,
    edit_receipt,
    fake_iris,
    fake_publication,
    fake_s3,
    printed_status,
    replace_evaluations,
    write_receipt,
    write_skyrl_run_record,
)
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.skyrl_test_support import native_json

pytestmark = RL_TEST_MARKS


def test_train_long_reenters_the_model_module_and_reports_checkpoint_evaluations(
    recorded_runs, preset_table, tmp_path, monkeypatch
):
    preset = snowball_model.SNOWBALL.by_name["train-long"]
    preset = replace(
        preset,
        checkpoints="best",
        export="best",
        recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, host_disk="3TB")),
    )
    table = preset_table(preset)
    initializer = ArtifactHfModel(
        ArtifactStep.adopt("tests/python-init", "2026.11.17", "s3://marin-test/models/python-init", kind=Artifact),
        "tests/python-tokenizer",
        "python-tokenizer-revision",
        relative_path="",
    )
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    monkeypatch.setenv("HF_TEST_KEY", "test-initializer-credential")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    fake_s3(monkeypatch, "marin-test/models/python-init")
    built = snowball.build_experiment(
        preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=("trainer.max_steps=12",)
    )
    source = recorded_runs(preset.name, built.runs[0].name, fingerprint=built.runs[0].fingerprint(), inputs=built.inputs)
    _json(source / "exports/best_checkpoint.json", {"step": 10})
    staged = source.with_name("before-native-execution")
    source.rename(staged)
    native_calls = []

    def native(command, *, env_vars=None):
        inspection = native_json()(
            "import json,sys\nfrom cloud.iris import launch\n"
            "args=launch.create_parser().parse_args(sys.argv[1:])\n"
            "config=launch.load_launch_config(args.config)\n"
            "print(json.dumps({'forward_env':args.forward_env,'disk':config.iris.allocation.disk,"
            "'tokenizer':config.inputs.model.tokenizer_uri,'model':config.run.model,"
            "'attempt':config.run.attempt_id,'export':config.artifacts.export_root}))\n",
            command[command.index("marinskyrl") + 1 :],
        )
        assert (inspection["forward_env"], inspection["disk"], inspection["tokenizer"]) == (
            ["HF_TEST_KEY"],
            "3TB",
            "tests/python-tokenizer",
        )
        assert read_attempt(inspection["export"], inspection["attempt"]) is None
        native_calls.append(command)
        shutil.copytree(staged, source, dirs_exist_ok=True)
        stored = json.loads((source / "terminal.json").read_text())
        stored["config"]["run"]["model"] = inspection["model"]
        write_receipt(source, stored)
        write_attempt(
            inspection["export"],
            AttemptRecord(
                run_id=stored["config"]["run"]["id"],
                attempt_id=stored["config"]["run"]["attempt_id"],
            ),
        )
        response = stored["result"]
        response["model"] = dict(
            policy_export_uri=str(source / "policy"),
            global_step=12,
            tokenizer_uri="tests/python-tokenizer",
            tokenizer_revision="python-tokenizer-revision",
            checkpoint_root=str(source / "checkpoints"),
            terminal_manifest_uri=str(source / "terminal.json"),
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(skyrl_api, "_run_launcher", native)
    published = []

    fake_publication(monkeypatch, published.append, "http://progress.test")
    arguments = [
        "--preset",
        preset.name,
        "--version",
        preset.revision,
        "--run",
        "--wait",
        "--marina-url",
        "http://127.0.0.1:55777",
        "--max-concurrent",
        "3",
        "--set",
        "trainer.max_steps=12",
    ]
    client, job = fake_iris(monkeypatch, "train-coordinator")
    mode, coordinator_module = "valid", "first_model_entry"

    def submit(**kwargs):
        assert kwargs["entrypoint"].command == ["python", "-m", coordinator_module, *arguments]
        environment = kwargs["environment"].env_vars
        if mode != "overlapping-environment-mismatch":
            assert environment["HF_TEST_KEY"] == "test-initializer-credential"
        assert "UNRELATED_SECRET" not in environment
        if mode == "failed-native":
            return SimpleNamespace(job_id=job)
        with monkeypatch.context() as child:
            for key, value in environment.items():
                child.setenv(key, value)
            child.setenv("IRIS_TASK_ID", "/test-user/train-coordinator/0")
            child.setattr(sys, "argv", [coordinator_module, *arguments])
            child.setattr(
                "experiments.post_training.rl_publication.publish_result",
                lambda *args: pytest.fail("Coordinator contacted Marina"),
            )
            info = JobInfo(task_id=JobName.from_wire("/test-user/train-coordinator/0"))
            with iris_ctx_scope(IrisContext.from_job_info(info)), pytest.raises(SystemExit) as completed:
                runpy.run_module(coordinator_module, run_name="__main__", alter_sys=True)
            assert completed.value.code == {"valid": 0, "failed-following": 3}.get(mode, 4)
        set_job_info(None)
        return SimpleNamespace(job_id=job)

    client.submit.side_effect = submit
    client.job.return_value.wait.side_effect = lambda *args, **kwargs: SimpleNamespace(
        state=JobState.SUCCEEDED if mode == "valid" else JobState.FAILED
    )

    policy_uri = "s3://marin-test/models/python-policy"
    policy_path = tmp_path / "python-policy"
    raw_load = SkyRLRun.raw_load
    monkeypatch.setattr(SkyRLRun, "raw_load", lambda path: raw_load(str(policy_path) if path == policy_uri else path))

    def module(
        name, *, entry_module=None, use_init=True, forward_env=("HF_TEST_KEY",), policy_forward=None, following="None"
    ):
        path = tmp_path / (name + ".py")
        dependency = ""
        if policy_forward is not None:
            dependency = (
                f"producer=rl.build_experiment(snowball_rl.SNOWBALL.by_name['train-long'],table=snowball_rl.SNOWBALL,"
                f"settings=('trainer.max_steps=2',),forward_env={policy_forward!r}).runs[0]\n"
                f"result=SkyRLRun.raw_load({str(source)!r}).model_copy(update={{'hf_model_uri':'s3://marin-test/models/python-init'}})\n"
                f"directory=Path({str(policy_path)!r});directory.mkdir(parents=True,exist_ok=True)\n"
                "record=ArtifactRecord(name=producer.name,version=producer.version,fingerprint=producer.fingerprint(),"
                "result_type=result_type_name(SkyRLRun),result=result.result_payload())\n"
                "(directory/RECORD_FILENAME).write_text(record.model_dump_json())\n"
                f"init=SkyRLPolicyModel(replace(producer,override_path={policy_uri!r}))\n"
            )
        path.write_text(
            "from dataclasses import replace\nfrom pathlib import Path\n"
            "from marin.execution.lazy import ArtifactStep, apply\n"
            "from marin.execution.artifact import Artifact,ArtifactRecord,RECORD_FILENAME,result_type_name\n"
            "from marin.rl.skyrl import ArtifactHfModel,SkyRLRun,SkyRLPolicyModel\n"
            "from experiments.post_training import rl, snowball_rl\n"
            "step=ArtifactStep.adopt('tests/python-init','2026.11.17','s3://marin-test/models/python-init',kind=Artifact)\n"
            "init=ArtifactHfModel(step,'tests/python-tokenizer','python-tokenizer-revision',relative_path='')\n"
            "def fail_following():\n    raise RuntimeError('following stage failed')\n"
            + dependency
            + f"following={following}\n"
            "table=snowball_rl.SNOWBALL\n"
            "if following is not None:\n"
            "    table=replace(table,presets=tuple(replace(preset,then=following) for preset in table.presets))\n"
            + f"raise SystemExit(rl.main(table,init={'init' if use_init else 'None'},"
            f"forward_env={forward_env!r},entry_module={entry_module!r}))\n"
        )
        # Same-size rewrites within one timestamp tick must load the new module.
        Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        monkeypatch.setitem(sys.modules, name, importlib.util.module_from_spec(spec))

    @click.command()
    def entry():
        monkeypatch.setattr(sys, "argv", ["first_model_entry", *arguments])
        with pytest.raises(SystemExit) as completed:
            runpy.run_module("first_model_entry", run_name="__main__", alter_sys=True)
        raise click.exceptions.Exit(completed.value.code)

    def invoke_with_preflight_stub(command, args=()):
        # Native preflight has its own tests. Use the real composer to create
        # recorded fixtures and inspect the native launch in this test.
        with monkeypatch.context() as preflight:
            preflight.setattr(
                gate, "compose_configs", lambda paths, **_: [{"runtime": {"training_type": "async"}} for _ in paths]
            )
            return CliRunner().invoke(command, args)

    for override in (None, "second_model_entry"):
        coordinator_module = override or "first_model_entry"
        module("first_model_entry", entry_module=override)
        if override is not None:
            module(override)
        result = invoke_with_preflight_stub(entry)
        assert result.exit_code == 0, (result.output, result.exception)
        assert "best_checkpoint_step: 10" in result.output and f"{preset.evaluations[0].name}@12/scored" in result.output
        assert "http://127.0.0.1:55777/" in result.output and "test-initializer-credential" not in result.output
    assert len(native_calls) == 1
    assert published[-1].optimizer_steps == published[-1].planned_updates == 12
    assert (published[-1].model, published[-1].issue) == ("snowball", None)

    for mode in (
        "initializer-mismatch",
        "environment-mismatch",
        "overlapping-environment-mismatch",
        "dependency-environment-mismatch",
    ):
        coordinator_module = "second_model_entry"
        module(
            "first_model_entry",
            entry_module=coordinator_module,
            forward_env=("WANDB_API_KEY",) if mode == "overlapping-environment-mismatch" else ("HF_TEST_KEY",),
            policy_forward=("WANDB_API_KEY",) if mode == "dependency-environment-mismatch" else None,
        )
        module(
            coordinator_module,
            use_init=mode != "initializer-mismatch",
            forward_env=() if mode in {"environment-mismatch", "overlapping-environment-mismatch"} else ("HF_TEST_KEY",),
            policy_forward=() if mode == "dependency-environment-mismatch" else None,
        )
        result = invoke_with_preflight_stub(entry)
        assert result.exit_code == 3 and "coordinator rebuilt a different RL graph" in result.output, (
            result.output,
            result.exception,
        )
        assert "test-initializer-credential" not in result.output and len(native_calls) == 1

    mode, coordinator_module = "failed-following", "first_model_entry"
    module(coordinator_module, following="lambda _: apply('tests/following', fail_following, version='2026.11.20.1')")
    result = invoke_with_preflight_stub(entry)
    assert result.exit_code == 3 and "post_run_error:" in result.output, (result.output, result.exception)
    assert "RL has a committed result" in result.output

    evaluations = read_evaluations(str(source / "exports"), attempt_id="recorded-attempt", evaluation="eval/")
    replace_evaluations(
        source,
        [
            row.model_copy(
                update={
                    "answers": [answer.model_copy(update={"error_treatment": "passthrough"}) for answer in row.answers]
                }
            )
            for row in evaluations
            if row.step != 10
        ],
    )
    (source / "exports/best_checkpoint.json").unlink()
    with edit_receipt(source) as receipt:
        receipt["result"]["model"] = None
    step = built.runs[0]
    write_skyrl_run_record(source, step.name, step.version, fingerprint=step.fingerprint(), global_step=None)
    planning_args = ["--preset", preset.name, "--version", preset.revision, "--set", "trainer.max_steps=12"]
    result = invoke_with_preflight_stub(
        snowball.command(table, argv=tuple(planning_args), init=initializer, forward_env=("HF_TEST_KEY",)),
        planning_args,
    )
    assert result.exit_code == 1, (result.output, result.exception)
    [status] = printed_status(result.stdout)["runs"]
    assert status["best_checkpoint_step"] == "none eligible"
    facts = {fact["name"]: fact for fact in status["evaluation_invariants"]}
    assert not facts[f"{preset.evaluations[0].name}/step-10/answers_complete"]["holds"]
    assert not facts[f"{preset.evaluations[0].name}/task_coverage"]["holds"]

    mode, coordinator_module = "valid", "first_model_entry"
    module(coordinator_module)
    resumed = invoke_with_preflight_stub(entry)
    assert resumed.exit_code == 1 and len(native_calls) == 1, (resumed.output, resumed.exception)
    assert "best_checkpoint_step: none eligible" in resumed.output

    replace_evaluations(source, evaluations)

    live_preset = replace(preset, checkpoints="none", checkpoint_every=None, export="none")
    live = snowball.build_experiment(
        live_preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=("trainer.max_steps=12",)
    )
    assert live.runs[0].name != built.runs[0].name
    with monkeypatch.context() as live_record:
        live_record.setattr(snowball_model, "SNOWBALL", replace(table, presets=(live_preset,)))
        recorded_runs(preset.name, live.runs[0].name, fingerprint=live.runs[0].fingerprint(), inputs=live.inputs)
    live_args = [*planning_args, "--checkpoints", "none", "--checkpoint-every", "0", "--export", "none"]
    result = invoke_with_preflight_stub(
        snowball.command(table, argv=tuple(live_args), init=initializer, forward_env=("HF_TEST_KEY",)), live_args
    )
    assert result.exit_code == 1, (result.output, result.exception)
    facts = {
        fact["name"]: fact for row in printed_status(result.stdout)["runs"] for fact in row["evaluation_invariants"]
    }
    assert not facts[f"{preset.evaluations[0].name}/step-10/answers_complete"]["holds"]

    mode, coordinator_module = "failed-native", "first_model_entry"
    module("first_model_entry")
    with edit_receipt(source) as receipt:
        receipt["result"]["state"] = "failed"
    result = invoke_with_preflight_stub(entry)
    assert result.exit_code == 3 and published[-1].status == "failed", (result.output, result.exception)
