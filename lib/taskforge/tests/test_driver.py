# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import httpx
import pytest
from experiments.taskforge import run_taskforge
from iris.cluster.types import JobName
from rolloutengine.test_support import arithmetic_task
from shellbox.backends.iris.machine import IrisMachineFactory
from taskcompendium.models import GraderMachine, ScriptGrader, StdoutReward

from taskforge import wiring
from taskforge.driver import eligible, input_bucket, input_key, inputs, seed, source_images
from taskforge.lowering import requirements
from taskforge.model.proposal import Environment, TaskProposal, Verification
from taskforge.policy import GRADER_BASE_IMAGE, load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.sandbox.factories import SANDBOX_SCHEDULING_TIMEOUT


@pytest.mark.parametrize("verification", list(Verification))
def test_proposal_verification_policy_filters_before_execution(
    proposal: TaskProposal, verification: Verification
) -> None:
    proposal = replace(
        proposal, header=replace(proposal.header, environment=Environment.SHELLSIM, verification=verification)
    )
    policy = load_policy(DEFAULTS_DIR)
    allowed = verification.value in {"code", "decision_model", "answer_in_box"}
    assert eligible(proposal, policy) is allowed
    assert eligible(proposal, policy.overlaid({"proposals.verifications": [verification.value]}))


def test_manifest_selects_ranks_and_path_list_preserves_output_identity(tmp_path: Path) -> None:
    paths = tuple(f"s3://bucket/run/d02.swe.repair/slot-{slot}.proposal.json" for slot in (15, 8, 24))
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps(
            {
                "format": "acceptance-measure-sample-v1",
                "sample": [
                    {"rank": rank, "proposal": path, "capability": "d02.swe.repair", "slot": slot}
                    for rank, path, slot in zip((0, 1, 2), paths, (15, 8, 24), strict=True)
                ],
            }
        )
    )
    assert inputs(str(sample), (0, 2)) == (paths[0], paths[2])
    assert str(input_key(paths[0])) == "d02.swe.repair/15"
    manifest = tmp_path / "proposals.txt"
    manifest.write_text("\n".join(paths) + "\n")
    assert inputs(str(manifest)) == paths
    with pytest.raises(ValueError, match="missing"):
        inputs(str(sample), (3,))
    manifest.write_text(f"{paths[0]}\n{paths[0]}\n")
    with pytest.raises(ValueError, match="duplicate output"):
        inputs(str(manifest))


def test_prefix_lists_proposals_and_source_tasks_under_distinct_identities(tmp_path: Path) -> None:
    run = tmp_path / "swe-rebench"
    run.mkdir()
    for name in ("slot-1.proposal.json", "astropy-1.task.json", "notes.json"):
        (run / name).write_text("{}")
    paths = inputs(str(run))
    assert [Path(path).name for path in paths] == ["astropy-1.task.json", "slot-1.proposal.json"]
    assert [str(input_key(path)) for path in paths] == ["swe-rebench/astropy-1", "swe-rebench/1"]
    assert str(input_key("s3://bucket/run/slot-1.task.json")) == "run/slot-1", "a source task keeps its whole name"


def test_empty_prefix_fails_instead_of_reporting_a_successful_empty_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No proposal"):
        inputs(str(tmp_path))


def test_launcher_passes_secrets_by_reference_only(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {key: f"value-of-{key}-0123456789" for key in run_taskforge.SECRETS}
    for key, value in {**values, "GLM_RELAY_JOB": "/me/relay"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("GLM_BASE_URL", raising=False)
    commands: list[list[str]] = []
    monkeypatch.setattr(run_taskforge.subprocess, "run", lambda command, **_: commands.append(command))
    argv = ["launcher", "--arm", "S", "--drivers", "2", "--concurrency", "4", "--cpu", "8"]
    argv += [item for name in ("proposals", "output", "policy", "job-prefix") for item in (f"--{name}", "test")]
    monkeypatch.setattr(sys, "argv", argv)
    run_taskforge.main()
    assert len(commands) == 2
    for command in commands:
        assert not [arg for arg in command for value in values.values() if value in arg]
        secrets = {command[i + 1]: command[i + 2] for i, arg in enumerate(command) if arg == "--secret"}
        assert secrets == {key: f"env:{key}" for key in values}
        assert command[command.index("GLM_RELAY_JOB") + 1] == "/me/relay"


@pytest.mark.parametrize("drivers,span", [(7, None), (32, "24:32"), (64, "48:64")])
def test_launcher_bucket_coverage(monkeypatch: pytest.MonkeyPatch, drivers: int, span: str | None) -> None:
    for key in ("CW_KEY_ID", "CW_KEY_SECRET", "GLM_API_KEY", "GLM_BASE_URL"):
        monkeypatch.setenv(key, "fake")
    monkeypatch.delenv("GLM_RELAY_JOB", raising=False)
    commands: list[list[str]] = []
    monkeypatch.setattr(run_taskforge.subprocess, "run", lambda command, **_: commands.append(command))
    argv = ["launcher", "--arm", "S", "--drivers", str(drivers), "--concurrency", "4", "--cpu", "8"]
    argv += [item for name in ("proposals", "output", "policy", "job-prefix") for item in (f"--{name}", "test")]
    monkeypatch.setattr(sys, "argv", argv + (["--driver-range", span] if span else []))
    run_taskforge.main()
    ranges = [tuple(map(int, command[command.index("--bucket-range") + 1].split(":"))) for command in commands]
    covered = [bucket for start, end in ranges for bucket in range(start, end)]
    assert covered == list(range(3072 if span else 0, 4096))
    assert len(commands) == (int(span.split(":")[1]) - int(span.split(":")[0]) if span else drivers)
    assert max(end - start for start, end in ranges) - min(end - start for start, end in ranges) <= 1
    assert (
        3998
        == input_bucket("s3://old/cap/slot-8.proposal.json", 4096)
        == input_bucket("s3://new/prefix/cap/slot-8.proposal.json", 4096)
    )


async def test_iris_runtime_on_linux_never_constructs_local_backend_or_probes_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("Runtime initialization must not probe a sandbox or transport")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("IRIS_CONTROLLER_URL", raising=False)
    monkeypatch.setattr(wiring, "get_job_info", lambda: SimpleNamespace(controller_address="http://controller"))
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected)
    policy = load_policy(Path(__file__).parents[1] / "src/taskforge/policy/defaults")
    source = wiring.endpoint_source("http://fake/v1", None, "fake", "glm-5.3", policy.setting("provider.pool", str))
    async with wiring.runtime(policy, "S", source, "fake") as (_, resources):
        assert set(resources.factories) == {"gvisor", "shellsim"}
        factory = resources.factories["gvisor"]
        assert isinstance(factory, IrisMachineFactory)
        assert factory.controller_url == "http://controller"
        assert factory.cluster is None
        assert factory.scheduling_timeout == SANDBOX_SCHEDULING_TIMEOUT


@pytest.mark.parametrize("pool", ["high", "bulk"])
def test_relay_reresolves_by_job_and_model_without_exposing_token(monkeypatch: pytest.MonkeyPatch, pool: str) -> None:
    class Resolver:
        def __init__(self) -> None:
            self.url = "http://first:8000"
            self.models: list[str] = []

        def resolve(self, name: str) -> SimpleNamespace:
            self.models.append(name)
            return SimpleNamespace(endpoints=(SimpleNamespace(url=self.url),))

    resolver = Resolver()
    jobs: list[str] = []

    def for_job(job: JobName) -> Resolver:
        jobs.append(str(job))
        return resolver

    monkeypatch.setattr(wiring, "iris_ctx", lambda: SimpleNamespace(client=SimpleNamespace(resolver_for_job=for_job)))
    source = wiring.endpoint_source(None, "/user/relay", "secret-token", "glm-5.3", pool)
    endpoint = source.current()
    assert endpoint.base_url == "http://first:8000/v1"
    assert endpoint.pool.value == pool
    resolver.url = "http://replacement:8000/v1/"
    assert source.current().base_url == "http://replacement:8000/v1"
    assert jobs == ["/user/relay", "/user/relay"]
    assert resolver.models == ["glm-5.3", "glm-5.3"]
    assert wiring.endpoint_source("http://fixed/v1", None, "secret-token", "glm-5.3", pool).current().pool.value == pool
    assert "secret-token" not in repr(source)
    assert (
        wiring.endpoint_source("http://fixed/v1", None, "secret-token", "glm-5.3", pool).current().base_url
        == "http://fixed/v1"
    )
    with pytest.raises(ValueError, match="exactly one"):
        wiring.endpoint_source("http://fixed/v1", "/user/relay", "secret-token", "glm-5.3", pool)
    with pytest.raises(ValueError, match="exactly one"):
        wiring.endpoint_source(None, None, "secret-token", "glm-5.3", pool)


def test_source_images_are_the_source_tasks_own_task_and_grader_images(tmp_path: Path) -> None:
    task_image = "ghcr.io/example/swe-task@sha256:" + "a" * 64
    grader_image = "ghcr.io/example/swe-grader@sha256:" + "b" * 64
    grader = ScriptGrader(
        argv=("bash", "/tests/test.sh"),
        reward=StdoutReward(),
        placement=GraderMachine(environment=requirements(image=grader_image, workdir="/testbed")),
        answer_path=None,
        timeout=600,
    )
    policy = load_policy(DEFAULTS_DIR)
    for name, grading, expected in (
        ("separate", grader, (task_image, grader_image)),
        (
            "pinned",
            grader.model_copy(update={"placement": GraderMachine(environment=requirements(image=GRADER_BASE_IMAGE))}),
            (task_image,),
        ),
    ):
        path = tmp_path / f"{name}.task.json"
        task = arithmetic_task(
            environment_requirements=requirements(image=task_image, workdir="/testbed"), grader=grading
        )
        path.write_text(task.model_dump_json())
        workpiece = seed(str(path), policy)
        assert workpiece is not None and source_images(workpiece) == tuple(sorted(expected))
