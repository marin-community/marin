# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import httpx
import pytest
from experiments.forge import run_forge
from iris.cluster.types import JobName
from shellbox.backends.iris.machine import IrisMachineFactory

from forge import driver, wiring
from forge.driver import eligible, inputs, proposal_bucket, proposal_key
from forge.model.proposal import TASK_PROPOSAL, Environment, TaskProposal, Verification
from forge.policy import load_policy
from forge.policy.defaults import DEFAULTS_DIR
from forge.resources.sandbox.factories import SANDBOX_SCHEDULING_TIMEOUT

pytest_plugins = ("lib.forge.tests.test_proposal",)


@pytest.mark.parametrize("verification", list(Verification))
def test_proposal_verification_policy_filters_before_execution(
    proposal: TaskProposal, verification: Verification, caplog: pytest.LogCaptureFixture
) -> None:
    proposal = replace(
        proposal, header=replace(proposal.header, environment=Environment.SHELLSIM, verification=verification)
    )
    policy = load_policy(DEFAULTS_DIR)
    allowed = verification.value in {"code", "decision_model", "answer_in_box"}
    with caplog.at_level("INFO"):
        assert eligible(proposal, policy) is allowed
    if not allowed:
        assert f"SKIP {proposal.header.id}: verification={verification.value}" in caplog.text
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
    assert str(proposal_key(paths[0])) == "d02.swe.repair/15"
    manifest = tmp_path / "proposals.txt"
    manifest.write_text("\n".join(paths) + "\n")
    assert inputs(str(manifest)) == paths
    with pytest.raises(ValueError, match="missing"):
        inputs(str(sample), (3,))
    manifest.write_text(f"{paths[0]}\n{paths[0]}\n")
    with pytest.raises(ValueError, match="duplicate output"):
        inputs(str(manifest))


def test_empty_prefix_fails_instead_of_reporting_a_successful_empty_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No proposal"):
        inputs(str(tmp_path))


@pytest.mark.parametrize("drivers,span", [(7, None), (32, "24:32"), (64, "48:64")])
def test_launcher_bucket_coverage(monkeypatch: pytest.MonkeyPatch, drivers: int, span: str | None) -> None:
    for key in ("CW_KEY_ID", "CW_KEY_SECRET", "GLM_API_KEY", "GLM_BASE_URL"):
        monkeypatch.setenv(key, "fake")
    monkeypatch.delenv("GLM_RELAY_JOB", raising=False)
    commands: list[list[str]] = []
    monkeypatch.setattr(run_forge.subprocess, "run", lambda command, **_: commands.append(command))
    argv = ["launcher", "--arm", "S", "--drivers", str(drivers), "--concurrency", "4", "--cpu", "8"]
    argv += [item for name in ("proposals", "output", "policy", "job-prefix") for item in (f"--{name}", "test")]
    monkeypatch.setattr(sys, "argv", argv + (["--driver-range", span] if span else []))
    run_forge.main()
    ranges = [tuple(map(int, command[command.index("--bucket-range") + 1].split(":"))) for command in commands]
    covered = [bucket for start, end in ranges for bucket in range(start, end)]
    assert covered == list(range(3072 if span else 0, 4096))
    assert len(commands) == (int(span.split(":")[1]) - int(span.split(":")[0]) if span else drivers)
    assert max(end - start for start, end in ranges) - min(end - start for start, end in ranges) <= 1
    assert (
        3998
        == proposal_bucket("s3://old/cap/slot-8.proposal.json", 4096)
        == proposal_bucket("s3://new/prefix/cap/slot-8.proposal.json", 4096)
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
    policy = load_policy(Path(__file__).parents[1] / "src/forge/policy/defaults")
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
    source = wiring.endpoint_source(None, "/muchanem/relay", "secret-token", "glm-5.3", pool)
    endpoint = source.current()
    assert endpoint.base_url == "http://first:8000/v1"
    assert endpoint.pool.value == pool
    resolver.url = "http://replacement:8000/v1/"
    assert source.current().base_url == "http://replacement:8000/v1"
    assert jobs == ["/muchanem/relay", "/muchanem/relay"]
    assert resolver.models == ["glm-5.3", "glm-5.3"]
    assert wiring.endpoint_source("http://fixed/v1", None, "secret-token", "glm-5.3", pool).current().pool.value == pool
    assert "secret-token" not in repr(source)
    assert (
        wiring.endpoint_source("http://fixed/v1", None, "secret-token", "glm-5.3", pool).current().base_url
        == "http://fixed/v1"
    )
    with pytest.raises(ValueError, match="exactly one"):
        wiring.endpoint_source("http://fixed/v1", "/muchanem/relay", "secret-token", "glm-5.3", pool)
    with pytest.raises(ValueError, match="exactly one"):
        wiring.endpoint_source(None, None, "secret-token", "glm-5.3", pool)


async def test_driver_defers_decision_model_proposals_while_held_and_resumes_them_from_the_log(
    proposal: TaskProposal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinds = {1: Verification.DECISION_MODEL, 2: Verification.CODE, 3: Verification.DECISION_MODEL, 4: Verification.CODE}
    paths = []
    for slot, verification in kinds.items():
        path = tmp_path / "cap" / f"slot-{slot}.proposal.json"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(
            TASK_PROPOSAL.dump_json(replace(proposal, header=replace(proposal.header, verification=verification)))
        )
        paths.append(str(path))
    manifest = tmp_path / "proposals.txt"
    manifest.write_text("\n".join(paths) + "\n")
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"proposals.environments": ["container"], "decision_model.retry_interval": 0.2}))
    # Slot 1 is held twice (its first resume is the probe that finds Clef still down), then Clef answers.
    exits = {paths[0]: [driver.DECISION_MODEL_HELD, driver.DECISION_MODEL_HELD]}
    launches: list[tuple[str, bool]] = []

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            path = command[command.index("--proposals") + 1]
            launches.append((Path(path).name.split(".")[0], "--resume-held" in command))
            self.code = exits.get(path, [0]).pop(0) if exits.get(path) else 0

        def wait(self) -> int:
            return self.code

        def poll(self) -> int:
            return self.code

    monkeypatch.setattr(driver, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(driver.subprocess, "Popen", FakeProcess)
    args = argparse.Namespace(
        proposals=str(manifest),
        ranks=(),
        buckets=1,
        bucket_range=None,
        concurrency=1,
        policy=str(overlay),
        arm="S",
        output=str(tmp_path / "out"),
        work_dir=str(tmp_path / "work"),
        resume_held=False,
    )
    await driver.run(args)
    assert launches == [
        ("slot-1", False),  # held: Clef unavailable
        ("slot-2", False),  # slot-3 is deferred without a launch; other kinds continue
        ("slot-4", False),
        ("slot-1", True),  # the one probe after retry_interval, still down
        ("slot-3", True),  # the next probe answers
        ("slot-1", True),
    ]
