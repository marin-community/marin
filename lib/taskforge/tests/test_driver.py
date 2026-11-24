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
from experiments.post_training.capability_driven_envs import launch
from iris.cluster.types import JobName
from shellbox.backends.iris.machine import IrisMachineFactory

from taskforge import wiring
from taskforge.driver import cli, eligible, glm_source, inputs, proposal_bucket, proposal_key
from taskforge.model.proposal import Environment, TaskProposal, Verification
from taskforge.policy import load_policy
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


def test_driver_accepts_the_launchers_command() -> None:
    argv = ["--proposals", "p", "--output", "o", "--policy", "q", "--job-prefix", "j", "--drivers", "2"]
    argv += ["--concurrency", "4", "--cpu", "8", "--ranks", "0,1", "--resume-held", "--glm-relay-job", "/me/relay"]
    args = launch.parser().parse_args(argv)
    command = launch.commands(args, {"CW_KEY_ID": "k", "CW_KEY_SECRET": "s", "GLM_API_KEY": "g"})[1]
    parsed = cli().parse_args(command[command.index("taskforge.driver") + 1 :])
    assert (parsed.command, parsed.buckets, parsed.bucket_range) == ("run", 4096, (2048, 4096))
    assert (parsed.concurrency, parsed.ranks, parsed.resume_held) == (4, (0, 1), True)
    assert (parsed.glm_relay_job, parsed.glm_base_url) == ("/me/relay", None)


@pytest.mark.parametrize("command", ["run", "one"])
def test_driver_endpoint_flags_build_the_gateway_source(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLM_API_KEY", "secret-token")
    # The flags alone name the gateway: the environment variables they replaced are ignored.
    monkeypatch.setenv("GLM_BASE_URL", "http://stray/v1")
    monkeypatch.setenv("GLM_RELAY_JOB", "/stray/relay")
    policy = load_policy(DEFAULTS_DIR)
    argv = [command, "--proposals", "p", "--output", "o", "--policy", "q"] + ["--concurrency", "1"] * (command == "run")
    relay = glm_source(policy, cli().parse_args([*argv, "--glm-relay-job", "/user/relay"]))
    assert isinstance(relay, wiring.RelayEndpoint) and relay.job == "/user/relay"
    assert relay.model == policy.setting("author.model", str)
    assert relay.pool.value == policy.setting("provider.pool", str)
    fixed = glm_source(policy, cli().parse_args([*argv, "--glm-base-url", "http://fixed/v1"])).current()
    assert (fixed.base_url, fixed.token) == ("http://fixed/v1", "secret-token")


@pytest.mark.parametrize("command", ["run", "one"])
@pytest.mark.parametrize(
    "endpoints, expected_errors",
    [
        ((), ("--glm-base-url", "--glm-relay-job")),
        (("--glm-base-url", "http://fixed/v1", "--glm-relay-job", "/user/relay"), ("--glm-base-url", "--glm-relay-job")),
        (("--glm-relay-job", ""), ("argument --glm-relay-job",)),
        (("--glm-relay-job", " "), ("argument --glm-relay-job",)),
        (("--glm-base-url", ""), ("argument --glm-base-url",)),
        (("--glm-base-url", "http://glm"), ("argument --glm-base-url",)),
    ],
)
def test_driver_refuses_missing_conflicting_or_invalid_endpoint_flags(
    command: str, endpoints: tuple[str, ...], expected_errors: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [command, "--proposals", "p", "--output", "o", "--policy", "q"]
    if command == "run":
        argv += ["--concurrency", "1"]
    with pytest.raises(SystemExit) as stopped:
        cli().parse_args([*argv, *endpoints])
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert all(expected in error for expected in expected_errors)


def test_proposal_buckets_ignore_the_proposal_root() -> None:
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
    monkeypatch.setattr(wiring, "get_job_info", lambda: SimpleNamespace(controller_address="http://controller"))
    monkeypatch.setenv("IRIS_CONTROLLER_URL", "http://stray")  # only the entry point reads it, and passes it in
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected)
    policy = load_policy(Path(__file__).parents[1] / "src/taskforge/policy/defaults")
    source = wiring.endpoint_source("http://fake/v1", None, "fake", "glm-5.3", policy.setting("provider.pool", str))
    async with wiring.runtime(policy, source, "fake") as bindings:
        assert isinstance(bindings, wiring.CampaignWiring)
        _, resources = bindings
        assert set(resources.factories) == {"gvisor", "shellsim"}
        factory = resources.factories["gvisor"]
        assert isinstance(factory, IrisMachineFactory)
        assert factory.controller_url == "http://controller"
        assert factory.cluster is None
        assert factory.scheduling_timeout == SANDBOX_SCHEDULING_TIMEOUT
    async with wiring.runtime(policy, source, "fake", controller_url="http://explicit") as (_, resources):
        explicit = resources.factories["gvisor"]
        assert isinstance(explicit, IrisMachineFactory) and explicit.controller_url == "http://explicit"


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
