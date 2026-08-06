# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris service deploy contract: spec validation, the code-hash redeploy trigger,
submit/terminate semantics, and the Command stdout the component parses.

Guards the boundaries a bad deploy crosses: a malformed spec must fail before any
cluster interaction, the hash must ignore mtimes but see content (else previews lie),
and ``down`` must treat a missing job as success (else ``pulumi destroy`` wedges after
an out-of-band terminate).
"""

import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iac.iris.deploy import (
    check_bundle_includes,
    cli,
    resources_from_spec,
    run_build_commands,
    submit_service,
    terminate_service,
)
from iac.iris.service import IrisServiceArgs, _parse_outputs, code_hash, wire_spec
from iac.iris.spec import ALWAYS_ON_RETRIES, ServiceSpec
from iris.cluster.types import JobName, ResourceSpec, tpu_device
from iris.rpc import job_pb2

# Mirrors IrisServiceArgs minus the component-only fields; keyword overrides per test.
_SPEC_KWARGS = dict(
    cluster="test-cluster",
    name="svc",
    user="ops",
    entrypoint=("python", "-m", "svc.server"),
    resources={"cpuMillicores": 4000, "memoryBytes": 1024},
    regions=("us-east5",),
    port="svc",
    endpoint="/svc",
    health_path="/health",
)


def _spec(**overrides) -> ServiceSpec:
    return ServiceSpec(**{**_SPEC_KWARGS, **overrides})


class TestSpec:
    def test_json_round_trip(self):
        spec = _spec(
            env={"A": "1"},
            secret_env={"B": "env:B_SRC"},
            pip_packages=("xprof==2.22.3",),
            sync_packages=("marin-levanter",),
            deploy_generation=3,
        )
        assert ServiceSpec.from_json(spec.to_json()) == spec

    def test_to_json_ignores_dict_insertion_order(self):
        # The JSON is a Pulumi input: two programs building the same env in different
        # orders must serialize identically or every deploy shows a phantom diff.
        forward = _spec(env={"A": "1", "B": "2"}, secret_env={"X": "env:X", "Y": "env:Y"})
        reversed_ = _spec(env={"B": "2", "A": "1"}, secret_env={"Y": "env:Y", "X": "env:X"})
        assert forward.to_json() == reversed_.to_json()

    def test_unknown_field_rejected(self):
        raw = json.loads(_spec().to_json())
        raw["nope"] = 1
        with pytest.raises(ValueError, match="unknown spec fields"):
            ServiceSpec.from_json(json.dumps(raw))

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"name": "a/b"}, "no '/'"),
            ({"user": ""}, "user"),
            ({"entrypoint": ()}, "entrypoint"),
            ({"regions": ()}, "regions"),
            ({"port": ""}, "port"),
            ({"endpoint": "svc"}, "start with '/'"),
            ({"health_path": "health"}, "must start with '/'"),
            ({"env": {"KEY": "gcp-secret://projects/p/secrets/s/versions/1"}}, "move it to secret_env"),
            ({"secret_env": {"KEY": "plaintext-value"}}, "not a secret reference"),
        ],
    )
    def test_validate_rejects(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            _spec(**overrides).validate()


def _args(resources: ResourceSpec) -> IrisServiceArgs:
    return IrisServiceArgs(**_SPEC_KWARGS | {"resources": resources})  # pyrefly: ignore


class TestResources:
    def test_round_trip_through_wire_dict(self):
        # wire_spec serializes via the proto; resources_from_spec must reconstruct an
        # equivalent ResourceSpec (identical proto), or the deployed job differs from
        # what the committed stack declared.
        resources = ResourceSpec(cpu=180.0, memory="690GB", device=tpu_device("v6e-4"))
        spec = dataclasses.replace(_spec(), resources=wire_spec(_args(resources)).resources)
        assert resources_from_spec(spec).to_proto() == resources.to_proto()

    def test_cpu_only_round_trip(self):
        resources = ResourceSpec(cpu=4.0, memory="8GB")
        spec = dataclasses.replace(_spec(), resources=wire_spec(_args(resources)).resources)
        reconstructed = resources_from_spec(spec)
        assert reconstructed.device is None
        assert reconstructed.to_proto() == resources.to_proto()


class _CapturingClient:
    def __init__(self):
        self.kwargs = None

    def submit(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(job_id="/ops/svc")


class TestSubmit:
    def test_always_on_budgets_and_policy(self):
        client = _CapturingClient()
        submit_service(
            client,
            _spec(pip_packages=("xprof==2.22.3",), sync_packages=("marin-iris", "marin-rigging")),
            {"K": "v"},
        )  # pyrefly: ignore
        kwargs = client.kwargs
        assert kwargs["max_retries_preemption"] == ALWAYS_ON_RETRIES
        assert kwargs["max_retries_failure"] == ALWAYS_ON_RETRIES
        # max_task_failures defaults to 0 job-wide: one hard container failure would
        # otherwise end the service regardless of the per-task budget.
        assert kwargs["max_task_failures"] == ALWAYS_ON_RETRIES
        # Submit is always RECREATE — the only policy the deploy uses.
        assert kwargs["existing_job_policy"] == job_pb2.EXISTING_JOB_POLICY_RECREATE
        assert kwargs["user"] == "ops"
        assert kwargs["ports"] == ["svc"]
        assert kwargs["environment"].env_vars == {"K": "v"}
        assert kwargs["environment"].pip_packages == ("xprof==2.22.3",)
        assert kwargs["environment"].sync_packages == ("marin-iris", "marin-rigging")

    def test_region_pin(self):
        client = _CapturingClient()
        submit_service(client, _spec(regions=("us-east5",)), {})  # pyrefly: ignore
        (constraint,) = client.kwargs["constraints"]
        assert "us-east5" in str(constraint)


class _TerminatingClient:
    """States plays back per job_state call; terminate() is recorded."""

    def __init__(self, states):
        self._states = list(states)
        self.terminated = False

    def job_state(self, job_id):
        state = self._states.pop(0)
        if state is None:
            raise ConnectError(Code.NOT_FOUND, "not found")
        return state

    def terminate(self, job_id):
        self.terminated = True


class TestTerminate:
    def test_missing_job_is_success(self):
        client = _TerminatingClient([None])
        terminate_service(client, JobName.root("ops", "svc"))  # pyrefly: ignore
        assert not client.terminated

    def test_terminal_job_is_success(self):
        client = _TerminatingClient([job_pb2.JOB_STATE_KILLED])
        terminate_service(client, JobName.root("ops", "svc"))  # pyrefly: ignore
        assert not client.terminated

    def test_running_job_terminates_and_waits(self):
        client = _TerminatingClient([job_pb2.JOB_STATE_RUNNING, job_pb2.JOB_STATE_KILLED])
        terminate_service(client, JobName.root("ops", "svc"), wait=5)  # pyrefly: ignore
        assert client.terminated

    def test_job_vanishing_during_drain_is_success(self):
        client = _TerminatingClient([job_pb2.JOB_STATE_RUNNING, None])
        terminate_service(client, JobName.root("ops", "svc"), wait=5)  # pyrefly: ignore
        assert client.terminated


class TestBundleIncludes:
    def test_missing_build_output_fails(self, tmp_path):
        with pytest.raises(click.ClickException, match="matches no files"):
            check_bundle_includes(tmp_path, ("dist/**/*",))

    def test_present_build_output_passes(self, tmp_path):
        (tmp_path / "dist").mkdir()
        (tmp_path / "dist" / "index.html").write_text("x")
        # Not raising is the contract: the check's only job is to reject a missing
        # build output, so a matched glob must pass the deploy through untouched.
        check_bundle_includes(tmp_path, ("dist/**/*",))


class TestBuildCommands:
    def test_commands_run_in_order_from_workspace(self, tmp_path):
        run_build_commands(tmp_path, ("mkdir dist", "echo built > dist/app.js"))
        assert (tmp_path / "dist" / "app.js").read_text().strip() == "built"

    def test_failing_command_aborts(self, tmp_path):
        with pytest.raises(click.ClickException, match="build command failed"):
            run_build_commands(tmp_path, ("false", "mkdir dist"))
        # The failing command must abort the sequence (nothing after it runs).
        assert not (tmp_path / "dist").exists()


def _git_repo(root: Path, files: dict[str, str]) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


class TestCodeHash:
    def test_stable_across_mtimes(self, tmp_path):
        _git_repo(tmp_path, {"lib/svc/a.py": "one", "lib/svc/b.py": "two"})
        first = code_hash(tmp_path, ("lib/svc",))
        for path in tmp_path.rglob("*.py"):
            path.touch()
        assert code_hash(tmp_path, ("lib/svc",)) == first

    def test_content_change_changes_hash(self, tmp_path):
        _git_repo(tmp_path, {"lib/svc/a.py": "one"})
        first = code_hash(tmp_path, ("lib/svc",))
        (tmp_path / "lib/svc/a.py").write_text("changed")
        # git ls-files lists the path; the hash reads working-tree bytes.
        assert code_hash(tmp_path, ("lib/svc",)) != first

    def test_out_of_scope_change_ignored(self, tmp_path):
        _git_repo(tmp_path, {"lib/svc/a.py": "one", "lib/other/b.py": "two"})
        first = code_hash(tmp_path, ("lib/svc",))
        (tmp_path / "lib/other/b.py").write_text("changed")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        assert code_hash(tmp_path, ("lib/svc",)) == first

    def test_generated_outputs_not_hashed(self, tmp_path):
        # Build outputs are rebuilt inside every `up`, so the trigger hashes sources
        # only — a dist/ produced (or not yet produced) on this machine must not move
        # the hash, or fresh CI runners and warm operator checkouts would disagree.
        _git_repo(tmp_path, {"lib/svc/a.py": "one", ".gitignore": "dist/\n"})
        first = code_hash(tmp_path, ("lib/svc",))
        dist = tmp_path / "lib/svc/dashboard/dist"
        dist.mkdir(parents=True)
        (dist / "app.js").write_text("bundle-v1")
        assert code_hash(tmp_path, ("lib/svc",)) == first


class TestOutputs:
    def test_parse_outputs(self):
        outputs = _parse_outputs('{"job_id": "/ops/svc", "url": "https://c/proxy/svc", "ready": true}')
        assert outputs["job_id"] == "/ops/svc"

    def test_non_json_stdout_rejected(self):
        with pytest.raises(ValueError, match="not a JSON document"):
            _parse_outputs("Establishing tunnel to controller...\n{}")

    def test_missing_keys_rejected(self):
        with pytest.raises(ValueError, match="missing job_id/url"):
            _parse_outputs('{"job_id": "/ops/svc"}')


class TestCli:
    def test_up_without_spec_fails(self, monkeypatch):
        monkeypatch.delenv("IRIS_SVC_SPEC", raising=False)
        result = CliRunner().invoke(cli, ["up"])
        assert result.exit_code != 0
        assert "IRIS_SVC_SPEC" in result.output

    def test_up_with_invalid_spec_fails_before_connecting(self, monkeypatch):
        monkeypatch.setenv("IRIS_SVC_SPEC", _spec(regions=()).to_json())
        result = CliRunner().invoke(cli, ["up"])
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)

    def test_up_with_missing_build_output_fails_before_connecting(self, monkeypatch, tmp_path):
        # The bundle-include check must run before any controller interaction, so a
        # forgotten dashboard build can never RECREATE (and so kill) the live instance.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("IRIS_SVC_SPEC", _spec(extra_bundle_includes=("dist/**/*",)).to_json())
        result = CliRunner().invoke(cli, ["up"])
        assert result.exit_code != 0
        assert "matches no files" in result.output
