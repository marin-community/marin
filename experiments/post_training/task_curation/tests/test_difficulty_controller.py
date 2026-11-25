# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller credentials reach Iris through versioned Secret Manager references."""

import json
import sys
from contextlib import ExitStack
from types import SimpleNamespace

import google.cloud
import pytest
import requests
from click.testing import CliRunner
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finelog.client import LogClient
from finelog.embedded import is_available as finelog_native_available
from finelog.embedded import require_embedded_server
from finestore.rl.difficulty import Solver
from iris.cli.job import run
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.bundle import BundleStore
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import JobName
from iris.rpc import controller_pb2
from iris.testing.controller import MockController, make_controller_state
from rigging.secrets import resolve_secret_spec
from rigging.server_auth import VerifiedIdentity, identity_scope

from experiments.post_training.task_curation.difficulty import controller
from experiments.post_training.task_curation.difficulty.iris_backend import IrisBackend
from experiments.post_training.task_curation.difficulty.planner import RebuildEnvironmentArtifact
from experiments.post_training.task_curation.difficulty.state import CatalogSource


@pytest.fixture(autouse=True)
def controller_credentials(monkeypatch):
    for name in controller.CONTROLLER_SECRET_NAMES:
        monkeypatch.setenv(name, "fixture-" + name)


@pytest.mark.parametrize("serving_pool", [None, "registry"])
def test_controller_resolves_versioned_credentials_before_external_submission(tmp_path, monkeypatch, serving_pool):
    requested = []
    monkeypatch.setenv("DAYTONA_DATA_API_KEY", "fixture-data-account")

    class SecretManager:
        def access_secret_version(self, *, name):
            requested.append(name)
            return SimpleNamespace(payload=SimpleNamespace(data=b"fixture-value"))

    sdk = SimpleNamespace(SecretManagerServiceClient=SecretManager)
    monkeypatch.setitem(sys.modules, "google.cloud.secretmanager", sdk)
    monkeypatch.setattr(google.cloud, "secretmanager", sdk, raising=False)
    monkeypatch.setattr(controller, "list_iris_jobs", lambda: [])
    monkeypatch.setattr(controller, "load_credentials", lambda profile: SimpleNamespace(edge_refresh_token="fixture"))
    accepted = tmp_path / "external-submission.json"

    def scheduler_submit(command, *, check, env):
        resolved = {}
        for index, value in enumerate(command):
            if value == "--secret" and command[index + 1] in {"DAYTONA_DATA_API_KEY", "DAYTONA_RL_API_KEY"}:
                resolved[command[index + 1]] = resolve_secret_spec(command[index + 2]).value
        accepted.write_text(
            json.dumps(
                {
                    "job_name": command[command.index("--job-name") + 1],
                    "daytona": resolved,
                    "priority": command[command.index("--priority") + 1],
                    "failure_retries": command[command.index("--max-retries") + 1],
                    "unlimited_gpu_hours": "--unlimited-gpu-hours" in command,
                    "excluded_source": command[command.index("--exclude-source") + 1],
                }
            )
        )

    monkeypatch.setattr(controller.subprocess, "run", scheduler_submit)
    controller.launch_controller(
        "ledger",
        "applet",
        False,
        frozenset(),
        "generation",
        None if serving_pool else 800,
        serving_pool=serving_pool,
        excluded_sources=frozenset({"held-source"}),
    )
    assert requested == [
        "projects/hai-gcp-models/secrets/DAYTONA_RL_API_KEY/versions/1",
    ]
    assert json.loads(accepted.read_text())["job_name"].startswith("atlas-difficulty-reconciler-controller-")
    assert json.loads(accepted.read_text())["daytona"]["DAYTONA_DATA_API_KEY"] == "fixture-data-account"
    submitted = json.loads(accepted.read_text())
    assert submitted["priority"] == ("production" if serving_pool else "interactive")
    assert submitted["failure_retries"] == ("3" if serving_pool else "0")
    assert submitted["unlimited_gpu_hours"] == bool(serving_pool)
    assert submitted["excluded_source"] == "held-source"


@pytest.mark.parametrize("default_parent", [False, True], ids=["controller-launch", "reproduce-default-parent"])
def test_controller_can_delegate_child_profile_through_iris_client(tmp_path, monkeypatch, default_parent):
    if not finelog_native_available():
        pytest.skip("finelog native server extension not available")
    with ExitStack() as stack:
        server = require_embedded_server()(log_dir=str(tmp_path / "logs"))
        stack.callback(server.stop)
        logs = LogClient.connect(server.address)
        stack.callback(logs.close)
        state = stack.enter_context(make_controller_state())
        service = ControllerServiceImpl(
            controller=MockController(),
            bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
            log_client=logs,
            db=state._db,
            auth=ControllerAuth(provider="static"),
            endpoint_service=EndpointServiceImpl(db=state._db),
        )
        parent = JobName.root("fixture", controller.CONTROLLER_JOB_PREFIX + "generation")

        # Replace only RPC transport; keep both production client layers and controller authorization.
        class Transport:
            def list_jobs(self, request):
                return service.list_jobs(request, None)

            def launch_job(self, request, *, timeout_ms):
                return service.launch_job(request, None)

            def close(self):
                pass

        remote = RemoteClusterClient("http://unused", bundle_id="fixture-bundle")
        remote._client.close()
        remote._client = Transport()
        client = IrisClient(remote)
        stack.callback(client.shutdown)
        monkeypatch.setattr(IrisClient, "remote", lambda *args, **kwargs: client)
        monkeypatch.setattr("iris.cli.job.resolve_secret_spec", lambda source: SimpleNamespace(value="fixture"))

        def scheduler_submit(command, *, check, env):
            args = command[command.index("job") + 2 :]
            # This controller is local; the production peer transport is outside this authorization test.
            index = args.index("--target-cluster=cw-rno2a")
            del args[index]
            if default_parent:
                index = args.index("--container-profile")
                del args[index : index + 2]
            with identity_scope(VerifiedIdentity(user_id="fixture", role="admin")):
                result = CliRunner().invoke(
                    run,
                    ["--user", "fixture", *args],
                    obj={"controller_url": "http://unused", "config": None, "credentials": None},
                    env=env,
                )
            assert result.exit_code == 0, result.output + repr(result.exception)

        monkeypatch.setattr(controller, "list_iris_jobs", lambda: [])
        monkeypatch.setattr(
            controller, "load_credentials", lambda profile: SimpleNamespace(edge_refresh_token="fixture")
        )
        monkeypatch.setattr(controller, "subprocess", SimpleNamespace(run=scheduler_submit))
        controller.launch_controller(
            "ledger", "applet", False, frozenset(), "generation", 800, excluded_sources=frozenset()
        )
        backend = IrisBackend(client, parent, str(tmp_path / "ledger"))
        action = RebuildEnvironmentArtifact("s3://fixture/artifact", (), ())
        manifest = "s3://fixture/manifest.json"
        identity = VerifiedIdentity(user_id="fixture", role="task", job_id=str(parent))
        with identity_scope(identity), iris_ctx_scope(IrisContext(job_id=parent, client=client)):
            if default_parent:
                with pytest.raises(ConnectError) as denied:
                    backend.submit(action, manifest)
                assert denied.value.code == Code.PERMISSION_DENIED
            else:
                child = backend.submit(action, manifest)
                assert child.startswith(str(parent) + "/difficulty-reconcile-")
                assert backend.submit(action, manifest) == child
            children = service.list_jobs(
                controller_pb2.Controller.ListJobsRequest(
                    query=controller_pb2.Controller.JobQuery(job_id_prefix=str(parent) + "/")
                ),
                None,
            ).jobs
        assert len(children) == (0 if default_parent else 1)


def test_controller_recovers_catalog_connection_failure_before_reconciling(tmp_path, monkeypatch):
    source = CatalogSource("fixture", "revision", (("task", "sha"),), "queue", "digest")
    catalog_calls, reconciled, sleeps = [], [], []

    def catalog(*args, **kwargs):
        catalog_calls.append(True)
        if len(catalog_calls) == 1:
            raise requests.ConnectionError("[Errno 101] Network is unreachable")
        return [source, CatalogSource("held", "revision", (("task", "sha"),), "queue", "digest")]

    monkeypatch.setattr(controller, "iris_ctx", lambda: SimpleNamespace(client=None, job_id="fixture"))
    monkeypatch.setattr(controller, "IrisBackend", lambda *args: SimpleNamespace(jobs=lambda: []))
    monkeypatch.setattr(controller, "require_secret", lambda name: "fixture-refresh")
    monkeypatch.setattr(controller, "served_catalog", catalog)
    monkeypatch.setattr(controller, "read_rows", lambda ledger: [])
    monkeypatch.setattr(controller, "available_artifact_uris", lambda cells: frozenset())
    monkeypatch.setattr(controller.time, "sleep", sleeps.append)
    monkeypatch.setattr(controller, "reconcile_once", lambda ledger, sources, *args: reconciled.append(sources) or {})
    controller.run_controller(str(tmp_path), "fixture", True, frozenset(), 800, excluded_sources=frozenset({"held"}))
    assert reconciled == [[source]]
    assert sleeps == [controller.CONTROLLER_INTERVAL]


def test_launcher_loads_provider_credentials_from_explicit_file(tmp_path, monkeypatch):
    names = ("GLM_BULK_TOKEN", "HF_TOKEN", "DAYTONA_DATA_API_KEY")
    secret_file = tmp_path / "credentials.env"
    secret_file.write_text("\n".join(f"{name}=file-{name}" for name in names))
    for name in names:
        monkeypatch.delenv(name)
    accepted = []
    monkeypatch.setattr(controller, "list_iris_jobs", lambda: [])
    monkeypatch.setattr(controller, "load_credentials", lambda _: SimpleNamespace(edge_refresh_token="fixture"))
    monkeypatch.setattr(
        controller.subprocess, "run", lambda command, **kwargs: accepted.append((command, kwargs["env"]))
    )
    controller.launch_controller(
        "ledger", "applet", False, frozenset(), "generation", 800, secret_file, excluded_sources=frozenset()
    )
    ((command, environment),) = accepted
    assert {name: environment[name] for name in names} == {name: f"file-{name}" for name in names}
    assert not any("file-" in argument for argument in command)


def test_tpu_controller_waits_for_final_pool_allocation_before_submitting_work(tmp_path, monkeypatch):
    pools = tmp_path / "pools"
    pools.mkdir()
    for solver in controller.SELF_HOSTED_SOLVERS:
        (pools / (solver.value + ".json")).write_text("{}")
    sleeps, reconciled = [], []

    def pool(solver, registry):
        chips = 128 if solver == Solver.LARGE and sleeps else 16 if solver == Solver.LARGE else 64
        return {"platform": "tpu", "region": "us-central2", "chip_count": chips}

    monkeypatch.setattr(controller, "iris_ctx", lambda: SimpleNamespace(client=None, job_id="fixture"))
    monkeypatch.setattr(controller, "IrisBackend", lambda *args: SimpleNamespace(jobs=lambda: []))
    monkeypatch.setattr(controller, "require_secret", lambda name: "fixture-refresh")
    monkeypatch.setattr(controller, "campaign_record", lambda ledger: {"holds": []})
    monkeypatch.setattr(controller, "solver_pool_record", pool)
    monkeypatch.setattr(controller, "served_catalog", lambda *args, **kwargs: [])
    monkeypatch.setattr(controller, "read_rows", lambda ledger: [])
    monkeypatch.setattr(controller, "available_artifact_uris", lambda cells: frozenset())
    monkeypatch.setattr(controller.time, "sleep", sleeps.append)
    monkeypatch.setattr(controller, "reconcile_once", lambda *args: reconciled.append(True) or {})
    controller.run_controller(
        str(tmp_path), "fixture", True, frozenset(), None, str(pools), excluded_sources=frozenset()
    )
    assert sleeps == [controller.CONTROLLER_INTERVAL]
    assert reconciled == [True]
