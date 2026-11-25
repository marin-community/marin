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
from finestore.rl.difficulty import SOLVER_MODELS, Solver
from iris.cli.job import run
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.bundle import BundleStore
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.constraints import AttributeValue, Constraint, evaluate_constraint
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import JobName
from iris.rpc import controller_pb2
from iris.testing.controller import MockController, make_controller_state
from rigging.secrets import resolve_secret_spec
from rigging.server_auth import VerifiedIdentity, identity_scope

from experiments.post_training.task_curation.difficulty import controller, iris_backend, runtime
from experiments.post_training.task_curation.difficulty.iris_backend import IrisBackend
from experiments.post_training.task_curation.difficulty.planner import (
    RebuildEnvironmentArtifact,
    action_id,
    action_payload,
)
from experiments.post_training.task_curation.difficulty.state import CatalogSource
from experiments.post_training.task_curation.difficulty_protocol import PINS


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
    monkeypatch.setattr(
        controller,
        "list_iris_jobs",
        lambda: [
            SimpleNamespace(
                job_id="/fixture/atlas-difficulty-reconciler-controller-old/difficulty-reconcile-live", active=True
            )
        ],
    )
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
@pytest.mark.parametrize("excluded_cpu_tpu_names", [(), ("blocked-slice",)])
def test_controller_can_delegate_child_profile_through_iris_client(
    tmp_path, monkeypatch, default_parent, excluded_cpu_tpu_names
):
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
        if excluded_cpu_tpu_names:
            monkeypatch.setattr(iris_backend, "campaign_record", lambda ledger: {})
        controller.launch_controller(
            "ledger",
            "applet",
            False,
            frozenset(),
            "generation",
            800,
            excluded_sources=frozenset(),
            excluded_cpu_tpu_names=excluded_cpu_tpu_names,
        )
        parent_request = service.get_job_status(
            controller_pb2.Controller.GetJobStatusRequest(job_id=str(parent)), None
        ).request
        parent_args = list(parent_request.entrypoint.run_command.argv)
        if excluded_cpu_tpu_names:
            assert parent_args[parent_args.index("--exclude-cpu-tpu-name") + 1] == "blocked-slice"
        else:
            assert "--exclude-cpu-tpu-name" not in parent_args
        backend = IrisBackend(client, parent, str(tmp_path / "ledger"), excluded_cpu_tpu_names=excluded_cpu_tpu_names)
        action = RebuildEnvironmentArtifact("s3://fixture/artifact", (), ())
        manifest_path = tmp_path / "ledger/v1/actions" / (action_id(action) + ".json")
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps(action_payload(action)))
        manifest = str(manifest_path)
        identity = VerifiedIdentity(user_id="fixture", role="task", job_id=str(parent))
        with identity_scope(identity), iris_ctx_scope(IrisContext(job_id=parent, client=client)):
            if default_parent:
                with pytest.raises(ConnectError) as denied:
                    backend.submit(action, manifest)
                assert denied.value.code == Code.PERMISSION_DENIED
            else:
                child = backend.submit(action, manifest)
                assert child.startswith(str(parent) + "/difficulty-reconcile-")
                request = service.get_job_status(
                    controller_pb2.Controller.GetJobStatusRequest(job_id=child), None
                ).request
                exclusions = [Constraint.from_proto(c) for c in request.constraints if c.key == "tpu-name"]
                assert len(exclusions) == len(excluded_cpu_tpu_names)
                for constraint in exclusions:
                    assert not constraint.is_soft
                    assert not evaluate_constraint(AttributeValue("blocked-slice"), constraint)
                    assert evaluate_constraint(AttributeValue("healthy-slice"), constraint)
                assert backend.submit(action, manifest) == child
                other = IrisBackend(client, parent, str(tmp_path / "other-ledger"))
                other_manifest = tmp_path / "other-ledger/v1/actions" / manifest_path.name
                other_manifest.parent.mkdir(parents=True)
                other_manifest.write_text(manifest_path.read_text())
                other_child = other.submit(action, str(other_manifest))
                assert other_child != child
                assert {job.job_id for job in backend.jobs()} == {child}
                assert {job.job_id for job in other.jobs()} == {other_child}
                restarted = IrisBackend(
                    client, JobName.root("fixture", controller.CONTROLLER_JOB_PREFIX + "next"), backend.ledger
                )
                assert restarted.submit(action, manifest) == child
            children = service.list_jobs(
                controller_pb2.Controller.ListJobsRequest(
                    query=controller_pb2.Controller.JobQuery(job_id_prefix=str(parent) + "/")
                ),
                None,
            ).jobs
        assert len(children) == (0 if default_parent else 2)


def test_cpu_slice_exclusions_reject_a_nonregional_campaign(tmp_path):
    client = SimpleNamespace(list_jobs=lambda **kwargs: [])
    backend = IrisBackend(
        client,
        JobName.root("fixture", controller.CONTROLLER_JOB_PREFIX + "generation"),
        str(tmp_path),
        excluded_cpu_tpu_names=("blocked-slice",),
    )
    with pytest.raises(ValueError, match="regional TPU campaign"):
        backend.submit(RebuildEnvironmentArtifact("s3://fixture/artifact", (), ()), "manifest")


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


def test_tpu_controller_admits_ready_pools_then_large_after_its_full_receipt(tmp_path, monkeypatch):
    pools = tmp_path / "pools"
    pools.mkdir()
    for solver in controller.SELF_HOSTED_SOLVERS:
        (pools / (solver.value + ".json")).write_text("{}")
    reconciled = []

    def pool(solver, registry):
        chips = 128 if solver == Solver.LARGE and reconciled else 32 if solver == Solver.LARGE else 64
        return {"platform": "tpu", "region": "us-central2", "chip_count": chips}

    def reconcile(ledger, catalog, policy, backend, notifications, capacity, *args):
        reconciled.append(capacity)
        if len(reconciled) == 2:
            raise StopIteration
        return {}

    monkeypatch.setattr(controller, "iris_ctx", lambda: SimpleNamespace(client=None, job_id="fixture"))
    monkeypatch.setattr(controller, "IrisBackend", lambda *args: SimpleNamespace(jobs=lambda: []))
    monkeypatch.setattr(controller, "require_secret", lambda name: "fixture-refresh")
    monkeypatch.setattr(controller, "campaign_record", lambda ledger: {"holds": []})
    monkeypatch.setattr(controller, "solver_pool_record", pool)
    monkeypatch.setattr(controller, "_tpu_pool_is_live", lambda pool: True)
    monkeypatch.setattr(controller, "served_catalog", lambda *args, **kwargs: [])
    monkeypatch.setattr(controller, "read_rows", lambda ledger: [])
    monkeypatch.setattr(controller, "available_artifact_uris", lambda cells: frozenset())
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(controller, "reconcile_once", reconcile)
    with pytest.raises(StopIteration):
        controller.run_controller(
            str(tmp_path), "fixture", False, frozenset(), None, str(pools), excluded_sources=frozenset()
        )
    assert reconciled == [frozenset({Solver.SMALL, Solver.SNOWBALL, Solver.HOSTED}), frozenset(Solver)]


@pytest.mark.parametrize("failure", ["withdrawn", "http503", "wrong-model"])
def test_tpu_controller_withholds_stale_pool_and_readmits_it_after_recovery(
    tmp_path, monkeypatch, review_server, failure
):
    pools = tmp_path / "pools"
    pools.mkdir()
    receipt = {
        "solver": Solver.SMALL.value,
        "model": SOLVER_MODELS[Solver.SMALL],
        "revision": PINS[Solver.SMALL].revision,
        "platform": "tpu",
        "region": "us-central2",
        "chip_count": 64,
        "endpoint_name": "/serve/frozen-small-generation",
    }
    path = pools / "small.json"
    original = json.dumps(receipt)
    path.write_text(original)
    snowball = {
        **receipt,
        "solver": Solver.SNOWBALL.value,
        "model": SOLVER_MODELS[Solver.SNOWBALL],
        "revision": PINS[Solver.SNOWBALL].revision,
        "endpoint_name": "/serve/frozen-snowball-generation",
    }
    snowball_path = pools / "snowball.json"
    snowball_original = json.dumps(snowball)
    snowball_path.write_text(snowball_original)
    review_server.model_ids[:] = [receipt["model"], snowball["model"]]
    broken = False
    original_get = review_server.RequestHandlerClass.do_GET

    def get(handler):
        if broken and failure == "http503" and handler.path == "/small/v1/models":
            handler.send_error(503)
        else:
            original_get(handler)

    def resolve(name):
        assert name in {receipt["endpoint_name"], snowball["endpoint_name"]}
        if broken and failure == "withdrawn" and name == receipt["endpoint_name"]:
            raise ConnectError(Code.NOT_FOUND, "Pool owner has withdrawn its endpoint")
        solver = "small" if name == receipt["endpoint_name"] else "snowball"
        return f"http://127.0.0.1:{review_server.server_port}/{solver}"

    monkeypatch.setattr(review_server.RequestHandlerClass, "do_GET", get)
    monkeypatch.setattr(runtime, "iris_ctx", lambda: SimpleNamespace(client=SimpleNamespace(resolve_endpoint=resolve)))
    expected = frozenset({Solver.SMALL, Solver.SNOWBALL, Solver.HOSTED})
    assert controller.qualified_tpu_solvers(str(pools)) == expected
    broken = True
    if failure == "wrong-model":
        review_server.model_ids.remove(receipt["model"])
    assert controller.qualified_tpu_solvers(str(pools)) == frozenset({Solver.SNOWBALL, Solver.HOSTED})
    assert path.read_text() == original
    assert snowball_path.read_text() == snowball_original
    broken = False
    review_server.model_ids[:] = [receipt["model"], snowball["model"]]
    assert controller.qualified_tpu_solvers(str(pools)) == expected
    assert path.read_text() == original


def test_explicit_controller_handover_preserves_live_children_and_refuses_third_parent(tmp_path, monkeypatch):
    old = "/fixture/atlas-difficulty-reconciler-controller-old"
    new = "/fixture/atlas-difficulty-reconciler-controller-new"
    active = [SimpleNamespace(job_id=old, state="running", active=True)]
    submitted = []
    monkeypatch.setattr(controller, "list_iris_jobs", lambda: active)
    monkeypatch.setattr(controller, "campaign_record", lambda ledger: {"holds": []})
    monkeypatch.setattr(controller, "load_credentials", lambda profile: SimpleNamespace(edge_refresh_token="fixture"))
    monkeypatch.setattr(controller.subprocess, "run", lambda command, **kwargs: submitted.append(command))
    args = (str(tmp_path), "applet", False, frozenset(), "new", None)
    controller.launch_controller(*args, excluded_sources=frozenset())
    assert submitted == []
    controller.launch_controller(*args, excluded_sources=frozenset(), handover_from=old)
    assert len(submitted) == 1
    assert submitted[0][submitted[0].index("--job-name") + 1] == new.rsplit("/", 1)[-1]
    assert not any("cancel" in word for command in submitted for word in command)
    active.append(SimpleNamespace(job_id=new, state="running", active=True))
    controller.launch_controller(*args, excluded_sources=frozenset(), handover_from=old)
    assert len(submitted) == 1
    active.append(
        SimpleNamespace(job_id="/fixture/atlas-difficulty-reconciler-controller-unrelated", state="running", active=True)
    )
    with pytest.raises(ValueError, match="unrelated active controller"):
        controller.launch_controller(*args, excluded_sources=frozenset(), handover_from=old)
    assert len(submitted) == 1
