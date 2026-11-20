# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller credentials reach Iris through versioned Secret Manager references."""

import json
import sys
from contextlib import ExitStack
from types import SimpleNamespace

import google.cloud
import pytest
from connectrpc.errors import ConnectError
from finelog.client import LogClient
from finelog.embedded import is_available as finelog_native_available
from finelog.embedded import require_embedded_server
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.bundle import BundleStore
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from iris.testing.controller import MockController, make_controller_state, make_job_request
from rigging.secrets import resolve_secret_spec
from rigging.server_auth import VerifiedIdentity, identity_scope

from experiments.post_training.task_curation.difficulty import controller
from experiments.post_training.task_curation.difficulty.iris_backend import IrisBackend
from experiments.post_training.task_curation.difficulty.planner import RebuildEnvironmentArtifact


def test_controller_resolves_versioned_credentials_before_external_submission(tmp_path, monkeypatch):
    requested = []

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
        for index, value in enumerate(command):
            if value == "--secret" and command[index + 2].startswith("gcp-secret://"):
                resolve_secret_spec(command[index + 2])
        accepted.write_text(json.dumps({"job_name": command[command.index("--job-name") + 1]}))

    monkeypatch.setattr(controller.subprocess, "run", scheduler_submit)
    controller.launch_controller("ledger", "applet", False, frozenset(), "generation", 800)
    assert requested == [
        "projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/1",
        "projects/hai-gcp-models/secrets/DAYTONA_RL_API_KEY/versions/1",
    ]
    assert json.loads(accepted.read_text())["job_name"].startswith("atlas-difficulty-reconciler-controller-")


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

        def scheduler_submit(command, *, check, env):
            request = make_job_request(str(parent), priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE)
            if not default_parent and "--container-profile" in command:
                request.container_profile = job_pb2.ContainerProfile.Value(
                    command[command.index("--container-profile") + 1]
                )
            with identity_scope(VerifiedIdentity(user_id="fixture", role="admin")):
                service.launch_job(request, None)

        monkeypatch.setattr(controller, "list_iris_jobs", lambda: [])
        monkeypatch.setattr(
            controller, "load_credentials", lambda profile: SimpleNamespace(edge_refresh_token="fixture")
        )
        with monkeypatch.context() as submission:
            submission.setattr(controller.subprocess, "run", scheduler_submit)
            controller.launch_controller("ledger", "applet", False, frozenset(), "generation", 800)

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
        backend = IrisBackend(client, parent, str(tmp_path / "ledger"))
        action = RebuildEnvironmentArtifact("s3://fixture/artifact", (), ())
        identity = VerifiedIdentity(user_id="fixture", role="task", job_id=str(parent))
        with identity_scope(identity), iris_ctx_scope(IrisContext(job_id=parent, client=client)):
            if default_parent:
                with pytest.raises(ConnectError, match="set_container_profile not allowed for role task"):
                    backend.submit(action, "s3://fixture/manifest.json")
            else:
                child = backend.submit(action, "s3://fixture/manifest.json")
                assert child.startswith(str(parent) + "/difficulty-reconcile-")
                assert backend.submit(action, "s3://fixture/manifest.json") == child
            children = service.list_jobs(
                controller_pb2.Controller.ListJobsRequest(
                    query=controller_pb2.Controller.JobQuery(job_id_prefix=str(parent) + "/")
                ),
                None,
            ).jobs
        assert len(children) == (0 if default_parent else 1)
