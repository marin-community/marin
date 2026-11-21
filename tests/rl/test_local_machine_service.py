# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The parent keeps its CPU service owned through readiness and run failure."""

from dataclasses import replace

import marin.rl.local_machine_service as service
import marin.rl.services as service_launch
import pytest
from fray.current_client import set_current_client
from fray.types import ResourceConfig, create_environment
from iris.actor.server import ActorServer
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from rigging.timing import Duration
from shellbox.backends.iris.local_images import LocalImageMirror, LocalMachineSettings
from shellbox.backends.iris.machine_service import MachineServiceContract
from shellbox.machine import LINUX_PROCESS_CAPABILITIES, Backend, CommandSemantics

IMAGE = "fixture@sha256:" + "a" * 64


class DescriptorActor:
    def __init__(self, contract):
        self.contract = contract

    def describe(self):
        return self.contract


class ServiceJob:
    job_id = "/fixture/run/local-service"

    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


class ServiceClient:
    def __init__(self):
        self.job = ServiceJob()

    def submit(self, request, *, adopt_existing):
        return self.job


@pytest.mark.parametrize("body_error", [None, RuntimeError("training failed")])
def test_parent_terminates_the_verified_service_after_success_or_failure(monkeypatch, tmp_path, body_error):
    config = service.LocalMachineServiceConfig(
        local=LocalMachineSettings(
            region="US-EAST-08A",
            image_mirrors=(LocalImageMirror(image=IMAGE, uri="s3://marin-us-east-08a/fixture/oci"),),
        ),
        endpoint_name="local-service",
        slots=2,
        cpus_per_machine=4,
        memory_mb_per_machine=4096,
    )
    launch = service.LocalMachineServiceLaunch(
        config=config,
        resources=ResourceConfig(cpu=10, ram="32GB", disk="8GB"),
        environment=create_environment(),
        startup_timeout=10,
        rpc_timeout=10,
        lifetime=Duration.from_hours(1),
    )
    contract = MachineServiceContract(
        Backend.LOCAL,
        CommandSemantics.LINUX_PROCESS,
        LINUX_PROCESS_CAPABILITIES,
        (IMAGE,),
        4,
        4096,
        "US-EAST-08A",
        "x86_64",
    )
    monkeypatch.setattr(service_launch, "require_secret", lambda name: "fixture")
    info = JobInfo(task_id=JobName.from_wire("/fixture/run/0"))
    parent = ServiceClient()
    with IrisClient.in_cluster("http://controller.fixture", workspace=tmp_path) as client:
        context = IrisContext.from_job_info(info, client=client)
        actor_name = f"{context.namespace}/{config.endpoint_name}"
        server = ActorServer(host="127.0.0.1", port=0)
        actor = DescriptorActor(contract)
        server.register(actor_name, actor)
        port = server.serve_background()
        monkeypatch.setattr(service_launch, "_wait_for_endpoint", lambda *args: (f"http://127.0.0.1:{port}", {}))
        try:
            with iris_ctx_scope(context), set_current_client(parent):
                if body_error is None:
                    with service.local_machine_service(launch) as endpoint:
                        assert endpoint.images == (IMAGE,)
                else:
                    with pytest.raises(RuntimeError, match="training failed"):
                        with service.local_machine_service(launch):
                            raise body_error
                assert parent.job.terminated
                parent.job.terminated = False
                with pytest.raises(ValueError, match="differs from the run binding"):
                    actor.contract = replace(contract, region="US-EAST-02A")
                    with service.local_machine_service(launch):
                        pytest.fail("A differently placed service cannot start training")
                assert parent.job.terminated
        finally:
            server.stop()
