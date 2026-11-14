# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from omegaconf import OmegaConf

from cloud.iris.iris_backend import BundledLaunchConfig, RLStoragePaths, launch
from cloud.iris.ray_storage import RaySpillBackend
from cloud.iris.runtime_environment import CHECKPOINT_EXPORT_ENTRYPOINT, RuntimeProfile


@pytest.mark.parametrize(
    "target_cluster,parent_cluster_config,interface,logger,entrypoint,preemption_retries",
    [
        ("cw-rno2a", "marin.yaml", None, "records", "skyrl_train.entrypoints.main_base", 0),
        (None, "marin.yaml", None, "console", "skyrl_train.entrypoints.main_base", 1000),
        (None, None, None, "console", "skyrl_train.entrypoints.main_base", 1000),
        ("cw-rno2a", "marin.yaml", "eth0", "records", "skyrl_train.entrypoints.main_base", 0),
        (None, "marin.yaml", None, "records", CHECKPOINT_EXPORT_ENTRYPOINT, 1000),
    ],
)
def test_nested_launch_uses_parent_controller_and_execution_cluster_defaults(
    tmp_path, monkeypatch, target_cluster, parent_cluster_config, interface, logger, entrypoint, preemption_retries
):
    requests = []

    class ControllerStub:
        def __init__(self, *, address, **kwargs):
            self.address = address

        def launch_job(self, request, *, timeout_ms=None):
            requests.append((self.address, request))
            return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

        def close(self):
            pass

    # Replace only the controller RPC transport; launch and both Iris client
    # layers still construct the real child submission and environment.
    monkeypatch.setattr("iris.cluster.client.remote_client.ControllerServiceClientSync", ControllerStub)
    monkeypatch.setenv("IRIS_CONTROLLER_URL", "http://parent-controller:8080")
    config = OmegaConf.create(
        {
            "runtime": {
                "entrypoint": entrypoint,
                "task_env": {} if interface is None else {"NCCL_SOCKET_IFNAME": interface},
            },
            "skyrl": {"trainer": {"debug_mode": "off", "logger": [logger]}},
        }
    )
    args = SimpleNamespace(
        entrypoint=entrypoint,
        target_cluster=target_cluster,
        parent_cluster_config=parent_cluster_config,
        cluster="cw-rno2a",
        cluster_config="unused-peer-config.yaml",
        job_name="train",
        config=config,
        launch_config="launch.yaml",
        launch_config_bundle=BundledLaunchConfig("/app/launch.yaml", "e30="),
        model_path="s3://models/policy",
        gpu_variant="H100",
        gpus_per_node=8,
        num_nodes=2,
        cpu=32,
        memory="512GB",
        disk="512GB",
        runtime_profile=RuntimeProfile.MEGATRON,
        priority="interactive",
        storage_paths=RLStoragePaths("checkpoints", "exports", "trajectories", "logs", 2),
        ray_spill_backend=RaySpillBackend.LOCAL,
        ray_spill_dir="/tmp/ray-spill",
        rendezvous_dir="s3://runs/rendezvous",
        dry_run=False,
        no_wait=True,
        wandb_entity=None,
        container_profile=job_pb2.CONTAINER_PROFILE_UNSPECIFIED,
        max_retries=0,
        timeout=600,
    )
    previous = get_job_info()
    set_job_info(JobInfo(task_id=JobName.from_wire("/alice/coordinator/0")))
    try:
        result = launch(args, tmp_path)
    finally:
        set_job_info(previous)

    assert result.job_id == "/alice/coordinator/train"
    [(address, request)] = requests
    assert address == "http://parent-controller:8080"
    assert request.name == result.job_id
    assert (request.max_retries_failure, request.max_task_failures, request.max_retries_preemption) == (
        0,
        0,
        preemption_retries,
    )
    assert all(constraint.key != CLUSTER_CONSTRAINT_KEY for constraint in request.constraints)
    assert request.environment.env_vars["NCCL_SOCKET_IFNAME"] == (
        "^ibs,ibp,lo,docker,veth,cilium,lxc" if interface is None else interface
    )
