# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from omegaconf import OmegaConf
from skyrlrecipe.locality import LocalityPlan

from cloud.iris.iris_backend import BundledLaunchConfig, RLStoragePaths, launch
from cloud.iris.ray_storage import RaySpillBackend
from cloud.iris.runtime_environment import RuntimeProfile
from cloud.iris.role_topology import RoleTopology


@pytest.mark.parametrize(
    "target_cluster,parent_cluster_config,interface,groups",
    [
        ("cw-rno2a", "marin.yaml", None, ()),
        (None, "marin.yaml", None, ()),
        (None, None, None, ()),
        ("cw-rno2a", "marin.yaml", "eth0", ()),
        ("cw-us-east-08a", "marin.yaml", None, (8, 16)),
        ("cw-us-east-08a", "marin.yaml", None, (16, 16, 16)),
    ],
)
def test_nested_launch_uses_parent_controller_and_execution_cluster_defaults(
    tmp_path, monkeypatch, target_cluster, parent_cluster_config, interface, groups
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
                "entrypoint": "skyrl_train.entrypoints.main_base",
                "task_env": {} if interface is None else {"NCCL_SOCKET_IFNAME": interface},
            },
            "skyrl": {"trainer": {"debug_mode": "off"}},
        }
    )
    args = SimpleNamespace(
        target_cluster=target_cluster,
        parent_cluster_config=parent_cluster_config,
        cluster=target_cluster or "cw-rno2a",
        cluster_config="unused-peer-config.yaml",
        job_name="train",
        config=config,
        launch_config="launch.yaml",
        launch_config_bundle=BundledLaunchConfig("/app/launch.yaml", "e30="),
        model_path="s3://models/policy",
        gpu_variant="GB200" if groups else "H100",
        gpus_per_node=4 if groups else 8,
        num_nodes=sum(groups) if groups else 2,
        role_topology=RoleTopology(("policy",), LocalityPlan(groups, (0,)), True) if groups else None,
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
    assert tuple(request.coscheduling.group_sizes) == groups
    assert request.coscheduling.group_by == ("nvlink.domain" if groups else "leafgroup")
    assert all(constraint.key != CLUSTER_CONSTRAINT_KEY for constraint in request.constraints)
    expected_interface = interface or ("^ibs,ibp,lo,docker,veth,cilium,lxc" if args.cluster == "cw-rno2a" else None)
    assert request.environment.env_vars.get("NCCL_SOCKET_IFNAME") == expected_interface
