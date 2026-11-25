# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from iris.client.client import IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY
from iris.cluster.types import JobName
from iris.rpc import controller_pb2
from omegaconf import OmegaConf

from cloud.iris import launch
from cloud.iris.export_hf_checkpoint import ExportJobSpec, checkpoint_export_launch_config
from cloud.iris.launch_config import load_launch_config
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from cloud.iris.runtime_environment import CHECKPOINT_EXPORT_ENTRYPOINT
from skyrl_train.hf_export_schema import HFExportRequest


@pytest.mark.parametrize(
    "target_cluster,parent_cluster_config,interface,logger,entrypoint,preemption_retries,groups",
    [
        ("cw-rno2a", "marin.yaml", None, "records", "skyrl_train.entrypoints.main_base", 0, ()),
        (None, "marin.yaml", None, "console", "skyrl_train.entrypoints.main_base", 1000, ()),
        (None, None, None, "console", "skyrl_train.entrypoints.main_base", 1000, ()),
        ("cw-rno2a", "marin.yaml", "eth0", "records", "skyrl_train.entrypoints.main_base", 0, ()),
        (None, "marin.yaml", None, "records", CHECKPOINT_EXPORT_ENTRYPOINT, 1000, ()),
        ("cw-us-east-08a", "marin.yaml", None, "console", "skyrl_train.entrypoints.main_base", 1000, (16, 8)),
    ],
)
def test_nested_launch_uses_parent_controller_and_execution_cluster_defaults(
    tmp_path,
    monkeypatch,
    capsys,
    target_cluster,
    parent_cluster_config,
    interface,
    logger,
    entrypoint,
    preemption_retries,
    groups,
):
    requests = []
    monkeypatch.setenv("HF_TEST_KEY", "test-extra-credential")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")

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
    document = raw_launch_config()
    document["run"].update(submission="detach", export_hf=False)
    document["runtime"]["task_env"] = {} if interface is None else {"NCCL_SOCKET_IFNAME": interface}
    document["iris"].update(
        cluster=target_cluster or "cw-rno2a",
        cluster_config="unused-peer-config.yaml",
        job_name="train",
        priority="interactive",
        target_cluster=target_cluster,
        parent_cluster_config=parent_cluster_config,
        max_retries=0,
        timeout=600,
    )
    document["iris"]["allocation"].update(num_nodes=2, cpu=32, memory="512GB", disk="512GB")
    document["skyrl"]["trainer"]["logger"] = [logger]
    document["skyrl"]["trainer"].update(train_batch_size=16, policy_mini_batch_size=16)
    document["skyrl"]["generator"]["num_inference_engines"] = 16
    document["skyrl"]["trainer"]["placement"].update(policy_num_nodes=2, ref_num_nodes=2)
    if groups:
        document["iris"]["role_topology"] = "nvlink.domain"
        document["iris"]["allocation"].update(num_nodes=24, gpus_per_node=4, gpu_variant="GB200")
        document["skyrl"]["trainer"].update(train_batch_size=64, policy_mini_batch_size=64)
        document["skyrl"]["trainer"]["placement"].update(
            colocate_all=False, policy_num_nodes=16, policy_num_gpus_per_node=4
        )
        document["skyrl"]["generator"].update(
            num_inference_engines=1, inference_engine_data_parallel_size=32, inference_engine_expert_parallel_size=32
        )
    for field in document["artifacts"]:
        document["artifacts"][field] = str(tmp_path / field)
    path = tmp_path / "launch.yaml"
    OmegaConf.save(OmegaConf.create(document), path)
    config = load_launch_config(path)
    if entrypoint == CHECKPOINT_EXPORT_ENTRYPOINT:
        request = HFExportRequest(
            step=7,
            checkpoint_base_path=str(tmp_path / "checkpoints"),
            checkpoint_path=str(tmp_path / "checkpoints/global_step_7"),
            export_path=str(tmp_path / "exports"),
            model_path="org/policy",
            num_nodes=2,
            gpus_per_node=8,
        )
        spec = ExportJobSpec(
            request=request,
            cluster="cw-rno2a",
            priority="interactive",
            gpu_variant="H100",
            job_name="train",
            timeout=600,
            no_wait=True,
        )
        config = checkpoint_export_launch_config(config, request, spec)
        OmegaConf.save(config, path)
    previous = get_job_info()
    info = JobInfo(task_id=JobName.from_wire("/alice/coordinator/0"))
    set_job_info(info)
    try:
        with iris_ctx_scope(IrisContext.from_job_info(info)):
            assert launch.main(["iris", "launch", "--config", str(path), "--forward-env", "HF_TEST_KEY"]) == 0
    finally:
        set_job_info(previous)

    captured = capsys.readouterr()
    result = json.loads(captured.out.splitlines()[-1])
    assert result["iris_job_id"] == "/alice/coordinator/train"
    assert "test-extra-credential" not in captured.out + captured.err
    [(address, request)] = requests
    assert address == "http://parent-controller:8080"
    assert request.environment.env_vars["HF_TEST_KEY"] == "test-extra-credential"
    assert "UNRELATED_SECRET" not in request.environment.env_vars
    assert "test-extra-credential" not in OmegaConf.to_yaml(config)
    assert request.name == result["iris_job_id"]
    assert request.resources.disk_bytes == 512 * 1024**3
    assert (request.max_retries_failure, request.max_task_failures, request.max_retries_preemption) == (
        0,
        0,
        preemption_retries,
    )
    assert tuple(request.coscheduling.group_sizes) == groups
    assert request.coscheduling.group_by == ("nvlink.domain" if groups else "leafgroup")
    assert all(constraint.key != CLUSTER_CONSTRAINT_KEY for constraint in request.constraints)
    expected_interface = interface or (
        "^ibs,ibp,lo,docker,veth,cilium,lxc" if config.iris.cluster == "cw-rno2a" else None
    )
    assert request.environment.env_vars.get("NCCL_SOCKET_IFNAME") == expected_interface

    shell = request.entrypoint.run_command.argv[-1]
    assert "nccl_transport_probe.py --mode gate --gate-coverage all-gpus" in shell
    assert "/interconnect-gate; " in shell
    assert ("--pynccl-init" in shell) == (entrypoint != CHECKPOINT_EXPORT_ENTRYPOINT)


def test_excluded_nvlink_domains_become_required_ne_constraints():
    """Each excluded rack reaches Iris as a required `nvlink.domain NE <rack>` constraint."""
    from iris.cluster.constraints import ConstraintOp
    from iris.rpc import job_pb2

    from cloud.iris.iris_backend import _gpu_constraints

    constraints = _gpu_constraints(
        job_pb2.ResourceSpecProto(), replicas=12, excluded_nvlink_domains=("dh1-r136-us-east-08a", "dh1-r137-us-east-08a")
    )
    exclusions = [constraint for constraint in constraints if constraint.key == "nvlink.domain"]
    assert [constraint.values[0].value for constraint in exclusions] == [
        "dh1-r136-us-east-08a",
        "dh1-r137-us-east-08a",
    ]
    assert all(constraint.op == ConstraintOp.NE for constraint in exclusions)


def test_launch_config_carries_excluded_nvlink_domains_into_the_submission_state(tmp_path):
    from cloud.iris.iris_backend import _iris_submission_state

    from cloud.iris.tests.launch_config_fixtures import raw_launch_config

    raw = raw_launch_config()
    raw["iris"]["excluded_nvlink_domains"] = ["dh1-r136-us-east-08a"]
    config_path = tmp_path / "launch.yaml"
    config_path.write_text(OmegaConf.to_yaml(OmegaConf.create(raw)))
    args = _iris_submission_state(config_path, load_launch_config(config_path))
    assert args.excluded_nvlink_domains == ("dh1-r136-us-east-08a",)
