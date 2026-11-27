# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
from pathlib import Path

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
from cloud.iris.rl_config_translation import RL_CONFIG_PAYLOAD_ENV, RL_CONFIG_TASK_DIR
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from cloud.iris.runtime_environment import CHECKPOINT_EXPORT_ENTRYPOINT, MARINSKYRL_TASK_ROOT
from skyrl_train.hf_export_schema import HFExportRequest


@pytest.mark.parametrize(
    "target_cluster,parent_cluster_config,interface,logger,entrypoint,preemption_retries,groups,spill_backend",
    [
        ("cw-rno2a", "marin.yaml", None, "records", "skyrl_train.entrypoints.main_base", 0, (), "local"),
        (None, "marin.yaml", None, "console", "skyrl_train.entrypoints.main_base", 1000, (), "local"),
        (None, None, None, "console", "skyrl_train.entrypoints.main_base", 1000, (), "r2"),
        ("cw-rno2a", "marin.yaml", "eth0", "records", "skyrl_train.entrypoints.main_base", 0, (), "local"),
        (None, "marin.yaml", None, "records", CHECKPOINT_EXPORT_ENTRYPOINT, 1000, (), "local"),
        ("cw-us-east-08a", "marin.yaml", None, "console", "skyrl_train.entrypoints.main_base", 1000, (16, 8), "local"),
        ("cw-us-east-08a", "marin.yaml", None, "console", "skyrl_train.entrypoints.main_base", 1000, (15,), "local"),
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
    spill_backend,
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
    spill = tmp_path / "spill path '$literal'"
    document["ray"]["spill_backend"] = spill_backend
    if spill_backend == "local":
        document["ray"]["spill_dir"] = str(spill)
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
        document["iris"]["allocation"].update(num_nodes=sum(groups), gpus_per_node=4, gpu_variant="GB200")
        document["skyrl"]["trainer"].update(train_batch_size=64, policy_mini_batch_size=64)
        policy_nodes, inference_gpus = (12, 12) if len(groups) == 1 else (16, 32)
        document["skyrl"]["trainer"]["placement"].update(
            colocate_all=False, policy_num_nodes=policy_nodes, policy_num_gpus_per_node=4
        )
        document["skyrl"]["generator"].update(
            num_inference_engines=1,
            inference_engine_data_parallel_size=inference_gpus,
            inference_engine_expert_parallel_size=inference_gpus,
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

    assert not spill.exists()
    payload = base64.b64decode(request.environment.env_vars[RL_CONFIG_PAYLOAD_ENV])
    digest = hashlib.sha256(payload).hexdigest()[:16]
    expected_profile = "megatron-export" if entrypoint == CHECKPOINT_EXPORT_ENTRYPOINT else "megatron"
    assert request.environment.setup_scripts == [
        f"bash {MARINSKYRL_TASK_ROOT}/cloud/iris/setup_task_runtime.sh {expected_profile}"
    ]
    assert request.entrypoint.run_command.argv == [
        "bash",
        f"{MARINSKYRL_TASK_ROOT}/cloud/iris/node_entrypoint.sh",
        str(Path(RL_CONFIG_TASK_DIR) / f"{digest}.yaml"),
        *(["--pynccl-init"] if entrypoint != CHECKPOINT_EXPORT_ENTRYPOINT else []),
        *(["--gate-single-nvlink-domain"] if len(groups) == 1 else []),
    ]
