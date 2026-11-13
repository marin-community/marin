"""Locality planning and production Ray placement with CPU backend workers."""

from importlib.machinery import ModuleSpec
import json
import socket
import sys
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
import ray
import yaml
from iris.client import IrisClient
from iris.client.client import IrisContext, iris_ctx_scope
from iris.cluster.client import ClusterClient
from iris.cluster.types import JobName, TaskAttempt
from iris.rpc import job_pb2
from omegaconf import OmegaConf
from ray.cluster_utils import Cluster
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from cloud.iris.launch_config import load_launch_config
from cloud.iris.role_topology import LOCALITY_GROUP_LABEL, NVLINK_DOMAIN_LABEL, derive_role_topology
from cloud.iris.skyrl_entrypoint import run_config
from cloud.iris.task_runtime import locality_node_labels
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from skyrl_train.entrypoints.main_base import BasePPOExp, create_ray_wrapped_inference_engines_from_config


def launch_config(tmp_path, *, nodes=48, policy_nodes=16, engines=2, dp=64, width=4, variant="GB200", locality=True):
    raw = raw_launch_config()
    if locality:
        raw["iris"]["role_topology"] = "nvlink.domain"
    raw["iris"]["allocation"].update(num_nodes=nodes, gpus_per_node=width, gpu_variant=variant)
    raw["skyrl"]["trainer"]["placement"].update(
        colocate_all=False, policy_num_nodes=policy_nodes, policy_num_gpus_per_node=width,
    )
    raw["skyrl"]["generator"].update(
        num_inference_engines=engines, inference_engine_data_parallel_size=dp, inference_engine_expert_parallel_size=dp,
    )
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize(
    "geometry,hosts,roles",
    [
        ({}, (16, 16, 16), (0, 1, 2)),
        ({"nodes": 24, "policy_nodes": 8, "engines": 1}, (8, 16), (0, 1)),
        ({"nodes": 24, "dp": 16}, (16, 8), (0, 1, 1)),
        ({"nodes": 32, "dp": 32}, (16, 16), (0, 1, 1)),
        ({"nodes": 6, "policy_nodes": 1, "engines": 3, "dp": 6}, (6,), (0, 0, 0, 0)),
        ({"nodes": 34, "policy_nodes": 1, "engines": 7, "dp": 18}, (15, 14, 5), (0, 0, 0, 0, 1, 1, 1, 2)),
    ],
)
def test_launch_plan_keeps_whole_models_local_and_counts_host_padding(tmp_path, geometry, hosts, roles):
    config = load_launch_config(launch_config(tmp_path, **geometry))
    topology = derive_role_topology(OmegaConf.to_container(config, resolve=True))
    assert topology.locality.group_sizes == hosts
    assert topology.locality.role_groups == roles


def test_single_domain_mode_rejects_a_trainer_that_needs_multiple_racks(tmp_path):
    path = launch_config(tmp_path, nodes=34, policy_nodes=18, engines=1)
    with pytest.raises(ValueError, match="72 GPUs; one domain holds 64"):
        load_launch_config(path)


def test_ray_labels_use_the_current_iris_attempts_topology(tmp_path, monkeypatch):
    path = launch_config(tmp_path, nodes=24, policy_nodes=8, engines=1)
    config = load_launch_config(path)
    topology = derive_role_topology(OmegaConf.to_container(config, resolve=True))
    replies = []
    for uid, domain in (("old-uid", "rack-old"), ("current-uid", "rack-current")):
        status = job_pb2.TaskStatus(task_id="/user/job/8", state=job_pb2.TASK_STATE_RUNNING)
        status.attempts.add(
            attempt_id=0, attempt_uid=uid, state=job_pb2.TASK_STATE_RUNNING,
            topology={"ds.coreweave.com/nvlink.domain": domain},
        )
        replies.append(status)

    class ControllerTransport:
        def get_task_status(self, _task_name, *, deadline=None):
            return replies.pop(0)

    client = IrisClient(cast(ClusterClient, ControllerTransport()))
    ctx = IrisContext(
        job_id=JobName.from_wire("/user/job"),
        task_attempt=TaskAttempt(JobName.from_wire("/user/job/8"), 0),
        client=client,
    )
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "current-uid")
    with iris_ctx_scope(ctx):
        labels = locality_node_labels(topology, 8, timeout=2)
    assert labels == {LOCALITY_GROUP_LABEL: "1", NVLINK_DOMAIN_LABEL: "rack-current"}


@pytest.mark.slow
@pytest.mark.timeout(180)
@pytest.mark.parametrize("locality,dp", [(True, 1), (False, 1), (True, 4)])
def test_production_factories_constrain_workers_and_preserve_default_placement(tmp_path, monkeypatch, locality, dp):
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    path = launch_config(
        tmp_path, nodes=2, policy_nodes=1, engines=1 if dp > 1 else 2, dp=dp, width=8, variant="H100", locality=locality
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
    raw = yaml.safe_load(path.read_text())
    raw["inputs"]["model"]["uri"] = str(model)
    raw["inputs"]["model"]["local_path"] = str(model)
    raw["inputs"]["model"]["tokenizer_uri"] = str(model)
    raw["skyrl"]["generator"]["enforce_eager"] = False
    path.write_text(yaml.safe_dump(raw))
    captured = []
    monkeypatch.setattr("skyrl_train.entrypoints.main_base.run", captured.append)
    run_config(path)
    [config] = captured

    cluster = Cluster()
    try:
        # Start the inference group first, so unconstrained policy placement
        # would take its capacity. CPU Ray nodes simulate the group capacity.
        engine_nodes = 2 if dp > 1 else 1
        for _ in range(engine_nodes):
            cluster.add_node(
                num_cpus=8, num_gpus=2 if dp > 1 else 8,
                labels={LOCALITY_GROUP_LABEL: "1"} if locality else {},
                object_store_memory=80 * 1024**2, include_dashboard=False,
            )
        cluster.add_node(
            num_cpus=8, num_gpus=8, labels={LOCALITY_GROUP_LABEL: "0"} if locality else {},
            object_store_memory=80 * 1024**2, include_dashboard=False,
        )
        ray.init(address=cluster.address)

        @ray.remote(num_cpus=1, num_gpus=1)
        class BackendWorker:
            def __init__(self, data_parallel_rank=0, data_parallel_size=1, **_kwargs):
                self.rank, self.size = data_parallel_rank, data_parallel_size

            def identity(self):
                pg = ray.util.get_current_placement_group()
                return {"node": ray.get_runtime_context().get_node_id(), "gpus": ray.get_gpu_ids(), "pg": pg.id.hex()}

            def report_engine_hosts(self):
                return [ray.util.get_node_ip_address()]

            def report_engine_placement(self):
                identity = self.identity()
                return [{
                    "host": socket.gethostname(), "gpu_uuid": f"cpu-{identity['node']}-{identity['gpus'][0]}",
                    "dp_rank": self.rank, "dp_world_size": self.size,
                    "ep_rank": self.rank, "ep_world_size": self.size,
                    "torch_rank": self.rank, "torch_world_size": self.size,
                }]

            def get_model_max_len(self):
                return 64

        backend = ModuleType("vllm")
        backend.__spec__, backend.__version__ = ModuleSpec("vllm", loader=None), "0.0.dev"
        registry = ModuleType("vllm.model_executor.models")
        registry.ModelRegistry = SimpleNamespace(get_supported_archs=lambda: ["LlamaForCausalLM"])
        actor_module = ModuleType("skyrl_train.inference_engines.vllm.vllm_engine")
        actor_module.AsyncVLLMRayActor = BackendWorker
        monkeypatch.setitem(sys.modules, "vllm", backend)
        monkeypatch.setitem(sys.modules, registry.__name__, registry)
        monkeypatch.setitem(sys.modules, actor_module.__name__, actor_module)

        policy_pg = BasePPOExp.get_policy_pg(SimpleNamespace(cfg=config), timeout=60)
        policy_actors = []
        if locality:
            handles = [BackendWorker.options(
                num_gpus=4,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=policy_pg, placement_group_bundle_index=0,
                ),
            ).remote() for _ in range(2)]
            policy_actors = ray.get([handle.identity.remote() for handle in handles], timeout=60)
            policy_nodes = {node["NodeID"] for node in ray.nodes() if node["Labels"].get(LOCALITY_GROUP_LABEL) == "0"}
            assert {actor["node"] for actor in policy_actors} == policy_nodes
        else:
            assert policy_pg is None

        engines = create_ray_wrapped_inference_engines_from_config(config, None)
        inference_actors = ray.get([engine.inference_engine_actor.identity.remote() for engine in engines], timeout=60)
        assert len({actor["pg"] for actor in inference_actors}) == (config.generator.num_inference_engines if locality else 1)
        if locality:
            allowed = {node["NodeID"] for node in ray.nodes() if node["Labels"].get(LOCALITY_GROUP_LABEL) == "1"}
            assert {actor["node"] for actor in inference_actors} <= allowed
            assert {actor["node"] for actor in policy_actors}.isdisjoint(actor["node"] for actor in inference_actors)
            assert len({actor["node"] for actor in inference_actors}) == engine_nodes
    finally:
        ray.shutdown()
        cluster.shutdown()
