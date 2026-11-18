"""Inference startup preserves deadlines and verifies physical worker placement."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from ray.util.placement_group import PlacementGroup

from skyrl_train.inference_engines import placement
from skyrl_train.inference_engines import ray_wrapped_inference_engine as startup
from skyrl_train.inference_engines import utils as engine_utils
from skyrl_train.utils import utils as ray_utils


@pytest.mark.parametrize("eventually_ready", [True, False])
def test_startup_wait_preserves_pending_actors_across_progress_polls(monkeypatch, eventually_ready):
    clock = [0.0]
    replies = ["ready", "slow"]
    killed = []

    def wait(pending, *, num_returns, timeout, fetch_local):
        if pending == replies:
            return ["ready"], ["slow"]
        clock[0] += timeout
        if eventually_ready and clock[0] >= 60:
            return ["slow"], []
        return [], pending

    monkeypatch.setattr(startup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(startup.ray, "wait", wait)
    monkeypatch.setattr(startup.ray, "get", lambda refs: refs)
    monkeypatch.setattr(startup.ray, "kill", killed.append)
    if eventually_ready:
        assert startup.wait_for_inference_engine_startup(replies, ["actor0", "actor1"], timeout_seconds=90) == replies
        assert killed == []
    else:
        with pytest.raises(TimeoutError, match=r"pending engine actors: \[1\]"):
            startup.wait_for_inference_engine_startup(replies, ["actor0", "actor1"], timeout_seconds=90)
        assert killed == ["actor0", "actor1"]
        assert clock[0] == 90


class RemoteBoundary:
    def __init__(self, callback):
        self.callback = callback
        self.options_kwargs = {}

    def options(self, **kwargs):
        boundary = RemoteBoundary(self.callback)
        boundary.options_kwargs = kwargs
        return boundary

    def remote(self, *args, **kwargs):
        return self.callback(self.options_kwargs, *args, **kwargs)


@pytest.mark.parametrize("bad_report", [None, "duplicate_gpu", "wrong_host", "wrong_ep", "sleeping"])
def test_shared_generation_cross_host_ep_verifies_actual_workers(tmp_path, monkeypatch, bad_report):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    shared_group = PlacementGroup.empty()
    table = {
        "bundles": {rank: {"GPU": 1, "CPU": 1} for rank in range(16)},
        "bundles_to_node_id": {rank: f"node-{rank // 4}" for rank in range(16)},
    }
    nodes = [
        {"Alive": True, "NodeID": f"node-{node}", "NodeManagerHostname": f"host-{node}", "Resources": {"GPU": 4}}
        for node in range(4)
    ]
    monkeypatch.setattr(placement, "placement_group_table", lambda group: table)
    monkeypatch.setattr(ray_utils, "placement_group_table", lambda group: table)
    monkeypatch.setattr(startup.ray, "nodes", lambda: nodes)
    monkeypatch.setattr(startup.ray, "get", lambda refs: refs)
    monkeypatch.setattr(startup.ray, "wait", lambda refs, **kwargs: (refs, []))
    killed = []
    monkeypatch.setattr(startup.ray, "kill", lambda actor: killed.append(actor))
    monkeypatch.setattr(startup.get_all_env_variables, "remote", lambda: {})
    monkeypatch.setattr(
        ray_utils,
        "InfoActor",
        RemoteBoundary(
            lambda options: SimpleNamespace(
                get_gpu_id=RemoteBoundary(lambda _: options["scheduling_strategy"].placement_group_bundle_index % 4)
            )
        ),
    )
    monkeypatch.setattr(
        engine_utils,
        "RendezvousPortReservation",
        RemoteBoundary(
            lambda options, count, excluded: SimpleNamespace(
                details=RemoteBoundary(lambda _: ("127.0.0.1", list(range(30000, 30000 + count))))
            )
        ),
    )
    # The failed implementation probes node IPs before creating serving actors.
    monkeypatch.setattr(
        startup.ray,
        "remote",
        lambda **kwargs: lambda function: RemoteBoundary(
            lambda options: f"10.0.0.{options['scheduling_strategy'].placement_group_bundle_index // 4 + 1}"
        ),
    )
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.20.0dev"
    registry = ModuleType("vllm.model_executor.models")
    registry.ModelRegistry = SimpleNamespace(get_supported_archs=lambda: [])
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", registry)
    actors = []

    def serving_actor(options, **kwargs):
        rank = kwargs["data_parallel_rank"]
        report = {
            "host": f"host-{rank // 4}",
            "gpu_uuid": f"gpu-{rank}",
            "dp_rank": rank,
            "dp_world_size": 16,
            "ep_rank": rank,
            "ep_world_size": 16,
            "torch_rank": rank,
            "torch_world_size": 16,
        }
        if rank == 15:
            if bad_report == "duplicate_gpu":
                report["gpu_uuid"] = "gpu-0"
            elif bad_report == "wrong_host":
                report["host"] = "host-0"
            elif bad_report == "wrong_ep":
                report["ep_rank"] = 0
        actor = SimpleNamespace(
            report_engine_placement=RemoteBoundary(lambda _: [report]),
            get_model_max_len=RemoteBoundary(lambda _: 65536),
        )
        actors.append(actor)
        return actor

    vllm_engine = ModuleType("skyrl_train.inference_engines.vllm.vllm_engine")
    vllm_engine.AsyncVLLMRayActor = RemoteBoundary(serving_actor)
    monkeypatch.setitem(sys.modules, vllm_engine.__name__, vllm_engine)
    kwargs = dict(
        num_inference_engines=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=16,
        expert_parallel_size=16,
        model_dtype="bfloat16",
        pretrain=str(model),
        seed=1,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=False,
        engine_init_timeout_seconds=10,
        shared_pg=shared_group,
        inference_engine_enable_sleep=bad_report == "sleeping",
        engine_init_kwargs={"language_model_only": True},
    )
    if bad_report == "sleeping":
        with pytest.raises(RuntimeError, match="a DP/EP group must be node-local"):
            startup.create_ray_wrapped_inference_engines(**kwargs)
        assert actors == []
        return
    if bad_report is not None:
        error = {"duplicate_gpu": "distinct", "wrong_host": "host disagrees", "wrong_ep": "EP rank"}[bad_report]
        with pytest.raises(ValueError, match=error):
            startup.create_ray_wrapped_inference_engines(**kwargs)
        assert len(actors) == 16
        assert all(actor in killed for actor in actors)
        return
    engines = startup.create_ray_wrapped_inference_engines(**kwargs)
    workers = [engine.worker_placements[0].worker for engine in engines]
    assert {worker.host for worker in workers} == {f"host-{node}" for node in range(4)}
    assert {worker.gpu_uuid for worker in workers} == {f"gpu-{rank}" for rank in range(16)}
    assert [worker.ep_rank for worker in workers] == list(range(16))
    assert all(engine.max_model_len == 65536 for engine in engines)
