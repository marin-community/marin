# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from fray.cluster import ResourceConfig
from thalas.execution.context import executor_context
from thalas.execution.executor import compute_output_path

from experiments.downstream_scaling.evals.algorithms import iid_xregion
from experiments.downstream_scaling.evals.algorithms.iid_xregion import (
    IIDChunkSpec,
    IIDConfig,
    IIDExecutionConfig,
    IIDLocalEngineWorkerConfig,
    IIDModelConfig,
    IIDPoolConfig,
    IIDSamplingConfig,
    _child_config_from_file,
    _chunk_specs,
    _run_iid_chunk,
    _write_child_config,
    iid_tp1_placements,
    make_iid_completion_step,
)
from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement, WorkerPoolConfig

SAMPLING_A = IIDSamplingConfig(
    temperature=0.6,
    top_p=1.0,
    top_k=1000,
    max_tokens=64,
    stop=("stop-a", "stop-b"),
)
SAMPLING_B = IIDSamplingConfig(
    temperature=0.0,
    top_p=0.9,
    top_k=50,
    max_tokens=32,
)

PATH_CONTRACT_PREFIX = "/xregion-tp-path-contract"
IID_TP1_PATH = f"{PATH_CONTRACT_PREFIX}/test/iid-xregion-b17d04"


def _read_jsonl_gz(path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def _worker_pool() -> WorkerPoolConfig:
    return WorkerPoolConfig(
        pool_id="test",
        num_workers=1,
        worker_resources=ResourceConfig.with_cpu(),
        vm_count=1,
        chips_per_vm=4,
    )


def _pool_config(placements: tuple[EnginePlacement, ...] | None = None) -> IIDPoolConfig:
    pool = _worker_pool()
    return IIDPoolConfig(
        pool=pool,
        placements=iid_tp1_placements(pool.chips_per_vm) if placements is None else placements,
    )


def _output_path(tmp_path, config: IIDConfig) -> str:
    with executor_context():
        step = make_iid_completion_step(
            name="test/iid-xregion",
            model_path="/model",
            prompts_path="/prompts",
            config=config,
        )
    return compute_output_path(step.name, step.config, prefix=str(tmp_path))


def test_chunk_specs_tile_each_sampling_config_exactly_once():
    specs = _chunk_specs(
        "chunks",
        num_prompts=3,
        n_samples=2,
        num_sampling_configs=2,
        chunk_size=4,
    )

    assert [spec.chunk_id for spec in specs] == [0, 1, 2, 3]
    assert len({spec.output_path for spec in specs}) == len(specs)
    ranges_by_config: dict[int, list[tuple[int, int]]] = {}
    for spec in specs:
        ranges_by_config.setdefault(spec.sampling_config_index, []).append((spec.chunk_start, spec.chunk_end))
    assert ranges_by_config == {0: [(0, 4), (4, 6)], 1: [(0, 4), (4, 6)]}


def test_chunk_output_uses_selected_sampling_config_and_global_indices(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.jsonl.gz"
    with gzip.open(prompts_path, "wt") as f:
        f.write(json.dumps({"id": "prompt-0", "prompt": "Question"}) + "\n")

    captured: dict[str, object] = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_kwargs"] = kwargs

    class FakeLLM:
        def collective_rpc(self, fn, args):
            captured["rpc"] = (fn, args)

        def generate(self, prompts, _sampling_params):
            captured["prompts"] = prompts
            return [
                SimpleNamespace(outputs=[SimpleNamespace(text=f"answer-{index}", finish_reason="stop")])
                for index, _ in enumerate(prompts)
            ]

    model = IIDModelConfig(max_model_len=8192)

    def fake_load_vllm(model_path, tensor_parallel_size, received_model):
        captured["loader_args"] = (model_path, tensor_parallel_size, received_model)
        return FakeLLM(), FakeSamplingParams

    monkeypatch.setattr(iid_xregion, "_load_vllm", fake_load_vllm)
    chunk = IIDChunkSpec(
        chunk_id=1,
        sampling_config_index=1,
        chunk_start=0,
        chunk_end=2,
        output_path=str(tmp_path / "chunk.jsonl.gz"),
    )

    _run_iid_chunk(
        chunk,
        model_path="/model",
        prompts_path=str(prompts_path),
        model=model,
        sampling=SAMPLING_B,
        n_samples=2,
        seed=7,
        tensor_parallel_size=1,
    )

    assert captured["loader_args"] == ("/model", 1, model)
    assert captured["sampling_kwargs"] == {
        "n": 1,
        "temperature": SAMPLING_B.temperature,
        "top_p": SAMPLING_B.top_p,
        "top_k": SAMPLING_B.top_k,
        "max_tokens": SAMPLING_B.max_tokens,
        "stop": None,
    }
    assert captured["rpc"] == (iid_xregion._reseed_sampler, (8,))
    assert captured["prompts"] == ["Question", "Question"]

    records = _read_jsonl_gz(chunk.output_path)
    assert [record["completion_index"] for record in records] == [2, 3]
    assert [record["completion"]["text"] for record in records] == ["answer-0", "answer-1"]
    persisted_sampling = json.loads(json.dumps(asdict(SAMPLING_B)))
    assert all(record["completion"]["metadata"]["sampling_config"] == persisted_sampling for record in records)


def test_child_config_json_round_trip(tmp_path):
    config = IIDLocalEngineWorkerConfig(
        model_path="/model",
        prompts_path="/prompts",
        n_samples=3,
        seed=11,
        sampling_configs=(SAMPLING_A, SAMPLING_B),
        model=IIDModelConfig(
            max_model_len=16384,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=True,
            apply_rpa_block_size_patch=True,
        ),
        ledger_path="/ledger",
        poll_backoff=0.5,
        owner="worker-0",
        placement=EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2),
    )

    path = _write_child_config(tmp_path, config)

    assert _child_config_from_file(str(path)) == config


def test_completion_path_versions_sampling_semantics_and_max_model_len(tmp_path):
    model = IIDModelConfig(
        max_model_len=8192,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        apply_rpa_block_size_patch=False,
    )
    config = IIDConfig(
        n_samples=2,
        seed=7,
        sampling_configs=(SAMPLING_A,),
        execution=IIDExecutionConfig(worker_pools=(_pool_config(),)),
        model=model,
    )
    base_path = _output_path(tmp_path, config)

    assert _output_path(PATH_CONTRACT_PREFIX, config) == IID_TP1_PATH
    assert _output_path(tmp_path, config) == base_path
    assert _output_path(tmp_path, replace(config, n_samples=3)) != base_path
    assert _output_path(tmp_path, replace(config, seed=8)) != base_path
    assert _output_path(tmp_path, replace(config, sampling_configs=(SAMPLING_B,))) != base_path
    assert _output_path(tmp_path, replace(config, model=replace(model, max_model_len=16384))) != base_path

    assert _output_path(tmp_path, replace(config, model=replace(model, gpu_memory_utilization=0.9))) == base_path
    assert _output_path(tmp_path, replace(config, model=replace(model, enable_prefix_caching=True))) == base_path
    assert _output_path(tmp_path, replace(config, model=replace(model, apply_rpa_block_size_patch=True))) == base_path

    multichip = _pool_config((EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2),))
    assert _output_path(tmp_path, replace(config, execution=IIDExecutionConfig(worker_pools=(multichip,)))) == base_path


def test_pool_config_rejects_overlapping_and_out_of_range_chips():
    with pytest.raises(ValueError, match="overlap"):
        _pool_config(
            (
                EnginePlacement((0, 1), (2, 1, 1), 1),
                EnginePlacement((1, 2), (2, 1, 1), 1),
            )
        )

    with pytest.raises(ValueError, match="outside"):
        _pool_config((EnginePlacement((4,), (1, 1, 1), 1),))


def test_engine_placement_rejects_bounds_with_wrong_volume():
    with pytest.raises(ValueError, match="volume 1, expected 2"):
        EnginePlacement((0, 1), (1, 1, 1), 1)
