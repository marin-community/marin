# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose a production launch document in SkyRL's frozen CPU environment."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

from cloud.iris.launch_config import load_launch_config
from cloud.iris.rl_config_translation import RL_ENTRYPOINTS, RLEntrypoint
from cloud.iris.role_plan import ModelRoleKind, RoleExecution, derive_role_plan
from cloud.iris.runtime_environment import IRIS_TASK_IMAGE
from cloud.iris.task_runtime import prepare_policy_tokenizer
from omegaconf import OmegaConf
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from rigging.secrets import get_secret
from skyrl_train import objective  # noqa: F401 - populate the runtime policy-loss registry
from skyrl_train.config.driver_validation import validate_cfg, validate_generator_cfg
from skyrl_train.models.grug_context_parallel import grug_cp_communication, validate_cp_attention
from skyrl_train.models.grug_moe import GrugMoeConfig, grug_long_layer_flags
from skyrl_train.rollouts.agent_runtime import InstalledAgentSettings
from skyrl_train.rollouts.task_worker import TaskRolloutWorkerSpec
from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.trajectory_runners.harbor.configuration import HarborConfigBuilder


async def validate_rollout_worker(launch) -> dict:
    """Build and close the production worker without starting inference engines."""
    if sys.platform == "linux":
        descriptor = os.pidfd_open(os.getpid())
        os.close(descriptor)
    config = launch.skyrl
    prepared = prepare_policy_tokenizer(
        argparse.Namespace(
            policy_tokenizer=str(launch.inputs.model.tokenizer_uri),
            policy_tokenizer_revision=str(launch.inputs.model.tokenizer_revision),
        )
    )
    assert prepared is not None
    tokenizer_config = OmegaConf.merge(
        config,
        {"trainer": {"policy": {"model": {"tokenizer_path": prepared.local_path, "tokenizer_revision": None}}}},
    )
    sessions = config.environment.task_sessions
    for settings in (
        sessions.get("remote_judge") or {},
        sessions.get("daytona") or {},
        config.terminal_bench_config.get("harbor", {}),
    ):
        for key, name in settings.items():
            if key.endswith("api_key_env") and name is not None and (value := get_secret(name)) is not None:
                os.environ[name] = value
    worker = TaskRolloutWorkerSpec.from_config(config, [], harbor_config=config.terminal_bench_config).build(
        tokenizer_from_config(tokenizer_config), WorkerShard(0, 1)
    )
    try:
        return {
            "worker_type": type(worker).__name__,
            "installed_agent": worker.installed_agents is not None,
            "judge_base_url": None if worker.judge_connection is None else worker.judge_connection.base_url,
            "inference_engines": 0,
        }
    finally:
        await worker.shutdown()


def main(source: Path, destination: Path, model_config: Path | None = None, *, build_worker: bool = False) -> None:
    project = Path(__file__).resolve().parents[3] / "vendor/skyrl"
    config = load_launch_config(source)
    HarborConfigBuilder(config.skyrl.terminal_bench_config)
    installed = config.skyrl.terminal_bench_config.get("installed_agent")
    if installed is not None:
        InstalledAgentSettings.model_validate(OmegaConf.to_container(installed, resolve=True))
    if config.runtime.entrypoint in {
        RL_ENTRYPOINTS[RLEntrypoint.GENERATE],
        RL_ENTRYPOINTS[RLEntrypoint.TERMINAL_BENCH_GENERATE],
    }:
        validate_generator_cfg(config.skyrl)
    else:
        validate_cfg(config.skyrl)
    if config.skyrl.trainer.strategy == "megatron" and config.runtime.entrypoint not in {
        RL_ENTRYPOINTS[RLEntrypoint.GENERATE],
        RL_ENTRYPOINTS[RLEntrypoint.TERMINAL_BENCH_GENERATE],
    }:
        if model_config is None:
            fs, key = filesystem_for(prefix_join(config.inputs.model.uri, "config.json"))
            model_bytes = fs.cat_file(key)
        else:
            model_bytes = model_config.read_bytes()
        model_dict = json.loads(model_bytes)
        if model_dict.get("model_type") == "grug_moe":
            model = GrugMoeConfig(**model_dict)
            long_layers = grug_long_layer_flags(model.num_hidden_layers, model.global_every)
            for worker in (config.skyrl.trainer.policy, config.skyrl.trainer.ref):
                topology = worker.megatron_config
                communications = topology.transformer_config_kwargs.get(
                    "cp_comm_type", grug_cp_communication(long_layers, hero=model.uses_hero_architecture)
                )
                if isinstance(communications, str):
                    communications = [communications] * len(long_layers)
                for is_long, communication in zip(long_layers, communications, strict=True):
                    validate_cp_attention(
                        query_heads=model.num_attention_heads,
                        kv_heads=(model.global_kv_heads if is_long else model.local_kv_heads)
                        or model.num_key_value_heads,
                        tensor_parallel=topology.tensor_model_parallel_size,
                        context_parallel=topology.context_parallel_size,
                        communication=communication,
                        sliding_window=not is_long,
                    )
    submitted = OmegaConf.load(source)
    role_geometry = []
    for claim in derive_role_plan(OmegaConf.to_container(config.skyrl, resolve=True)).claims:
        if claim.execution is RoleExecution.REMOTE:
            continue
        ranks = claim.tensor_parallel_size * claim.pipeline_parallel_size * claim.data_parallel_size
        required = ranks * (claim.replicas if claim.kind is ModelRoleKind.ROLLOUT else 1)
        allocated = claim.num_nodes * claim.gpus_per_node
        if required > allocated:
            raise ValueError(f"Role {claim.role_id} requires {required} GPUs but selected {allocated}")
        role_geometry.append({"role": claim.role_id, "required_gpus": required, "allocated_gpus": allocated})
    expected_routes = OmegaConf.select(submitted, "skyrl.trainer.rollout_buffer.source_routes", default={})
    actual_routes = OmegaConf.select(config, "skyrl.trainer.rollout_buffer.source_routes", default={})
    if OmegaConf.to_container(OmegaConf.create(expected_routes)) != OmegaConf.to_container(
        OmegaConf.create(actual_routes)
    ):
        raise ValueError("Strict composition changed literal consumed-route source identifiers")
    harnesses = OmegaConf.select(config, "skyrl.environment.task_sessions.task_spec.harness_by_dataset", default={})
    if "harbor_trial" in harnesses.values():
        engine = config.skyrl.generator.engine_init_kwargs
        if not engine.get("enable_auto_tool_choice") or not engine.get("tool_call_parser"):
            raise ValueError("Installed-agent tool requests require automatic tool choice and a tool parser")
    worker = asyncio.run(validate_rollout_worker(config)) if build_worker else None
    compiled = OmegaConf.to_yaml(config, resolve=True)
    destination.write_text(compiled)
    print(
        json.dumps(
            {
                "task_image": IRIS_TASK_IMAGE,
                "rollout_worker": worker,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "compiled_sha256": hashlib.sha256(compiled.encode()).hexdigest(),
                "validator": "load_launch_config + skyrl_train.utils.utils.validate_cfg",
                "runtime_entrypoint": config.runtime.entrypoint,
                "role_geometry": role_geometry,
                "python": sys.version,
                "python_executable": sys.executable,
                "python_base_prefix": sys.base_prefix,
                "platform": sys.platform,
                "skyrl_lock_sha256": hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest(),
                "initializer_config_sha256": (
                    hashlib.sha256(model_config.read_bytes()).hexdigest() if model_config is not None else None
                ),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("model_config", type=Path, nargs="?")
    parser.add_argument("--build-worker", action="store_true")
    args = parser.parse_args()
    main(args.source, args.destination, args.model_config, build_worker=args.build_worker)
