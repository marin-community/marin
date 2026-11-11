# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise dataset lowering, route assignment and quota planning without inference."""

import argparse
import asyncio
import collections
import hashlib
import json
from pathlib import Path

import ray
from omegaconf import OmegaConf
from skyrl_train import objective  # noqa: F401 - register runtime policy losses
from skyrl_train.dataset.tasks import SourceTaskDataset
from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.rollouts.buffer import GroupRewards, PayloadReference, RolloutBuffer, RolloutVerdict
from skyrl_train.rollouts.context import RolloutRequestSpec, TrainingContext
from skyrl_train.telemetry import GeneratedWork
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.utils.utils import validate_cfg


async def probe(compiled: Path, data_root: Path, destination: Path) -> None:
    launch = OmegaConf.load(compiled)
    config = launch.skyrl
    validate_cfg(config)
    tokenizer = tokenizer_from_config(config)
    paths = [data_root / item.relative_path for item in launch.inputs.train_data]
    dataset = SourceTaskDataset(
        datasets=[str(path) for path in paths],
        tokenizer=tokenizer,
        max_prompt_length=config.trainer.max_prompt_length,
        environment_configs=OmegaConf.to_container(config.environment.task_sessions, resolve=True),
        num_workers=8,
        chat_template_kwargs=dict(config.generator.chat_template_kwargs),
    )
    validation_paths = [data_root / item.relative_path for item in launch.inputs.validation_data]
    validation_dataset = SourceTaskDataset(
        datasets=[str(path) for path in validation_paths],
        tokenizer=tokenizer,
        max_prompt_length=config.trainer.max_prompt_length,
        environment_configs=OmegaConf.to_container(config.environment.task_sessions, resolve=True),
        num_workers=8,
        chat_template_kwargs=dict(config.generator.chat_template_kwargs),
    )
    counts = collections.Counter(dataset.dataframe["data_source"])
    routes = dict(config.trainer.rollout_buffer.source_routes)
    if set(counts) != set(routes):
        raise ValueError(f"Compiled routes and actual dataset sources differ: {set(counts) ^ set(routes)}")
    # This creates the real CPU buffer actor and executes context.py's setup.
    ray.init(num_cpus=1, include_dashboard=False, logging_level="ERROR")
    context = TrainingContext.from_config(config, dataset, workers=None)
    try:
        seen = set()
        for index, source in enumerate(dataset.dataframe["data_source"]):
            if source not in seen:
                prompt = dataset.collate_fn([dataset[index]])[0]
                RolloutRequestSpec.from_config(config).request(prompt, 1)
                seen.add(source)
        # Exercise the actual quota planner with scripted rewards, not model evidence.
        buffer = RolloutBuffer(context.config)
        await buffer.publish(1)
        live = set()
        consumed = collections.Counter()
        group_size = config.generator.n_samples_per_prompt
        rewards = tuple(float(i % 2) for i in range(group_size))
        for _ in range(config.trainer.train_batch_size):
            lease = await buffer.acquire_lease()
            prompt = context.loader.next_prompt(live, lease.route)
            if prompt is None:
                raise ValueError(f"Route {lease.route} cannot supply a prompt")
            live.add(prompt["uid"])
            consumed[prompt["group_route"]] += 1
            verdict = RolloutVerdict(
                prompt["uid"],
                (),
                GroupSelectionResult.KEEP,
                GroupRewards(rewards, rewards, True),
                GeneratedWork(group_size, group_size, group_size),
            )
            await buffer.commit(lease.lease_id, prompt, verdict, PayloadReference(lease.lease_id))
        admitted = await buffer.admit(1)
        expected = dict(config.trainer.rollout_buffer.consumed_route_quotas)
        if dict(consumed) != expected:
            raise ValueError(f"Incorrect consumed routes: {consumed}")
        destination.write_text(
            json.dumps(
                {
                    "compiled_sha256": hashlib.sha256(compiled.read_bytes()).hexdigest(),
                    "data_sha256": {
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [*paths, *validation_paths]
                    },
                    "validation_rows_after_runtime_filter": len(validation_dataset),
                    "rows_after_runtime_filter": len(dataset),
                    "source_counts": counts,
                    "source_routes": routes,
                    "consumed_routes": consumed,
                    "quota_metrics": admitted.selection.metrics,
                    "group_size": group_size,
                    "model_inference": False,
                    "gpu_qualification": "not exercised; scripted rewards test CPU setup only",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        await context.close()
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(probe(args.compiled, args.data_root, args.output))


if __name__ == "__main__":
    main()
