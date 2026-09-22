# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare FA4 forward backends in fresh processes on one fixed GB200 gang."""

import atexit
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import click
import jax
from iris.cluster.types import Entrypoint
from iris.runtime.jax_init import initialize_jax
from levanter.distributed import _finalize_iris_jax_after_clean_exit, _unregister_iris_exit_handler
from levanter.tracker.json_logger import JsonLoggerConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from rigging.filesystem.storage_path import StoragePath

from experiments.benchmarks.hero_profile import _run_restored_local, build_restored_benchmark, run_restored_benchmark
from experiments.grug.moe_hero_ep.hero_recipe import HeroThroughputResult
from experiments.grug.moe_hero_ep.train import GrugRunConfig, TrainingDataMode

ARMS = (
    ("control", "gpu_fa4_cute_wide"),
    ("native", "gpu_fa4_cute_sm100"),
)
PAIR_ORDERS = ((0, 1), (1, 0), (0, 1))
STEPS = 28
WARMUP_STEPS = 8


def _run_arm(
    config: GrugRunConfig, pair: int, arm: str, *, train: Callable[[GrugRunConfig], None] = _run_restored_local
) -> None:
    trainer = config.trainer.trainer
    # Each fresh distributed world needs its own registry entry; an old endpoint can
    # otherwise be observed while rank zero is still starting the next arm.
    initialize_jax(endpoint_name=f"fa4-{trainer.id}")
    if jax.process_index() == 0:
        source = Path(__file__).resolve().parents[3]
        paths = sorted((source / "lib/levanter/src/levanter/grug/attention").glob("*.py"))
        paths += sorted((source / "experiments/benchmarks/fa4").glob("*.py"))
        paths += [
            source / "experiments/benchmarks/hero_profile.py",
            source / "experiments/grug/moe_hero_ep/train.py",
            source / "lib/levanter/src/levanter/data/loader.py",
            source / "uv.lock",
        ]
        click.echo(
            "FA4_STUDY_ARM "
            + json.dumps(
                {
                    "run_id": trainer.id,
                    "pair": pair,
                    "arm": arm,
                    "implementation": config.model.attention_implementation,
                    "model_shape": {
                        name: getattr(config.model, name)
                        for name in (
                            "hidden_dim",
                            "num_layers",
                            "num_experts",
                            "num_heads",
                            "local_kv_heads",
                            "global_kv_heads",
                            "max_seq_len",
                        )
                    },
                    "batch_size": trainer.train_batch_size,
                    "expert_axis_size": config.trainer.expert_axis_size,
                    "seed": trainer.seed,
                    "checkpoint": trainer.load_checkpoint_path,
                    "initial_step": None if trainer.load_checkpoint else 0,
                    "steps": STEPS,
                    "score_relative_steps": [WARMUP_STEPS, STEPS - 1],
                    "devices": [str(d) for d in jax.devices()],
                    "source_sha256": {
                        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                    },
                    "packages": {
                        p: importlib.metadata.version(p)
                        for p in ("jax", "jaxlib", "jax-cuda13-pjrt", "flash-attn-4", "nvidia-cutlass-dsl")
                    },
                    "xla_flags": os.environ.get("XLA_FLAGS", ""),
                },
                sort_keys=True,
            )
        )
    try:
        train(config)
    finally:
        # The outer study owns job completion; each arm only owns its JAX world.
        _unregister_iris_exit_handler()
        atexit.register(_finalize_iris_jax_after_clean_exit)


def _run_study_local(config: GrugRunConfig, *, train: Callable[[GrugRunConfig], None] = _run_restored_local) -> None:
    trainer = config.trainer.trainer
    assert trainer.id is not None
    group = trainer.id
    completed = []
    for pair, order in enumerate(PAIR_ORDERS, start=1):
        for index in order:
            arm, implementation = ARMS[index]
            run_id = f"{group}-p{pair}-{arm}"
            arm_trainer = dataclasses.replace(
                trainer,
                id=run_id,
                tracker=JsonLoggerConfig(),
            )
            arm_config = dataclasses.replace(
                config,
                model=dataclasses.replace(config.model, attention_implementation=implementation),
                trainer=dataclasses.replace(config.trainer, trainer=arm_trainer),
            )
            entrypoint = Entrypoint.from_callable(_run_arm, arm_config, pair, arm, train=train)
            with tempfile.TemporaryDirectory(prefix="fa4-arm-") as directory:
                for name, content in entrypoint.workdir_files.items():
                    (Path(directory) / name).write_bytes(content)
                subprocess.run(
                    entrypoint.command,
                    env=dict(os.environ, IRIS_WORKDIR=directory, IRIS_PYTHON=sys.executable),
                    check=True,
                )
            completed.append((pair, arm))
            print("FA4_STUDY_ARM_COMPLETE " + json.dumps({"pair": pair, "arm": arm}), flush=True)
    assert len(completed) == sum(len(order) for order in PAIR_ORDERS)
    print("FA4_STUDY_COMPLETE " + json.dumps({"arms": completed}), flush=True)


@click.command()
@click.option("--run-id", required=True)
@click.option("--checkpoint", required=True, help="Exact permanent checkpoint URI shared by all six arms.")
@build_options
def main(run_id: str, checkpoint: str) -> ArtifactStep[HeroThroughputResult]:
    step = build_restored_benchmark(
        run_id=run_id,
        checkpoint=(checkpoint,),
        num_steps=STEPS,
        warmup_steps=WARMUP_STEPS,
        profile_steps=0,
        training_data_mode=TrainingDataMode.MIXTURE,
    )

    def build_config(ctx: StepContext) -> GrugRunConfig:
        config = step.build_config(ctx)
        # This also rules out a moving checkpoint root before allocating the GPU gang.
        with (StoragePath(checkpoint) / "metadata.json").open("r") as stream:
            metadata = json.load(stream)
        click.echo(json.dumps({"study_checkpoint": checkpoint, "checkpoint_step": metadata["step"]}))
        return config

    return dataclasses.replace(
        step,
        build_config=build_config,
        run=functools.partial(run_restored_benchmark, local_entrypoint=_run_study_local),
    )


if __name__ == "__main__":
    main()
