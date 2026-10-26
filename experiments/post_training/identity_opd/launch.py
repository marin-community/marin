# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch identity OPD from the 16-update context-distillation checkpoint on cw-rno2a.

Two arms share one recipe (``recipe.yaml``) and one frozen mix (``build_mix.py``):

- ``dry``: learning rate 0 for 2 updates, with the mismatch probe scoring the probe prompts through vLLM generation,
  a vLLM re-score and the Megatron trainer before and after each update, every rollout retained, and an HF export
  after update 2 for the weight hashes. It is the trainer's forward-parity check and the step-0 measurement.
- ``train``: the 20-update run.

Both run in the production band with no retries on 8 H100 nodes: 4 policy (PP2 x EP8), 2 rollout engines, 2 teachers.

Usage (prints the plan and output roots; add ``--run`` to submit the coordinator)::

    python -m experiments.post_training.identity_opd.launch --arm dry --version 2026.10.26.1

The storage prefix is pinned to the cluster's region and the mix files to their sha256s; either mismatch stops the
launch before anything is built.
"""

import copy
import hashlib
import os
from pathlib import Path

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import graph_handles
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    SKYRL_TEMPORARY_STORAGE_TTL_DAYS,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
    skyrl_temporary_run_path,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import marin_prefix

EXPERIMENT_NAME = "snowball-identity-opd-cd16"
MIX_NAME = "data/snowball-identity-opd-cd16-mix"
CLUSTER = "cw-rno2a"
# Every output of this run stays in the cluster's region; a 67B checkpoint must not land in another bucket.
# The temporary root holds the native checkpoints and attempt state; unset, it would fall to another region's bucket.
PREFIX = "s3://marin-rno2a/marin"
TEMP_PREFIX = "s3://marin-rno2a"
SEED = 20261026
RECIPE = Path(__file__).with_name("recipe.yaml")

# The 16-update context-distillation export (trained-save audit 78689e19; 502 tensors, 26 FP32 router biases).
STUDENT = ArtifactStep.adopt(
    "models/snowball-67b-a2b-identity-cd16",
    "2026.10.26",
    "s3://marin-rno2a/marin/experiments/snowball-identity/t310/sft-m16-r3/hf/step-15",
    kind=LevanterCheckpoint,
)
# v0.1's tokenizer and chat template (b3f20bc2), the same files the student export carries.
TOKENIZER_URI = "open-athena/Marin-Snowball-67B-A2B-v0.1"
TOKENIZER_REVISION = "adc0078a19a4383bc998fcdb43867b3360baf5f3"

MIX_ROOT = "s3://marin-rno2a/marin/experiments/snowball-identity/t310/opd-cd16-r1/mix"
MIX_FILE = "mixed.parquet"
PROBE_FILE = "probe.parquet"
# The frozen mix (mix-report.json b4f9ba91); the launch refuses any other bytes.
MIX_SHA256 = {
    MIX_FILE: "877a2f60d1f1236585bfa633f6e29460f3f21b0145838e06fae59b9d9e21a26c",
    PROBE_FILE: "ff9eb3021f9956781f06d8d8d843a4c0573b97bcc9e7f510021f6cb1019df327",
}

ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=2,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=64,
    policy_mini_batch_size=64,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=1,
)
NUM_NODES = 8

# Whole-job ceilings (margay's launch plan): 50 and 110 H100-hours on 64 GPUs.
JOB_TIMEOUT_SECONDS = {"dry": 2812, "train": 6187}

DRY_PROBE = {
    "enabled": True,
    "prompts": {"count": 32, "samples_per_prompt": 2},
    "seed": SEED,
    "updates": 2,
}


def pin_prefix() -> None:
    """Make ``PREFIX`` and ``TEMP_PREFIX`` the storage roots, refusing an environment that names others."""
    for variable, expected in (("MARIN_PREFIX", PREFIX), ("MARIN_TEMP_PREFIX", TEMP_PREFIX)):
        current = os.environ.get(variable)
        if current is not None and current.rstrip("/") != expected:
            raise click.UsageError(f"{variable} is {current!r}; this run writes only under {expected}")
        os.environ[variable] = expected
    if marin_prefix() != PREFIX:
        raise click.UsageError(f"storage prefix resolved to {marin_prefix()!r}, not {PREFIX}")


def temporary_root(output_root: str) -> str:
    """The run's native-checkpoint and attempt root, which must stay under ``TEMP_PREFIX``."""
    root = skyrl_temporary_run_path(output_root, ttl_days=SKYRL_TEMPORARY_STORAGE_TTL_DAYS)
    if not root.startswith(f"{TEMP_PREFIX}/"):
        raise click.UsageError(f"temporary root {root} is outside {TEMP_PREFIX}")
    return root


def verify_mix(root: str = MIX_ROOT) -> None:
    """Refuse to launch unless each mix file under ``root`` has its pinned sha256."""
    for name, expected in MIX_SHA256.items():
        digest = hashlib.sha256()
        fs, path = filesystem_for(f"{root}/{name}")
        with fs.open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise click.UsageError(f"{root}/{name} has sha256 {digest.hexdigest()}, expected {expected}")


def recipe(arm: str) -> str:
    config = yaml.safe_load(RECIPE.read_text())
    if arm == "dry":
        config = copy.deepcopy(config)
        trainer = config["trainer"]
        trainer["policy"]["optimizer_config"]["lr"] = 0.0
        # Megatron's scheduler needs warmup < total steps; at LR 0 the schedule has no effect.
        trainer["policy"]["optimizer_config"]["num_warmup_steps"] = 0
        trainer["max_steps"] = 2
        trainer["ckpt_interval"] = 2
        trainer["hf_save_interval"] = 2
        trainer["mismatch_probe"] = DRY_PROBE
        # The terminal HF export reads the native checkpoint at the last step, so both saves move to step 2.
        trainer["callbacks"] = [
            {**callback, "save_steps": 2} if "save_steps" in callback else callback for callback in trainer["callbacks"]
        ]
        # Keep every rollout for the step-0 measures; a failed publish fails the run.
        config["generator"]["trajectory_retention"] = {"sample_fraction": 1.0, "required": True}
    elif arm != "train":
        raise ValueError(f"unknown arm {arm!r}")
    return yaml.safe_dump(config, sort_keys=False)


def build(arm: str) -> ArtifactStep:
    mix = ArtifactStep.adopt(MIX_NAME, resolve_version(MIX_NAME, None), MIX_ROOT)
    name = user_owned_name(f"checkpoints/{EXPERIMENT_NAME}/{arm}")
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=recipe(arm),
            model=ArtifactHfModel(
                step=STUDENT, tokenizer_uri=TOKENIZER_URI, tokenizer_revision=TOKENIZER_REVISION, relative_path=""
            ),
            train_data=(ArtifactDataSource(mix, relative_path=MIX_FILE),),
            validation_data=(ArtifactDataSource(mix, relative_path=PROBE_FILE),) if arm == "dry" else (),
            topology=SkyRLTopology(num_nodes=NUM_NODES, gpus_per_node=8, gpu_variant="H100", role_plan=ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory="1800GB",
            disk="2TB",
            priority="production",
            max_retries=0,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=24,
            wandb_entity="marin-community",
            job_timeout_seconds=JOB_TIMEOUT_SECONDS[arm],
        ),
        export_hf=True,
    )


@click.command(help=__doc__)
@click.option("--arm", type=click.Choice(["dry", "train"]), required=True)
@rl_build_options
def main(arm: str) -> ArtifactStep:
    pin_prefix()
    verify_mix()
    step = build(arm)
    for handle in graph_handles([step]):
        click.echo(f"output root: {handle.name}@{handle.version} -> {handle.path(PREFIX)}")
    click.echo(f"temporary root: {temporary_root(step.path(PREFIX))}")
    return step


if __name__ == "__main__":
    main()
