# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch identity OPD from the 16-update context-distillation checkpoint on cw-rno2a.

Two arms share one recipe (``recipe.yaml``) and one frozen mix (``build_mix.py``):

- ``dry``: learning rate 0 for 2 updates, with the mismatch probe scoring the probe prompts through vLLM generation,
  a vLLM re-score and the Megatron trainer before and after each update, and an HF export after update 2 for the
  weight hashes. It is the trainer's forward-parity check and the step-0 measurement.
- ``train``: the 20-update run.

Both run in the production band with no retries on 8 H100 nodes: 4 policy (PP2 x EP8), 2 rollout engines, 2 teachers.
"""

import copy
from pathlib import Path

import click
import yaml
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

EXPERIMENT_NAME = "snowball-identity-opd-cd16"
CLUSTER = "cw-rno2a"
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


def recipe(arm: str) -> str:
    config = yaml.safe_load(RECIPE.read_text())
    if arm == "dry":
        config = copy.deepcopy(config)
        trainer = config["trainer"]
        trainer["policy"]["optimizer_config"]["lr"] = 0.0
        trainer["max_steps"] = 2
        trainer["ckpt_interval"] = 2
        trainer["hf_save_interval"] = 2
        trainer["mismatch_probe"] = DRY_PROBE
        trainer["callbacks"] = [
            {"type": "hf_model_save", "save_steps": 2, "save_on_train_end": True},
            {"type": "logging"},
            {"type": "progress"},
        ]
    elif arm != "train":
        raise ValueError(f"unknown arm {arm!r}")
    return yaml.safe_dump(config, sort_keys=False)


def build(arm: str, version: str):
    mix = ArtifactStep.adopt("data/snowball-identity-opd-cd16-mix", version, MIX_ROOT)
    name = f"checkpoints/{EXPERIMENT_NAME}/{arm}"
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(name),
            version=version,
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


@click.command()
@click.option("--arm", type=click.Choice(["dry", "train"]), required=True)
@click.option("--version", required=True, help="calendar version of the frozen mix and this run, e.g. 2026.10.26.1")
def main(arm: str, version: str) -> None:
    click.echo(build(arm, version))


if __name__ == "__main__":
    main()
