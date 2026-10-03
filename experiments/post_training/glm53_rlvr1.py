# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch GLM 5.3 RLVR1 from the Datakit SFT or Antidoom FTPO export."""

import hashlib
from pathlib import Path

import click
import yaml
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

MODEL_REPO = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_EXPORT = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
ANTIDOOM_MODEL_EXPORT = (
    "s3://marin-us-east-02a/marin/users/benfeuer/checkpoints/antidoom-ftpo-dapo-full-window/"
    "2026.10.01.1/exports/global_step_45/policy"
)
ANTIDOOM_EXPLORE_MODEL_EXPORT = (
    "s3://marin-us-east-02a/marin/users/benfeuer/checkpoints/antidoom-rlvr1-async/"
    "2026.10.02.6/exports-best-step12/global_step_12/policy"
)
DATA_ROOT = (
    "s3://marin-us-east-02a/marin/users/benfeuer/datasets/snowball-ultra-rlvr/"
    "20260920-9e7a35d6c979/ordinary-only-v2-agentic57t-compatible/rlvr1"
)
CONFIG_PATH = Path(__file__).parent / "configs" / "glm53_rlvr1_async.yaml"
SWEEP_CONFIG_DIR = CONFIG_PATH.parent / "glm53_rlvr1_sweep"
SWEEP_ARMS = tuple(path.stem for path in sorted(SWEEP_CONFIG_DIR.glob("*.yaml")))
MODEL_VERSION = "2026.09.21"
DATA_VERSION = "2026.09.23"
RL_NAME = user_owned_name("checkpoints/glm53-rlvr1-async")
ANTIDOOM_RL_NAME = user_owned_name("checkpoints/antidoom-rlvr1-async")
ANTIDOOM_EXPLORE_RL_NAME = user_owned_name("checkpoints/antidoom-rlvr1-async-explore")
ANTIDOOM_ABLATIONS = ("token_mean", "reference_kl")


def build_rl_step(
    tokenizer_revision: str,
    smoke: bool,
    sweep_arm: str | None = None,
    model_variant: str = "datakit",
    antidoom_ablation: str | None = None,
) -> ArtifactStep[SkyRLRun]:
    if model_variant not in {"datakit", "antidoom", "antidoom-explore"}:
        raise ValueError(f"Unknown model variant: {model_variant}")
    if antidoom_ablation is not None and model_variant != "antidoom-explore":
        raise ValueError("Antidoom ablations require --model-variant antidoom-explore")
    if antidoom_ablation is not None and sweep_arm is not None:
        raise ValueError("Select one ablation or sweep arm")
    if model_variant == "antidoom-explore":
        model = ArtifactStep.adopt(
            "checkpoints/antidoom-rlvr1-best-step12-hf",
            "2026.10.03.1",
            ANTIDOOM_EXPLORE_MODEL_EXPORT,
            kind=LevanterCheckpoint,
        )
    elif model_variant == "antidoom":
        model = ArtifactStep.adopt(
            "checkpoints/antidoom-ftpo-dapo-step45-hf",
            "2026.10.01.1",
            ANTIDOOM_MODEL_EXPORT,
            kind=LevanterCheckpoint,
        )
    else:
        model = ArtifactStep.adopt(
            "checkpoints/grug-datakit-sft-sep21-hf", MODEL_VERSION, MODEL_EXPORT, kind=LevanterCheckpoint
        )
    data = ArtifactStep.adopt("documents/glm53-rlvr1", DATA_VERSION, DATA_ROOT, kind=Artifact)
    if smoke and sweep_arm is not None:
        raise ValueError("The smoke and sweep configurations cannot be selected together")
    if model_variant in {"antidoom", "antidoom-explore"} and sweep_arm is not None:
        raise ValueError("Sweep arms use the September 21 Datakit model")
    if sweep_arm is not None:
        config_path = SWEEP_CONFIG_DIR / f"{sweep_arm}.yaml"
    elif antidoom_ablation is not None:
        config_path = CONFIG_PATH.with_name(f"glm53_rlvr1_antidoom_{antidoom_ablation}{'_smoke' if smoke else ''}.yaml")
    elif model_variant == "antidoom-explore":
        config_path = CONFIG_PATH.with_name(f"glm53_rlvr1_antidoom_explore{'_smoke' if smoke else ''}.yaml")
    elif model_variant == "antidoom":
        config_path = CONFIG_PATH.with_name(f"glm53_rlvr1_antidoom{'_smoke' if smoke else ''}.yaml")
    elif smoke:
        config_path = CONFIG_PATH.with_name("glm53_rlvr1_smoke.yaml")
    else:
        config_path = CONFIG_PATH
    config_text = config_path.read_text()
    config = yaml.safe_load(config_text)
    tokenizer_cache_key = hashlib.sha256(f"{MODEL_REPO}@{tokenizer_revision}".encode()).hexdigest()
    expected_template = str(Path("/tmp/marinskyrl/model_metadata") / tokenizer_cache_key / "chat_template.jinja")
    if config["generator"]["chat_template"]["name_or_path"] != expected_template:
        raise ValueError("The config's chat template path does not match the pinned tokenizer revision")
    trainer = config["trainer"]
    generator = config["generator"]
    placement = trainer["placement"]
    role_plan = SkyRLRolePlan(
        colocate_all=placement["colocate_all"],
        policy_num_nodes=placement["policy_num_nodes"],
        policy_num_gpus_per_node=placement["policy_num_gpus_per_node"],
        num_inference_engines=generator["num_inference_engines"],
        inference_engine_tensor_parallel_size=generator["inference_engine_tensor_parallel_size"],
        inference_engine_pipeline_parallel_size=generator["inference_engine_pipeline_parallel_size"],
        inference_engine_data_parallel_size=generator["inference_engine_data_parallel_size"],
        inference_engine_expert_parallel_size=generator["inference_engine_expert_parallel_size"],
        train_batch_size=trainer["train_batch_size"],
        policy_mini_batch_size=trainer["policy_mini_batch_size"],
        micro_train_batch_size_per_gpu=trainer["micro_train_batch_size_per_gpu"],
        n_samples_per_prompt=generator["n_samples_per_prompt"],
    )
    inference_gpu_count = (
        role_plan.num_inference_engines
        * role_plan.inference_engine_tensor_parallel_size
        * role_plan.inference_engine_pipeline_parallel_size
        * role_plan.inference_engine_data_parallel_size
    )
    inference_nodes, remainder = divmod(inference_gpu_count, role_plan.policy_num_gpus_per_node)
    if remainder:
        raise ValueError("The inference GPU allocation must fill complete nodes")
    if antidoom_ablation is not None:
        suffix = "-smoke" if smoke else ""
        name = user_owned_name(f"checkpoints/antidoom-rlvr1-{antidoom_ablation.replace('_', '-')}{suffix}")
    elif model_variant == "antidoom-explore":
        name = user_owned_name("checkpoints/antidoom-rlvr1-async-explore-smoke") if smoke else ANTIDOOM_EXPLORE_RL_NAME
    elif model_variant == "antidoom":
        name = user_owned_name("checkpoints/antidoom-rlvr1-async-smoke") if smoke else ANTIDOOM_RL_NAME
    elif sweep_arm is not None:
        name = user_owned_name(f"checkpoints/glm53-rlvr1-sweep-{sweep_arm}")
    elif smoke:
        name = user_owned_name("checkpoints/glm53-rlvr1-async-smoke")
    else:
        name = RL_NAME
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=config_text,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=model,
                relative_path="",
                tokenizer_uri=MODEL_REPO,
                tokenizer_revision=tokenizer_revision,
            ),
            train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
            validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
            topology=SkyRLTopology(
                num_nodes=role_plan.policy_num_nodes + inference_nodes,
                gpus_per_node=role_plan.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=role_plan,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=trainer["max_ckpts_to_keep"]),
            seed=trainer["seed"],
        ),
        IrisSkyRLExecution(
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            cpu=48,
            memory="1611Gi",
            disk="21745Gi",
            priority="interactive",
            max_retries=1,
            target_cluster=None,
            parent_cluster_config=None,
            coordinator_timeout_hours=168,
            wandb_entity="nyu-dice-lab",
        ),
    )


@click.command(help=__doc__)
@click.option("--tokenizer-revision", required=True, help="Published SFT model commit SHA on Hugging Face.")
@click.option("--smoke", is_flag=True, help="Run one optimizer step with the production geometry and a 64-prompt batch.")
@click.option("--sweep-arm", type=click.Choice(SWEEP_ARMS), help="Run one frozen sweep arm.")
@click.option("--model-variant", type=click.Choice(["datakit", "antidoom", "antidoom-explore"]), default="datakit")
@click.option("--antidoom-ablation", type=click.Choice(ANTIDOOM_ABLATIONS))
@rl_build_options
def main(
    tokenizer_revision: str, smoke: bool, sweep_arm: str | None, model_variant: str, antidoom_ablation: str | None
) -> ArtifactStep[SkyRLRun]:
    return build_rl_step(tokenizer_revision, smoke, sweep_arm, model_variant, antidoom_ablation)


if __name__ == "__main__":
    main()
