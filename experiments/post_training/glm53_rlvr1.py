# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch GLM 5.3 RLVR1 from the September 21 Datakit SFT export."""

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
DATA_ROOT = (
    "s3://marin-us-east-02a/marin/users/benfeuer/datasets/snowball-ultra-rlvr/"
    "20260920-9e7a35d6c979/ordinary-only-v2-agentic57t-compatible/rlvr1"
)
CONFIG_PATH = Path(__file__).parent / "configs" / "glm53_rlvr1_async.yaml"
MODEL_VERSION = "2026.09.21"
DATA_VERSION = "2026.09.23"
RL_NAME = user_owned_name("checkpoints/glm53-rlvr1-async")


def build_rl_step(tokenizer_revision: str, smoke: bool) -> ArtifactStep[SkyRLRun]:
    model = ArtifactStep.adopt(
        "checkpoints/grug-datakit-sft-sep21-hf", MODEL_VERSION, MODEL_EXPORT, kind=LevanterCheckpoint
    )
    data = ArtifactStep.adopt("documents/glm53-rlvr1", DATA_VERSION, DATA_ROOT, kind=Artifact)
    config_path = CONFIG_PATH.with_name("glm53_rlvr1_smoke.yaml") if smoke else CONFIG_PATH
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
    name = user_owned_name("checkpoints/glm53-rlvr1-async-smoke") if smoke else RL_NAME
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
@rl_build_options
def main(tokenizer_revision: str, smoke: bool) -> ArtifactStep[SkyRLRun]:
    return build_rl_step(tokenizer_revision, smoke)


if __name__ == "__main__":
    main()
