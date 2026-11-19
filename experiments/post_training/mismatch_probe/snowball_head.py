# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score frozen Snowball trajectories under a paired LM-head precision ablation."""

from dataclasses import asdict

import click
import yaml
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    MegatronGeometry,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from skyrlrecipe import ContextBudget, RecipePatch, RLEntrypoint, SkyRLRecipe, sections

from experiments.post_training.mismatch_probe.launch import ProbeSettings, probe_patch

HEAD_PRECISION = {
    "bf16": (False, False),
    "bf16-repeat": (False, False),
    "vllm-fp32": (True, False),
    "megatron-fp32": (False, True),
    "both-fp32": (True, True),
}
GPUS_PER_NODE = 8
SAMPLES_PER_PROMPT = 2
SEED = 17


def snowball_recipe(arm: str, context_parallel_size: int, prompt_count: int, reuse_probe: str | None) -> SkyRLRecipe:
    if (arm == "bf16") != (reuse_probe is None):
        raise ValueError("The first BF16 arm collects responses; every other arm must reuse its probe archive")
    serving_fp32, learner_fp32 = HEAD_PRECISION[arm]
    geometry = asdict(MegatronGeometry(1, 2, context_parallel_size, 8, 1))
    settings = ProbeSettings(
        seed=SEED,
        prompt_count=prompt_count,
        samples_per_prompt=SAMPLES_PER_PROMPT,
        updates=0,
        keep_fraction=1.0,
        cache_mode="off",
        reuse_probe=reuse_probe,
        resume_path=None,
        extra_trainer_modes=("router_replay",),
    )
    recipe = SkyRLRecipe(
        entrypoint=RLEntrypoint.STANDARD,
        context_budget=ContextBudget(request_window_tokens=4096, max_new_tokens_per_turn=2048, max_turns=1),
        data=sections.Data(kind="parquet", shuffle=False),
        environment=sections.Environment(task_sessions={"local": {}}),
        trainer=sections.Trainer(
            strategy="megatron",
            use_sample_packing=True,
            flash_attn=False,
            epochs=1,
            max_steps=1,
            micro_forward_batch_size_per_gpu=1,
            eval_interval=-1,
            eval_before_train=False,
            resume_mode="none",
            logger="console",
            algorithm=sections.Algorithm(advantage_estimator="grpo", use_kl_loss=False),
            policy=sections.Policy(
                megatron_config=sections.PolicyMegatronConfig(
                    **geometry,
                    transformer_config_kwargs={"cp_comm_type": "a2a", "moe_router_fusion": False},
                    moe_router_replay=True,
                    fp32_lm_head=learner_fp32,
                ),
            ),
            ref=sections.Ref(megatron_config=sections.RefMegatronConfig(**geometry)),
        ),
        generator=sections.Generator(
            backend="vllm",
            run_engines_locally=True,
            model_dtype="bfloat16",
            vllm_attention_backend="FLASH_ATTN",
            gpu_memory_utilization=0.75,
            max_num_seqs=64,
            max_num_batched_tokens=8192,
            weight_sync_backend="nccl",
            weight_sync_transport="expert_block",
            chat_template_kwargs={"enable_thinking": True},
            engine_init_kwargs={"kernel_config": {"moe_backend": "triton"}},
        ),
        extra_env={"VLLM_LOGGING_LEVEL": "DEBUG", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    head_patch = RecipePatch(
        generator=sections.Generator(engine_init_kwargs={"hf_overrides": {"head_dtype": "float32"}})
    )
    recipe = recipe.merge(probe_patch(settings))
    return recipe.merge(head_patch) if serving_fp32 else recipe


@click.command(help=__doc__)
@click.option("--arm", type=click.Choice(tuple(HEAD_PRECISION)), required=True)
@click.option("--cluster", type=click.Choice(("cw-rno2a", "cw-us-east-02a")), required=True)
@click.option("--context-parallel-size", type=click.Choice(("1", "5")), required=True)
@click.option("--model-uri", required=True, help="Regional Snowball 10/05 HF export.")
@click.option("--tokenizer", required=True)
@click.option("--tokenizer-revision", required=True)
@click.option("--data-uri", required=True, help="Regional artifact containing the pinned validation task parquet.")
@click.option("--data-file", default="validation-tasks.parquet", show_default=True)
@click.option("--input-version", required=True)
@click.option("--prompt-count", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--reuse-probe", help="Completed BF16 collector archive, required for every other arm.")
@rl_build_options
def main(
    arm: str,
    cluster: str,
    context_parallel_size: str,
    model_uri: str,
    tokenizer: str,
    tokenizer_revision: str,
    data_uri: str,
    data_file: str,
    input_version: str,
    prompt_count: int,
    reuse_probe: str | None,
) -> ArtifactStep[SkyRLRun]:
    cp = int(context_parallel_size)
    plan = SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=2 * cp,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=GPUS_PER_NODE,
        inference_engine_expert_parallel_size=GPUS_PER_NODE,
        train_batch_size=32,
        policy_mini_batch_size=32,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=SAMPLES_PER_PROMPT,
    )
    model = ArtifactStep.adopt(
        user_owned_name("models/snowball-1005-probe"), input_version, model_uri, kind=LevanterCheckpoint
    )
    data: ArtifactStep[Artifact] = ArtifactStep.adopt(
        user_owned_name("documents/snowball-head-probe"), input_version, data_uri
    )
    dataset = ArtifactDataSource(data, relative_path=data_file)
    name = user_owned_name(f"checkpoints/snowball-head-probe/{arm}")
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(snowball_recipe(arm, cp, prompt_count, reuse_probe).to_skyrl(), sort_keys=False),
            model=ArtifactHfModel(model, tokenizer, tokenizer_revision, relative_path=""),
            train_data=(dataset,),
            validation_data=(dataset,),
            topology=SkyRLTopology(
                num_nodes=plan.num_nodes, gpus_per_node=GPUS_PER_NODE, gpu_variant="H100", role_plan=plan
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1, temporary_storage_ttl_days=30),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=cluster,
            cluster_config=f"lib/iris/config/{cluster}.yaml",
            cpu=32,
            memory="1800GB",
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=2,
            job_timeout_seconds=(25 if cp == 5 else 75) * 60,
        ),
        export_hf=False,
    )


if __name__ == "__main__":
    main()
