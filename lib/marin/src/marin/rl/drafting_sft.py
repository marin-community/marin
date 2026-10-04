# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable authored recipes for online draft-model SFT during SkyRL rollouts."""

from rigging.filesystem.storage_path import StoragePath

from marin.skyrl_recipe import (
    Algorithm,
    ContextBudget,
    Data,
    Environment,
    Generator,
    Model,
    RecipePatch,
    RLEntrypoint,
    SamplingParams,
    SkyRLRecipe,
    SpeculativeDecoding,
    Trainer,
)


def draft_sft_recipe(
    initial_draft: StoragePath,
    initial_draft_identity: str,
    *,
    policy: RecipePatch,
    training: SpeculativeDecoding,
    environment: str,
    project_name: str,
    request_window_tokens: int,
    max_new_tokens_per_turn: int,
    max_steps: int,
    checkpoint_interval: int,
) -> SkyRLRecipe:
    """Combine target-policy settings with a bounded online draft trainer."""
    return SkyRLRecipe.combine(
        base=RecipePatch(
            entrypoint=RLEntrypoint.STANDARD,
            context_budget=ContextBudget(
                request_window_tokens=request_window_tokens, max_new_tokens_per_turn=max_new_tokens_per_turn, max_turns=1
            ),
            environment=Environment(env_class=environment),
            trainer=Trainer(
                strategy="megatron",
                flash_attn=False,
                use_sample_packing=False,
                offload_optimizer_during_rollouts=True,
                gradient_checkpointing=True,
                algorithm=Algorithm(advantage_estimator="grpo", use_kl_loss=False),
                epochs=1,
                max_steps=max_steps,
                update_epochs_per_batch=1,
                eval_before_train=False,
                eval_interval=-1,
                ckpt_interval=checkpoint_interval,
                resume_mode="latest",
                logger="wandb",
                project_name=project_name,
            ),
            generator=Generator(
                backend="vllm",
                model_dtype="bfloat16",
                vllm_attention_backend="FLASH_ATTN",
                gpu_memory_utilization=0.75,
                enforce_eager=False,
                run_engines_locally=True,
                weight_sync_backend="nccl",
                engine_init_kwargs={"async_scheduling": False},
                sampling_params=SamplingParams(temperature=1.0, top_p=1.0),
            ),
            data=Data(kind="parquet", train_data=(), val_data=()),
            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        ),
        policy=policy,
        draft=RecipePatch(
            generator=Generator(
                speculative_decoding=training.merge(
                    SpeculativeDecoding(
                        model=Model(source_uri=str(initial_draft), source_identity=initial_draft_identity)
                    )
                )
            )
        ),
    )
