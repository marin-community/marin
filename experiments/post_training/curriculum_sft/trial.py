# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared SFT-and-evaluate skeleton for curriculum examples on the September Snowball HF checkpoint.

Every example trains the same pinned checkpoint on its own chat rows with the same optimizer, mesh,
and step count, exports the final step to HF, and evaluates the base and trained models on the
example's matched benchmarks. Examples differ only in their datasets and evaluations.
"""

import dataclasses

from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.checkpoints import hf_to_levanter
from marin.experiment.namespacing import user_owned_name
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.models import SNOWBALL_VLLM_ARGS
from experiments.evaluation.pipeline import EvaluationResult, eval_step
from experiments.sft.launcher import (
    LLAMA3_CHAT_EOS_TOKEN_IDS,
    ArtifactDatasetSpec,
    ConvertedCheckpointModel,
    SFTSpec,
    sft_step,
)

HF_MODEL = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.20"
HF_REVISION = "9f2ee50f3d4a12c79b0808bb2414ddba2cdf0098"
CLUSTER = "cw-rno2a"
# Generated data outlives individual trials; checkpoints and evaluations expire sooner.
SOURCE_PREFIX = marin_temp_bucket(
    ttl_days=30,
    prefix="curriculum-math-20260924",
    source_prefix="s3://marin-us-east-02a/marin",
    use_env_override=False,
)
S3_TRIAL_PREFIX = marin_temp_bucket(
    ttl_days=7,
    prefix="curriculum-math-20260924",
    source_prefix="s3://marin-us-east-02a/marin",
)
CONVERSION_VERSION = "2026.09.25.2"
CONTEXT_LENGTH = 4096
BATCH_SIZE = 64
STEPS = 4
TRAIN_NODES = 4


def _gpu_resources(nodes: int) -> ResourceConfig:
    return ResourceConfig.with_gpu(
        "H100",
        count=8,
        cpu=32,
        ram="512g",
        disk="256g",
        replicas=nodes,
        preemptible=False,
    )


def _optimizer(learning_rate: float, warmup: int) -> AdamConfig:
    return AdamConfig(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.0,
        min_lr_ratio=0.1,
        warmup=warmup,
        lr_schedule="cosine",
    )


def snowball_model(name: str, location: str, revision: str | None) -> ModelConfig:
    """Serve a Snowball HF checkpoint in thinking mode with the pinned tokenizer."""
    return ModelConfig(
        name=name,
        location=location,
        revision=revision,
        tokenizer=HF_MODEL,
        tokenizer_revision=HF_REVISION,
        apply_chat_template=True,
        resource_hint=ResourceHint(gpu={"H100": 8}, memory="512g"),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=8,
            max_model_len=32768,
            max_num_batched_tokens=7168,
            max_num_seqs=32,
            auto_overrides=False,
            vllm_extra_args=SNOWBALL_VLLM_ARGS,
        ),
        generation=GenerationConfig(
            max_gen_toks=8192,
            extra_gen_kwargs={"skip_special_tokens": "false", "repetition_penalty": "1.1"},
        ),
    )


def build_trial(
    *,
    name: str,
    eval_name: str,
    datasets: list[ArtifactDatasetSpec],
    evals: str,
    version: str,
    learning_rate: float,
    warmup: int,
) -> dict[str, ArtifactStep]:
    """Bind baseline and trained evaluations to one Snowball SFT run.

    Args:
        name: Checkpoint name under ``checkpoints/curriculum-sft/``.
        eval_name: Slash-free prefix for the evaluated models' launch identities.
        datasets: Chat datasets to train on.
        evals: Comma-separated evaluation names for the baseline and trained models.
        version: Version shared by the SFT checkpoint and both evaluations.
        learning_rate: Peak Adam learning rate; 0 exercises the train/export/eval path without updates.
        warmup: Linear warmup length in optimizer steps.
    """
    conversion = hf_to_levanter(
        HF_MODEL,
        model_type="snowball",
        hf_revision=HF_REVISION,
        tokenizer=f"hf://{HF_MODEL}@{HF_REVISION}",
        version=CONVERSION_VERSION,
        resources=ResourceConfig.with_cpu(cpu=64, ram="512g", disk="256g"),
    )
    spec = SFTSpec(
        name=user_owned_name(f"checkpoints/curriculum-sft/{name}"),
        version=version,
        model=ConvertedCheckpointModel(conversion=conversion, eos_token_ids=LLAMA3_CHAT_EOS_TOKEN_IDS),
        chat_template=MARIN_CHAT_TEMPLATE,
        datasets=datasets,
        optimizer=_optimizer(learning_rate, warmup),
        # Snowball shards parameters over expert x (data, context). Placing context across nodes shards
        # fp32 weights and Adam moments over all 32 GPUs instead of replicating them on each node.
        mesh=MeshConfig(
            axes={"data": 1, "replica": 1, "model": 1, "expert": -1},
            dcn_axes={"context": TRAIN_NODES},
            compute_mapping={
                "batch": ["replica_dcn", "data", "expert"],
                "position": "context",
                "vocab": "model",
            },
        ),
        seq_len=CONTEXT_LENGTH,
        pack=False,
        batch_size=BATCH_SIZE,
        num_train_steps=STEPS,
        hf_save_dtype="bfloat16",
        wandb_project="marin_moe_sft",
    )
    trained = sft_step(spec, _gpu_resources(TRAIN_NODES))
    baseline_model = snowball_model(f"{eval_name}-base", HF_MODEL, HF_REVISION)
    trained_model = snowball_model(f"{eval_name}-trained", "<trained-hf>", None)

    def resolve_trained_model(ctx: StepContext) -> ModelConfig:
        # The trainer exports HF weights once, at the final (zero-indexed) step.
        final_export = prefix_join(ctx.artifact_path(trained), f"hf/step-{STEPS - 1}")
        return dataclasses.replace(trained_model, location=final_export)

    baseline: ArtifactStep[EvaluationResult] = eval_step(
        baseline_model,
        evals,
        version=version,
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    after: ArtifactStep[EvaluationResult] = eval_step(
        trained_model,
        evals,
        version=version,
        deps=(trained,),
        resolve_model=resolve_trained_model,
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    return {"baseline": baseline, "train": trained, "after": after}
