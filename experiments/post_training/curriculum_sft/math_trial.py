# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge-free math curriculum SFT trial on the September Snowball HF checkpoint."""

import dataclasses
import hashlib
import json

import click
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.checkpoints import hf_to_levanter
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.models import SNOWBALL_VLLM_ARGS
from experiments.evaluation.pipeline import EvaluationResult, eval_step
from experiments.post_training.curriculum_sft.pipeline import (
    CurriculumGenerationSpec,
    curriculum_generation_steps,
    prepare_curriculum_chat_step,
)
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
SOURCE_VERSION = "2026.09.24"
PREPARATION_VERSION = "2026.09.25.12"
CONVERSION_VERSION = "2026.09.25.2"
CURRICULUM_IDS = (
    "d01.algebra.exact-symbolic-evaluation",
    "d01.algebra.scalar-equations",
    "d01.algebra.number-theoretic-reasoning",
)
EVALS = "olympiadbench-deterministic,math500"
CONTEXT_LENGTH = 4096
BATCH_SIZE = 64
STEPS = 4
TRAIN_NODES = 4
ACCEPTED_PER_CAPABILITY = 256
REQUESTED_PER_CAPABILITY = 320
SEED = 17
MAX_COMPLETION_TOKENS = 4096
TASK_SPECIFICATION = (
    "Create an original, self-contained, exact mathematics problem in this capability. "
    "Make the answer unique and require several reasoning steps. Avoid published contest questions, "
    "external facts, and proof-only prompts. End the assistant solution with one final answer in "
    "\\boxed{...} notation. Check the answer by substitution or an independent calculation."
)
GENERATION = CurriculumGenerationSpec(
    requested_examples=REQUESTED_PER_CAPABILITY,
    accepted_examples=ACCEPTED_PER_CAPABILITY,
    seed=SEED,
    max_completion_tokens=MAX_COMPLETION_TOKENS,
    task_specification=TASK_SPECIFICATION,
)


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


def _optimizer() -> AdamConfig:
    return AdamConfig(
        learning_rate=5e-5,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.0,
        min_lr_ratio=0.1,
        warmup=0,
        lr_schedule="cosine",
    )


def _eval_model(name: str, location: str, revision: str | None) -> ModelConfig:
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


def build_generation(version: str) -> dict[str, ArtifactStep[Artifact]]:
    """Build generation steps for submission on the relay's east-region cluster."""
    return curriculum_generation_steps(CURRICULUM_IDS, version=version, generation=GENERATION)


def _staged_generation(version: str) -> dict[str, ArtifactStep[Artifact]]:
    sources: dict[str, ArtifactStep[Artifact]] = {}
    for capability_id in CURRICULUM_IDS:
        name = user_owned_name(f"documents/curriculum-sft/{capability_id}/staged-chat")
        generated_name = user_owned_name(f"documents/curriculum-sft/{capability_id}/generated-chat")
        generated_path = prefix_join(prefix_join(SOURCE_PREFIX, generated_name), version)
        sources[capability_id] = ArtifactStep.adopt(
            name=name,
            version=version,
            source=generated_path,
            kind=Artifact,
        )
    return sources


def build_trial(version: str) -> dict[str, ArtifactStep]:
    """Bind baseline and trained evaluations to Levanter's Snowball SFT run."""
    curriculum_key = hashlib.sha256(json.dumps(sorted(CURRICULUM_IDS)).encode()).hexdigest()[:12]
    generated = _staged_generation(SOURCE_VERSION)
    conversion = hf_to_levanter(
        HF_MODEL,
        model_type="snowball",
        hf_revision=HF_REVISION,
        tokenizer=f"hf://{HF_MODEL}@{HF_REVISION}",
        version=CONVERSION_VERSION,
        resources=ResourceConfig.with_cpu(cpu=64, ram="512g", disk="256g"),
    )
    datasets = [
        ArtifactDatasetSpec(
            slug=capability_id,
            artifact=prepare_curriculum_chat_step(
                generated[capability_id], capability_id=capability_id, version=PREPARATION_VERSION
            ),
            train_glob="*.parquet",
            weight=1.0,
        )
        for capability_id in CURRICULUM_IDS
    ]
    spec = SFTSpec(
        name=user_owned_name(f"checkpoints/curriculum-sft/{curriculum_key}/snowball"),
        version=version,
        model=ConvertedCheckpointModel(conversion=conversion, eos_token_ids=LLAMA3_CHAT_EOS_TOKEN_IDS),
        chat_template=MARIN_CHAT_TEMPLATE,
        datasets=datasets,
        optimizer=_optimizer(),
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
    baseline_model = _eval_model("curriculum-math-sep20-base", HF_MODEL, HF_REVISION)
    trained_model = _eval_model("curriculum-math-sep20-trained", "<trained-hf>", None)

    def resolve_trained_model(ctx: StepContext) -> ModelConfig:
        # The trainer exports HF weights once, at the final (zero-indexed) step.
        final_export = prefix_join(ctx.artifact_path(trained), f"hf/step-{STEPS - 1}")
        return dataclasses.replace(trained_model, location=final_export)

    baseline: ArtifactStep[EvaluationResult] = eval_step(
        baseline_model,
        EVALS,
        version=version,
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    after: ArtifactStep[EvaluationResult] = eval_step(
        trained_model,
        EVALS,
        version=version,
        deps=(trained,),
        resolve_model=resolve_trained_model,
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    return {"baseline": baseline, "train": trained, "after": after}


@click.command()
@click.option("--stage", type=click.Choice(["generate", "baseline", "train", "after", "full"]), default="baseline")
@build_options
def main(stage: str) -> dict[str, ArtifactStep]:
    version = resolve_version("curriculum-math-sep20", None)
    if stage == "generate":
        return build_generation(version)
    trial = build_trial(version)
    if stage == "full":
        return {"baseline": trial["baseline"], "after": trial["after"]}
    return {stage: trial[stage]}


if __name__ == "__main__":
    main()
