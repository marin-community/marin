# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge-free math curriculum SFT trial on the September Grug checkpoint."""

import dataclasses

import click
from fray.types import ResourceConfig
from levanter.optim.config import OptimizerConfig
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.models import SNOWBALL_VLLM_ARGS
from experiments.evaluation.pipeline import EvaluationResult, eval_step
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeAdamHConfig
from experiments.post_training.curriculum_sft.grug_pipeline import (
    GRUG_CHECKPOINTS_DIR,
    CurriculumGenerationSpec,
    curriculum_generation_steps,
    curriculum_grug_sft,
    september_grug_model,
)
from experiments.post_training.curriculum_sft.hf_export import grug_hf_export
from experiments.post_training.curriculum_sft.hf_import import snowball_hf_to_grug

HF_MODEL = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.20"
HF_REVISION = "9f2ee50f3d4a12c79b0808bb2414ddba2cdf0098"
CLUSTER = "cw-rno2a"
S3_TRIAL_PREFIX = "s3://marin-us-east-02a/tmp/ttl=30d/curriculum-math-20260924"
SOURCE_VERSION = "2026.09.24"
CURRICULUM_IDS = (
    "d01.algebra.exact-symbolic-evaluation",
    "d01.algebra.scalar-equations",
    "d01.algebra.number-theoretic-reasoning",
)
EVALS = "olympiadbench-deterministic,math500"
CONTEXT_LENGTH = 4096
EXPORT_CONTEXT_LENGTH = 262144
BATCH_SIZE = 64
STEPS = 4
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


def _optimizer() -> OptimizerConfig:
    return GrugMoeAdamHConfig(
        learning_rate=5e-5,
        adam_lr=5e-5,
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
        generated_path = prefix_join(prefix_join(S3_TRIAL_PREFIX, generated_name), version)
        sources[capability_id] = ArtifactStep.adopt(
            name=name,
            version=version,
            source=generated_path,
            kind=Artifact,
        )
    return sources


def build_trial(version: str) -> dict[str, ArtifactStep]:
    """Bind a baseline evaluation, packed SFT, export, and matched re-evaluation."""
    training_model = september_grug_model(CONTEXT_LENGTH)
    imported = snowball_hf_to_grug(
        HF_MODEL,
        hf_revision=HF_REVISION,
        model=training_model,
        version=SOURCE_VERSION,
        resources=_gpu_resources(1),
    )
    trained: ArtifactStep[LevanterCheckpoint] = curriculum_grug_sft(
        CURRICULUM_IDS,
        version=version,
        generation=GENERATION,
        generated=_staged_generation(SOURCE_VERSION),
        checkpoint=imported,
        checkpoint_subpath=GRUG_CHECKPOINTS_DIR,
        tokenizer=HF_MODEL,
        optimizer=_optimizer(),
        resources=_gpu_resources(8),
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        expert_parallel=8,
    )
    exported: ArtifactStep[Artifact] = grug_hf_export(
        trained,
        model=dataclasses.replace(training_model, max_seq_len=EXPORT_CONTEXT_LENGTH),
        tokenizer=HF_MODEL,
        version=version,
        resources=_gpu_resources(1),
    )
    baseline_model = _eval_model("curriculum-math-sep20-base", HF_MODEL, HF_REVISION)
    trained_model = _eval_model("curriculum-math-sep20-trained", "<export>", None)

    def resolve_trained_model(ctx: StepContext) -> ModelConfig:
        return dataclasses.replace(trained_model, location=ctx.artifact_path(exported))

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
        deps=(exported,),
        resolve_model=resolve_trained_model,
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    return {"baseline": baseline, "train": trained, "export": exported, "after": after}


@click.command()
@click.option(
    "--stage", type=click.Choice(["generate", "baseline", "train", "export", "after", "full"]), default="baseline"
)
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
