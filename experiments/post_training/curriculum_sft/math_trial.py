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
from marin.evaluation.hardware import AcceleratorChoice, Platform
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
from experiments.post_training.curriculum_sft.generation import (
    CHAT_FILENAME,
    generate_curriculum_problems,
    solve_curriculum_problems,
)
from experiments.post_training.curriculum_sft.self_distill import SELF_CHAT_FILENAME, self_distill_step
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
# Problems and solutions carry separate versions so a solve-recipe change reuses the accepted problems.
PROBLEMS_VERSION = "2026.09.26.1"
SOLUTIONS_VERSION = "2026.09.26.3"
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
REQUESTED_PROBLEMS_PER_CAPABILITY = 320
SAMPLES_PER_PROBLEM = 4
SOLUTIONS_PER_PROBLEM = 1
SEED = 17
PROBLEM_MAX_COMPLETION_TOKENS = 16384
SOLUTION_MAX_COMPLETION_TOKENS = 32768
SELF_DISTILL_VERSION = "2026.09.26.1"
SELF_DISTILL_TEMPERATURE = 0.7
# Leaves room for the prompt and template inside CONTEXT_LENGTH; longer samples are rejected anyway.
SELF_DISTILL_MAX_COMPLETION_TOKENS = 3584
TASK_SPECIFICATION = (
    "Target the difficulty of MATH levels 3-5 and AMC 12. Vary the setting, the quantities, and the "
    "structure across problems; do not default to solving one radical or logarithmic equation."
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


def build_generation() -> dict[str, ArtifactStep[Artifact]]:
    """Build GLM problem and blind-solve steps; run them on `cw-us-east-08a`, where the GLM relay is reachable."""
    steps: dict[str, ArtifactStep[Artifact]] = {}
    for capability_id in CURRICULUM_IDS:
        problems = generate_curriculum_problems(
            capability_id,
            version=PROBLEMS_VERSION,
            requested_problems=REQUESTED_PROBLEMS_PER_CAPABILITY,
            seed=SEED,
            max_completion_tokens=PROBLEM_MAX_COMPLETION_TOKENS,
            task_specification=TASK_SPECIFICATION,
        )
        steps[capability_id] = solve_curriculum_problems(
            problems,
            capability_id=capability_id,
            version=SOLUTIONS_VERSION,
            samples_per_problem=SAMPLES_PER_PROBLEM,
            solutions_per_problem=SOLUTIONS_PER_PROBLEM,
            tokenizer=HF_MODEL,
            tokenizer_revision=HF_REVISION,
            max_sequence_tokens=CONTEXT_LENGTH,
            seed=SEED,
            max_completion_tokens=SOLUTION_MAX_COMPLETION_TOKENS,
        )
    return steps


def _adopted_problems() -> dict[str, ArtifactStep[Artifact]]:
    sources: dict[str, ArtifactStep[Artifact]] = {}
    for capability_id in CURRICULUM_IDS:
        problems_name = user_owned_name(f"documents/curriculum-sft/{capability_id}/problems")
        sources[capability_id] = ArtifactStep.adopt(
            name=user_owned_name(f"documents/curriculum-sft/{capability_id}/staged-problems"),
            version=PROBLEMS_VERSION,
            source=prefix_join(prefix_join(SOURCE_PREFIX, problems_name), PROBLEMS_VERSION),
            kind=Artifact,
        )
    return sources


def build_self_distill() -> ArtifactStep[Artifact]:
    """Sample the base checkpoint on the accepted problems and keep its own verified solutions."""
    return self_distill_step(
        _adopted_problems(),
        name="documents/curriculum-sft/math-self-distill",
        version=SELF_DISTILL_VERSION,
        model=_eval_model("curriculum-math-sep20-self-distill", HF_MODEL, HF_REVISION),
        accelerator=AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8, target_cluster=CLUSTER),
        samples_per_problem=SAMPLES_PER_PROBLEM,
        solutions_per_problem=SOLUTIONS_PER_PROBLEM,
        temperature=SELF_DISTILL_TEMPERATURE,
        max_completion_tokens=SELF_DISTILL_MAX_COMPLETION_TOKENS,
        max_sequence_tokens=CONTEXT_LENGTH,
        seed=SEED,
    )


def _datasets(data: str) -> list[ArtifactDatasetSpec]:
    if data == "self":
        distilled = build_self_distill()
        return [
            ArtifactDatasetSpec(
                slug=capability_id,
                artifact=distilled,
                train_glob=SELF_CHAT_FILENAME.format(capability_id=capability_id),
                weight=1.0,
            )
            for capability_id in CURRICULUM_IDS
        ]
    generated = _staged_generation()
    return [
        ArtifactDatasetSpec(slug=capability_id, artifact=generated[capability_id], train_glob=CHAT_FILENAME, weight=1.0)
        for capability_id in CURRICULUM_IDS
    ]


def _staged_generation() -> dict[str, ArtifactStep[Artifact]]:
    sources: dict[str, ArtifactStep[Artifact]] = {}
    for capability_id in CURRICULUM_IDS:
        name = user_owned_name(f"documents/curriculum-sft/{capability_id}/staged-solved-chat")
        solved_name = user_owned_name(f"documents/curriculum-sft/{capability_id}/solved-chat")
        sources[capability_id] = ArtifactStep.adopt(
            name=name,
            version=SOLUTIONS_VERSION,
            source=prefix_join(prefix_join(SOURCE_PREFIX, solved_name), SOLUTIONS_VERSION),
            kind=Artifact,
        )
    return sources


def build_trial(version: str, learning_rate: float, warmup: int, data: str) -> dict[str, ArtifactStep]:
    """Bind baseline and trained evaluations to Levanter's Snowball SFT run.

    Args:
        version: Version shared by the SFT checkpoint and both evaluations.
        learning_rate: Peak Adam learning rate; 0 exercises the train/export/eval path without updates.
        warmup: Linear warmup length in optimizer steps.
        data: ``glm`` trains on GLM-solved rows; ``self`` trains on the base model's own verified samples.
    """
    curriculum_key = hashlib.sha256(json.dumps(sorted(CURRICULUM_IDS)).encode()).hexdigest()[:12]
    conversion = hf_to_levanter(
        HF_MODEL,
        model_type="snowball",
        hf_revision=HF_REVISION,
        tokenizer=f"hf://{HF_MODEL}@{HF_REVISION}",
        version=CONVERSION_VERSION,
        resources=ResourceConfig.with_cpu(cpu=64, ram="512g", disk="256g"),
    )
    datasets = _datasets(data)
    spec = SFTSpec(
        name=user_owned_name(f"checkpoints/curriculum-sft/{curriculum_key}/snowball{'-self' if data == 'self' else ''}"),
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
@click.option(
    "--stage", type=click.Choice(["generate", "distill", "baseline", "train", "after", "full"]), default="baseline"
)
@click.option("--data", type=click.Choice(["glm", "self"]), default="glm", help="Training rows for train/after.")
@click.option("--learning-rate", type=float, help="Peak Adam learning rate; required except for generation.")
@click.option("--warmup", type=int, help="Linear warmup steps; required except for generation.")
@build_options
def main(stage: str, data: str, learning_rate: float | None, warmup: int | None) -> dict[str, ArtifactStep]:
    if stage == "generate":
        return build_generation()
    if stage == "distill":
        return {"distill": build_self_distill()}
    version = resolve_version("curriculum-math-sep20", None)
    if learning_rate is None or warmup is None:
        raise click.UsageError(f"--stage {stage} requires --learning-rate and --warmup")
    trial = build_trial(version, learning_rate, warmup, data)
    if stage == "full":
        return {"baseline": trial["baseline"], "after": trial["after"]}
    return {stage: trial[stage]}


if __name__ == "__main__":
    main()
