# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distill a reusable EAGLE-3 draft for the Snowball SFT checkpoint.

Evalchemy generates a durable long-form math conversation corpus once. Later draft
versions replay those cached responses through the target as bulk prefills,
then train and validate the EAGLE head from the ephemeral verifier features.
"""

from __future__ import annotations

import click
from marin.evaluation.evalchemy.config import EvalchemyConfig, load_evalchemy_config
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ResourceHint, ServeConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.evaluation import evaluate_evalchemy
from marin.experiment.namespacing import user_owned_name
from marin.rl.eagle import EagleRolloutCorpus, eagle_rollout_corpus_step
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    EagleDraftDistillationSpec,
    EagleDraftModel,
    EagleDraftSource,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLTopology,
    eagle_draft_distillation_step,
)

from experiments.evaluation.evals import EVALS, EvalchemyDefinition, evalchemy_run_config
from experiments.post_training.curriculum_rl.launch import (
    MARIN_TOKENIZER,
    MARIN_TOKENIZER_REVISION,
    SNOWBALL_MODEL,
)

ARTIFACT_NAME = "models/snowball-67b-a2b-eagle3-distilled"
ROLLOUT_VERSION = "2026.09.20"
CORPUS_ARTIFACT_NAME = "data/snowball-67b-a2b-eagle3-math-corpus"
CORPUS_VERSION = "2026.09.20"
INITIAL_DRAFT_URI = "hf://laion/snowball-64k-eagle3-draft-r2egym"
INITIAL_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
CLUSTER = "cw-rno2a"
GPUS_PER_NODE = 8
MARINSKYRL_COMMIT = "b304c134c87797168b7bba0cb01c56d263950c9c"

DISTILLATION_CONFIG = """
entrypoint: generate

context_budget:
  request_window_tokens: 32768
  max_new_tokens_per_turn: 1
  max_turns: 1

environment:
  env_class: aime

trainer:
  strategy: megatron
  flash_attn: false
  use_sample_packing: false
  algorithm:
    advantage_estimator: grpo
    use_kl_loss: false
  train_batch_size: 64
  policy_mini_batch_size: 64
  eval_batch_size: 4096
  micro_forward_batch_size_per_gpu: 1
  micro_train_batch_size_per_gpu: 1
  logger: console
  project_name: snowball-eagle3-offline-distillation
  placement:
    colocate_all: false
    policy_num_nodes: 1
    policy_num_gpus_per_node: 8

generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  inference_engine_tensor_parallel_size: 1
  inference_engine_pipeline_parallel_size: 1
  # Keep expert collectives within one eight-GPU node while retaining 64
  # independent DP schedulers across the eight engines.
  inference_engine_data_parallel_size: 8
  inference_engine_expert_parallel_size: 8
  num_inference_engines: 8
  n_samples_per_prompt: 16
  eval_n_samples_per_prompt: 4
  gpu_memory_utilization: 0.75
  max_num_seqs: 16
  max_num_batched_tokens: 16384
  enforce_eager: false
  run_engines_locally: true
  weight_sync_backend: nccl
  async_engine: true
  batched: false
  engine_init_kwargs:
    async_scheduling: false
  speculative_decoding:
    method: eagle3
    model: {}
    num_speculative_tokens: 3
    training:
      interval_steps: 1
      max_tokens_per_update: 1000000
      max_window_tokens: 16384
      max_tokens_per_micro_batch: 8192
      max_sequences_per_prompt_group: 2
      min_train_sequences: 32
      holdout_fraction: 0.1
      min_holdout_sequences: 16
      epochs_per_update: 4
      optimizer: hybrid_muon
      learning_rate: 1.0e-5
      muon_learning_rate: 0.02
      max_validation_loss_increase: 0.05
      max_validation_agreement_decrease: 0.01
      reserved_gpu_memory_gib: 12
  sampling_params:
    temperature: 1.0
    top_p: 1.0
  eval_sampling_params:
    temperature: 1.0
    top_p: 1.0

data:
  # The launcher's record-file route stages both Parquet and JSONL inputs.
  kind: parquet
  train_data: []
  val_data: []
  shuffle: false
  eagle_replay: true
  eagle_replay_batch_size: 4096

extra_env:
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
"""


def build_rollout_corpus() -> ArtifactStep[EagleRolloutCorpus]:
    """Generate long-form math responses once and normalize them into replay JSONL."""
    sources = []
    for name in ("aime24", "math500", "olympiadbench"):
        definition = EVALS[name]
        assert isinstance(definition, EvalchemyDefinition)
        sources.append(load_evalchemy_config(definition.config_path))
    evalchemy_config = EvalchemyConfig(
        tasks=tuple(task for source in sources for task in source.tasks),
        task_options={name: options for source in sources for name, options in source.task_options.items()},
        apply_chat_template=True,
        limit=128,
        runtime_extras=tuple(extra for source in sources for extra in source.runtime_extras),
    )
    evaluation = evaluate_evalchemy(
        model_name="snowball-67b-a2b-sft-s2-thinking",
        model=SNOWBALL_MODEL,
        config=evalchemy_run_config("snowball-eagle-math", evalchemy_config),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=8,
            max_model_len=32768,
            max_num_batched_tokens=16384,
            max_num_seqs=16,
            vllm_extra_args=("--enable-expert-parallel",),
        ),
        resource_hint=ResourceHint(gpu={"H100": 8}, cpu=64, memory="512GB", disk="2TB"),
        accelerator=AcceleratorChoice(
            platform=Platform.GPU,
            gpu_type="H100",
            gpu_count=8,
        ),
        tokenizer=MARIN_TOKENIZER,
        discover_latest_checkpoint=False,
        version=ROLLOUT_VERSION,
    )
    return eagle_rollout_corpus_step(
        name=user_owned_name(CORPUS_ARTIFACT_NAME),
        version=CORPUS_VERSION,
        evaluations=(evaluation,),
    )


def build_distillation(
    version: str | None = None,
    *,
    corpus: ArtifactStep[EagleRolloutCorpus] | None = None,
) -> ArtifactStep[EagleDraftModel]:
    """Build the Snowball EAGLE distillation artifact."""
    resolved_version = version or resolve_version(ARTIFACT_NAME, None)
    corpus = corpus or build_rollout_corpus()
    return eagle_draft_distillation_step(
        EagleDraftDistillationSpec(
            name=user_owned_name(ARTIFACT_NAME),
            version=resolved_version,
            config_yaml=DISTILLATION_CONFIG,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON, commit=MARINSKYRL_COMMIT),
            target_model=ArtifactHfModel(
                step=SNOWBALL_MODEL,
                tokenizer_uri=MARIN_TOKENIZER,
                tokenizer_revision=MARIN_TOKENIZER_REVISION,
            ),
            initial_draft=EagleDraftSource(uri=INITIAL_DRAFT_URI, identity=INITIAL_DRAFT_REVISION),
            data=(ArtifactDataSource(corpus, relative_path="corpus.jsonl"),),
            topology=SkyRLTopology(
                num_nodes=8,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant="H100",
                role_plan=SkyRLRolePlan(
                    colocate_all=False,
                    policy_num_nodes=1,
                    policy_num_gpus_per_node=GPUS_PER_NODE,
                    num_inference_engines=8,
                    inference_engine_tensor_parallel_size=1,
                    inference_engine_pipeline_parallel_size=1,
                    inference_engine_data_parallel_size=8,
                    inference_engine_expert_parallel_size=8,
                    train_batch_size=64,
                    policy_mini_batch_size=64,
                    micro_train_batch_size_per_gpu=1,
                    n_samples_per_prompt=16,
                    evaluation_only=True,
                ),
            ),
            retention=SkyRLRetentionPolicy(),
            seed=17,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=16,
            memory="512GB",
            disk="2TB",
            priority="interactive",
            max_retries=1,
            wandb_entity="marin-community",
        ),
    )


@click.command(help=__doc__)
@build_options
def main() -> dict[str, ArtifactStep]:
    corpus = build_rollout_corpus()
    draft = build_distillation(corpus=corpus)
    return {"corpus": corpus, "draft": draft}


if __name__ == "__main__":
    main()
