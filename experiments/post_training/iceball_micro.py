# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iceball-micro: one Marin graph from random-init pretraining through evaluation.

The graph trains the Qwen3-0.6B architecture from scratch, applies a short chat SFT,
runs GSM8K GRPO through the pinned MarinSkyRL root package, then serves the
terminal policy once for Evalchemy GSM8K and Harbor AIME smoke evaluations.

Print or run the complete graph from the same entry point::

    python -m experiments.post_training.iceball_micro --version 2026.08.01
    python -m experiments.post_training.iceball_micro --version 2026.08.01 --run
    python -m experiments.post_training.iceball_micro --version 2026.08.01 --stage rl --run

Programmatic callers use :func:`build_workflow` and select any stage handle.

Run the same graph in Iris from a CPU coordinator::

    uv run iris --cluster=cw-us-east-08a job run --no-wait \
      -e DAYTONA_API_KEY "$DAYTONA_API_KEY" \
      -- python -m experiments.post_training.iceball_micro --version 2026.08.02 --run

The submitter resolves ``DAYTONA_API_KEY`` before launch because coordinator pods do not
receive cloud credentials. Iris redacts the value from the recorded submission command.
"""

from __future__ import annotations

import gzip
import itertools
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import Any

import click
import duckdb
from datasets import load_dataset
from fray.types import ANY_REGION, ResourceConfig
from huggingface_hub import HfFileSystem
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.data import tokenized
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.rl.skyrl import (
    SKYRL_POLICY_LOCATION,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLEvaluationModel,
    SkyRLModel,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import StoragePath, prefix_join
from zephyr.writers import write_parquet_file

from experiments.evaluation.pipeline import EvaluationResult, eval_step
from experiments.sft.launcher import DatasetSpec, LevanterCheckpointModel, SFTSpec, resources_from_accelerator, sft_step

logger = logging.getLogger(__name__)

ICEBALL_MODEL_NAME = "iceball-micro"
QWEN_TOKENIZER = "Qwen/Qwen3-0.6B-Base"
QWEN_TOKENIZER_REVISION = "da87bfb"
FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_REVISION = "87f0914"
FINEWEB_SUBSET = "sample-10BT"
FINEWEB_DATA_DIR = "sample/10BT"
FINEWEB_ROWS = 4096
GSM8K_DATASET = "openai/gsm8k"
GSM8K_REVISION = "e53f048"
GSM8K_TRAIN_ROWS = 1024
GSM8K_VALIDATION_ROWS = 128
ICEBALL_EVALS = "gsm8k-smoke,aime-smoke"
ICEBALL_CLUSTER = "cw-us-east-08a"
ICEBALL_CLUSTER_CONFIG = f"lib/iris/config/{ICEBALL_CLUSTER}.yaml"
ICEBALL_GPU_VARIANT = "GB200"
ICEBALL_TRAIN_ACCELERATOR = f"4x{ICEBALL_GPU_VARIANT}"
ICEBALL_EVAL_ACCELERATOR = f"{ICEBALL_GPU_VARIANT}x1"
ICEBALL_SEQUENCE_LENGTH = 512
ICEBALL_WANDB_PROJECT = f"marin-{ICEBALL_MODEL_NAME}"
FINEWEB_ARTIFACT_NAME = f"documents/{ICEBALL_MODEL_NAME}-fineweb-edu"
FINEWEB_TOKENIZED_ARTIFACT_NAME = f"tokenized/{ICEBALL_MODEL_NAME}-fineweb-edu-qwen3"
FINEWEB_TRAIN_FILENAME = "train.jsonl.gz"
GSM8K_ARTIFACT_NAME = f"documents/{ICEBALL_MODEL_NAME}-gsm8k-skyrl"
GSM8K_TRAIN_FILENAME = "train.parquet"
GSM8K_VALIDATION_FILENAME = "validation.parquet"

_FINAL_ANSWER = re.compile(r"####\s*(-?[0-9.,]+)")
_DATA_RESOURCES = ResourceConfig.with_cpu(cpu=4, ram="32g", disk="32g")
_TRAIN_RESOURCES = ResourceConfig.with_gpu(
    ICEBALL_GPU_VARIANT,
    count=4,
    cpu=64,
    ram="512g",
    disk="256g",
    regions=[ANY_REGION],
)

ICEBALL_QWEN3_CONFIG = Qwen3Config(
    max_seq_len=ICEBALL_SEQUENCE_LENGTH,
    hidden_dim=1024,
    intermediate_dim=3072,
    num_heads=16,
    head_dim=128,
    num_kv_heads=8,
    num_layers=28,
    layer_norm_epsilon=1e-6,
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000),
    tie_word_embeddings=True,
    use_sliding_window=False,
    tokenizer=QWEN_TOKENIZER,
    reference_checkpoint=QWEN_TOKENIZER,
)

# Keep policy/reference and rollout workers on separate nodes so this integration run
# exercises the multi-node handoff and mixed-role topology used by larger RL jobs.
ICEBALL_RL_ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=1,
    policy_num_gpus_per_node=4,
    num_inference_engines=4,
    inference_engine_tensor_parallel_size=1,
    train_batch_size=16,
    policy_mini_batch_size=16,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=4,
)

QWEN3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}<|im_end|>{% endgeneration %}\n"
    "{% else %}{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
)

ICEBALL_RL_CONFIG = f"""\
entrypoint: skyrl_train.entrypoints.main_base

context_budget:
  request_window_tokens: {ICEBALL_SEQUENCE_LENGTH}
  max_new_tokens_per_turn: 256
  max_turns: 1

environment:
  env_class: gsm8k

trainer:
  strategy: fsdp2
  flash_attn: false
  use_sample_packing: false
  algorithm:
    advantage_estimator: grpo
    use_kl_loss: true
  epochs: 1
  max_steps: 8
  update_epochs_per_batch: 1
  train_batch_size: {ICEBALL_RL_ROLE_PLAN.train_batch_size}
  policy_mini_batch_size: {ICEBALL_RL_ROLE_PLAN.policy_mini_batch_size}
  eval_batch_size: 16
  micro_forward_batch_size_per_gpu: 2
  micro_train_batch_size_per_gpu: {ICEBALL_RL_ROLE_PLAN.micro_train_batch_size_per_gpu}
  eval_before_train: false
  eval_interval: -1
  ckpt_interval: 2
  hf_save_interval: 8
  resume_mode: latest
  logger: wandb
  project_name: {ICEBALL_WANDB_PROJECT}
  policy:
    optimizer_config:
      lr: 2.0e-6
      max_grad_norm: 1.0
    fsdp_config:
      cpu_offload: false
      reshard_after_forward: true
  placement:
    colocate_all: {str(ICEBALL_RL_ROLE_PLAN.colocate_all).lower()}

generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  inference_engine_tensor_parallel_size: {ICEBALL_RL_ROLE_PLAN.inference_engine_tensor_parallel_size}
  num_inference_engines: {ICEBALL_RL_ROLE_PLAN.num_inference_engines}
  n_samples_per_prompt: {ICEBALL_RL_ROLE_PLAN.n_samples_per_prompt}
  gpu_memory_utilization: 0.70
  enforce_eager: true
  run_engines_locally: true
  weight_sync_backend: nccl
  async_engine: true
  batched: true
  sampling_params:
    temperature: 1.0
    top_p: 1.0

data:
  kind: parquet
  train_data: []
  val_data: []
"""


@dataclass(frozen=True)
class FineWebSliceConfig:
    output_path: str
    dataset: str = FINEWEB_DATASET
    revision: str = FINEWEB_REVISION
    subset: str = FINEWEB_SUBSET
    data_dir: str = FINEWEB_DATA_DIR
    rows: int = FINEWEB_ROWS


@dataclass(frozen=True)
class Gsm8kParquetConfig:
    output_path: str
    dataset: str = GSM8K_DATASET
    revision: str = GSM8K_REVISION
    train_rows: int = GSM8K_TRAIN_ROWS
    validation_rows: int = GSM8K_VALIDATION_ROWS


@dataclass(frozen=True)
class IceballMicroWorkflow:
    fineweb: ArtifactStep[TokenizedCache]
    pretrain: ArtifactStep[LevanterCheckpoint]
    sft: ArtifactStep[LevanterCheckpoint]
    gsm8k: ArtifactStep[Artifact]
    rl: ArtifactStep[SkyRLModel]
    evaluation: ArtifactStep[EvaluationResult]


ICEBALL_STAGES = tuple(field.name for field in fields(IceballMicroWorkflow))


def write_fineweb_slice(config: FineWebSliceConfig) -> None:
    """Stream an exact pinned FineWeb-Edu prefix into a compact JSONL artifact."""
    destination = prefix_join(config.output_path, FINEWEB_TRAIN_FILENAME)
    with StoragePath(destination).open("wb") as raw_destination:
        with gzip.GzipFile(fileobj=raw_destination, mode="wb", mtime=0) as compressed:
            for text in itertools.islice(_fineweb_texts(config), config.rows):
                compressed.write((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode())
    logger.info("Wrote %d FineWeb-Edu rows to %s", config.rows, destination)


def _fineweb_texts(config: FineWebSliceConfig) -> Iterator[str]:
    filesystem = HfFileSystem()
    pattern = str(StoragePath("datasets") / config.dataset / config.data_dir / "*.parquet")
    shards = sorted(filesystem.glob(pattern, revision=config.revision))
    if not shards:
        raise ValueError(f"No parquet files matched {config.dataset}@{config.revision}/{config.subset}")

    remaining = config.rows
    with duckdb.connect() as connection:
        connection.register_filesystem(filesystem)
        for shard in shards:
            rows = connection.execute(
                "SELECT text FROM read_parquet($shard) LIMIT $row_limit",
                {"shard": f"hf://{shard}", "row_limit": remaining},
            ).fetchall()
            yield from (text for (text,) in rows)
            remaining -= len(rows)
            if remaining == 0:
                return
    raise ValueError(f"Requested {config.rows} FineWeb-Edu rows but the pinned source ended early")


def _gsm8k_record(example: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    match = _FINAL_ANSWER.search(example["answer"])
    if match is None:
        raise ValueError(f"GSM8K row {split}/{index} has no final answer marker")
    answer = match.group(1).replace(",", "")
    question = example["question"]
    return {
        "data_source": GSM8K_DATASET,
        "prompt": [
            {
                "role": "user",
                "content": f'{question} Let\'s think step by step and output the final answer after "####".',
            }
        ],
        "env_class": "gsm8k",
        "reward_spec": {"method": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": index,
            "answer": example["answer"],
            "question": question,
        },
    }


def write_gsm8k_parquet(config: Gsm8kParquetConfig) -> None:
    """Write deterministic train/test prefixes in SkyRL's typed parquet schema."""
    dataset = load_dataset(config.dataset, "main", revision=config.revision)
    selections = (
        ("train", config.train_rows, GSM8K_TRAIN_FILENAME),
        ("test", config.validation_rows, GSM8K_VALIDATION_FILENAME),
    )
    for split, count, filename in selections:
        source = dataset[split].select(range(count))
        records = [_gsm8k_record(dict(example), split, index) for index, example in enumerate(source)]
        destination = prefix_join(config.output_path, filename)
        write_parquet_file(records, destination)
        logger.info("Wrote %d GSM8K rows to %s", len(records), destination)


def _fineweb_step(version: str) -> ArtifactStep[TokenizedCache]:
    raw = ArtifactStep(
        name=FINEWEB_ARTIFACT_NAME,
        version=version,
        artifact_type=Artifact,
        run=remote(write_fineweb_slice, resources=_DATA_RESOURCES),
        build_config=lambda ctx: FineWebSliceConfig(output_path=ctx.output_path),
    )
    return tokenized(
        FINEWEB_TOKENIZED_ARTIFACT_NAME,
        version=version,
        tokenizer=QWEN_TOKENIZER,
        raw=raw,
        glob=FINEWEB_TRAIN_FILENAME,
        tags=(ICEBALL_MODEL_NAME, "fineweb-edu"),
    )


def _gsm8k_step(version: str) -> ArtifactStep[Artifact]:
    return ArtifactStep(
        name=GSM8K_ARTIFACT_NAME,
        version=version,
        artifact_type=Artifact,
        run=remote(write_gsm8k_parquet, resources=_DATA_RESOURCES),
        build_config=lambda ctx: Gsm8kParquetConfig(output_path=ctx.output_path),
    )


def build_workflow(*, version: str | None = None) -> IceballMicroWorkflow:
    """Compose every iceball-micro stage as one inspectable artifact graph."""
    fineweb_version = version or resolve_version(FINEWEB_TOKENIZED_ARTIFACT_NAME, None)
    fineweb = _fineweb_step(fineweb_version)
    pretrain_name = f"checkpoints/{ICEBALL_MODEL_NAME}-pretrain"
    pretrain = train_lm(
        name=pretrain_name,
        version=version or resolve_version(pretrain_name, None),
        run_id=f"{ICEBALL_MODEL_NAME}-pretrain",
        model=ICEBALL_QWEN3_CONFIG,
        optimizer=AdamConfig(
            learning_rate=3e-4,
            weight_decay=0.1,
            lr_schedule="cosine",
            warmup=0.1,
            min_lr_ratio=0.1,
        ),
        datasets={fineweb: 1.0},
        batch_size=32,
        seq_len=ICEBALL_SEQUENCE_LENGTH,
        num_train_steps=16,
        z_loss_weight=1e-4,
        evals=None,
        resources=_TRAIN_RESOURCES,
        wandb_project=ICEBALL_WANDB_PROJECT,
        tags=(ICEBALL_MODEL_NAME, "pretrain", "qwen3"),
    )

    sft_name = f"checkpoints/{ICEBALL_MODEL_NAME}-sft"
    sft_spec = SFTSpec(
        name=sft_name,
        version=version or resolve_version(sft_name, None),
        model=LevanterCheckpointModel(
            init_from=pretrain,
            model=ICEBALL_QWEN3_CONFIG,
            tokenizer_path=QWEN_TOKENIZER,
            eos_token_ids=(151643, 151645),
        ),
        chat_template=QWEN3_CHAT_TEMPLATE,
        datasets=(
            DatasetSpec(
                slug="norobots",
                hf_dataset_id="HuggingFaceH4/no_robots",
                revision="e6f9a4a",
                adapter_kwargs={"conversation_column": "messages"},
                weight=1.0,
            ),
        ),
        optimizer=AdamConfig(
            learning_rate=1e-5,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-8,
            max_grad_norm=1.0,
            weight_decay=0.0,
            lr_schedule="cosine",
            warmup=0.1,
            min_lr_ratio=0.0,
        ),
        seq_len=ICEBALL_SEQUENCE_LENGTH,
        batch_size=16,
        num_train_steps=8,
        wandb_project=ICEBALL_WANDB_PROJECT,
    )
    sft = sft_step(sft_spec, resources_from_accelerator(ICEBALL_TRAIN_ACCELERATOR))

    gsm8k = _gsm8k_step(version or resolve_version(GSM8K_ARTIFACT_NAME, None))
    rl_name = f"checkpoints/{ICEBALL_MODEL_NAME}-rl"
    rl = skyrl_step(
        SkyRLSpec(
            name=rl_name,
            version=version or resolve_version(rl_name, None),
            config_yaml=ICEBALL_RL_CONFIG,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.FSDP),
            model=ArtifactHfModel(
                step=sft,
                tokenizer_uri=QWEN_TOKENIZER,
                tokenizer_revision=QWEN_TOKENIZER_REVISION,
            ),
            train_data=(ArtifactDataSource(gsm8k, relative_path=GSM8K_TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(gsm8k, relative_path=GSM8K_VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=2,
                gpus_per_node=4,
                gpu_variant=ICEBALL_GPU_VARIANT,
                role_plan=ICEBALL_RL_ROLE_PLAN,
            ),
            seed=17,
        ),
        IrisSkyRLExecution(
            cluster=ICEBALL_CLUSTER,
            cluster_config=ICEBALL_CLUSTER_CONFIG,
            cpu=16,
            memory="256GB",
            disk="4TB",
            priority="interactive",
            max_retries=3,
            wandb_entity="marin-community",
        ),
    )

    evaluation_name = f"evals/{ICEBALL_MODEL_NAME}/{ICEBALL_EVALS}"
    evaluation = eval_step(
        SkyRLEvaluationModel(
            step=rl,
            model=ModelConfig(
                name=ICEBALL_MODEL_NAME,
                location=SKYRL_POLICY_LOCATION,
                tokenizer=QWEN_TOKENIZER,
                apply_chat_template=True,
                resource_hint=ResourceHint(gpu={ICEBALL_GPU_VARIANT: 1}),
                serve=ServeConfig(
                    tensor_parallel_size=1,
                    max_model_len=ICEBALL_SEQUENCE_LENGTH,
                    max_num_seqs=32,
                    vllm_extra_args=("--enforce-eager",),
                ),
                generation=GenerationConfig(max_gen_toks=256),
            ),
        ),
        ICEBALL_EVALS,
        version=version or resolve_version(evaluation_name, None),
        accelerator=ICEBALL_EVAL_ACCELERATOR,
        submission_cluster=ICEBALL_CLUSTER,
        federated_cluster=ICEBALL_CLUSTER,
    )
    return IceballMicroWorkflow(
        fineweb=fineweb,
        pretrain=pretrain,
        sft=sft,
        gsm8k=gsm8k,
        rl=rl,
        evaluation=evaluation,
    )


@click.command(help=__doc__)
@click.option(
    "--stage",
    type=click.Choice(ICEBALL_STAGES),
    default="evaluation",
    show_default=True,
    help="Terminal workflow stage to plan or run; its dependencies are included automatically.",
)
@build_options
def main(stage: str) -> ArtifactStep:
    workflow = build_workflow()
    stages: dict[str, ArtifactStep] = {stage_name: getattr(workflow, stage_name) for stage_name in ICEBALL_STAGES}
    return stages[stage]


if __name__ == "__main__":
    main()
