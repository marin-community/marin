# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proportional Datakit SFT for the step-157k Grug 67B/A2B architecture."""

import dataclasses
import re
import tempfile
from dataclasses import dataclass

import click
from fray.types import ResourceConfig, TpuConfig
from levanter.kernels.pallas.splash_attention import SPLASH_BLOCK_GRANULARITY
from levanter.tracker.wandb import WandbConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.datakit.sft import SftInput, SftTokenStore, build_sft_store, sft_data_config
from marin.datakit.sft_sources import DatakitChatSource, all_sft_sources
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoTokenizer

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugMoeSFTConfig, run_grug_moe_sft_trial
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainerConfig
from experiments.marin_tokenizer import MARIN_CUSTOM_SPECIAL_TOKENS


@dataclass(frozen=True)
class DatakitSftConfig:
    init_checkpoint: str
    tokenizer: str
    tokenizer_revision: str
    run_id: str
    steps: int
    batch_size: int
    resources: ResourceConfig
    context_parallel: int = 4
    learning_rate: float = 5e-5
    sequence_length: int = 262_144
    seed: int = 0
    num_shards: int = 256
    max_workers: int = 128

    def __post_init__(self):
        if not isinstance(self.resources.device, TpuConfig):
            raise ValueError("The Datakit Grug recipe requires TPU resources")
        if not re.fullmatch(r"[0-9a-f]{40}", self.tokenizer_revision):
            raise ValueError("tokenizer_revision must be an immutable 40-character commit SHA")
        if min(self.steps, self.batch_size, self.num_shards, self.max_workers, self.context_parallel) < 1:
            raise ValueError("Training and preprocessing sizes must be positive")
        if self.sequence_length % (SPLASH_BLOCK_GRANULARITY * self.context_parallel):
            raise ValueError(f"Splash requires a multiple of {SPLASH_BLOCK_GRANULARITY} tokens per context shard")
        if self.sequence_length < 2 or self.learning_rate <= 0:
            raise ValueError("sequence_length must be >= 2 and learning_rate must be positive")


@dataclass(frozen=True)
class DatakitSftSteps:
    tokenizer: StepSpec
    store: StepSpec
    train: StepSpec


def _export_tokenizer(output_path: str, name: str, revision: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if len(tokenizer) != MoeMuonHHeuristic().vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match the 67B/A2B model")
    if tokenizer.bos_token != "<|begin_of_text|>" or tokenizer.eos_token != "<|end_of_text|>":
        raise ValueError("SFT requires the Marin BOS and document EOS tokens")
    for token_id, token in MARIN_CUSTOM_SPECIAL_TOKENS.items():
        if tokenizer.convert_tokens_to_ids(token) != token_id:
            raise ValueError(f"Tokenizer does not preserve Marin reasoning token {token}")
    tokenizer.chat_template = MARIN_CHAT_TEMPLATE
    # save_pretrained writes locally; upload only the small tokenizer files.
    with tempfile.TemporaryDirectory() as directory:
        tokenizer.save_pretrained(directory)
        source = StoragePath(prefix_join(directory, "*"))
        for path in source.glob():
            StoragePath(prefix_join(output_path, path.name)).write_bytes(path.read_bytes())


def sft_training_config(config: DatakitSftConfig, store: SftTokenStore, output_path: str) -> GrugMoeSFTConfig:
    """Bind the store's packing length to the reference model's training context."""
    if store.max_length != config.sequence_length:
        raise ValueError("SFT store and training context lengths must match")
    model = MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=config.sequence_length)
    model = dataclasses.replace(
        model,
        attention_implementation="tpu_splash",
        moe_implementation="ring",
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2048,
        use_array_stacked_blocks=True,
        qk_mult=1.75,
        max_seq_len=config.sequence_length,
    )
    return GrugMoeSFTConfig(
        model=model,
        data=sft_data_config(store),
        output_path=output_path,
        run_id=config.run_id,
        resources=config.resources,
        steps=config.steps,
        batch_size=config.batch_size,
        seed=config.seed,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(project="marin_moe_sft", tags=["sft", "datakit", "proportional"]),
        optimizer=GrugMoeMuonHConfig(
            learning_rate=config.learning_rate,
            adam_lr=config.learning_rate,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            max_grad_norm=None,
            weight_decay=0.0,
            min_lr_ratio=0.1,
            warmup=0.03,
            decay=0.1,
            lr_schedule="linear",
            rmsnorm_to_adam=True,
        ),
        init_from_path=config.init_checkpoint,
        context_parallel=config.context_parallel,
        grug_trainer=GrugTrainerConfig(replica_axis_size=1, z_loss_weight=1e-4),
    )


def build(config: DatakitSftConfig, sources: dict[str, DatakitChatSource] | None = None) -> DatakitSftSteps:
    """Build preprocessing and weights-only SFT steps; constructing the DAG runs no jobs."""
    sources = all_sft_sources() if sources is None else sources
    if not sources:
        raise ValueError("At least one SFT source is required")
    tokenizer = StepSpec(
        name="sft/tokenizer",
        fn=lambda output_path: _export_tokenizer(output_path, config.tokenizer, config.tokenizer_revision),
        hash_attrs={
            "tokenizer": config.tokenizer,
            "revision": config.tokenizer_revision,
            "template": MARIN_CHAT_TEMPLATE,
        },
    )
    normalized = {name: source.normalized for name, source in sorted(sources.items())}
    store = StepSpec(
        name="sft/proportional-store",
        fn=lambda output_path: build_sft_store(
            [SftInput(name, prefix_join(step.output_path, "outputs/main")) for name, step in normalized.items()],
            output_path=output_path,
            tokenizer=tokenizer.output_path,
            max_length=config.sequence_length,
            seed=config.seed,
            num_shards=config.num_shards,
            max_workers=config.max_workers,
        ),
        deps=[tokenizer, *normalized.values()],
        hash_attrs={
            "version": "v1",
            "max_length": config.sequence_length,
            "seed": config.seed,
            "num_shards": config.num_shards,
        },
    )

    def train(output_path: str) -> None:
        artifact = read_artifact(store.output_path, SftTokenStore)
        run_grug_moe_sft_trial(sft_training_config(config, artifact, output_path))

    training = StepSpec(
        name=f"sft/{config.run_id}", fn=train, deps=[store], hash_attrs={"config": dataclasses.asdict(config)}
    )
    return DatakitSftSteps(tokenizer=tokenizer, store=store, train=training)


@click.command()
@click.option("--init-checkpoint", required=True)
@click.option("--tokenizer", required=True)
@click.option("--tokenizer-revision", required=True)
@click.option("--run-id", required=True)
@click.option("--steps", type=int, required=True)
@click.option("--batch-size", type=int, required=True)
@click.option("--tpu", required=True)
@click.option("--zone", required=True)
@click.option("--sequence-length", type=int, default=262_144, show_default=True)
@click.option("--context-parallel", type=int, default=4, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--source", "source_names", multiple=True, help="Restrict to named sources; default includes all SFT sources."
)
@click.option("--stage", type=click.Choice(["store", "train"]), default="store", show_default=True)
@click.option("--run", is_flag=True, help="Execute the DAG. Otherwise print the resolved output paths.")
def main(
    init_checkpoint,
    tokenizer,
    tokenizer_revision,
    run_id,
    steps,
    batch_size,
    tpu,
    zone,
    sequence_length,
    context_parallel,
    seed,
    source_names,
    stage,
    run,
):
    sources = all_sft_sources()
    if source_names:
        sources = {name: sources[name] for name in source_names}
    config = DatakitSftConfig(
        init_checkpoint=init_checkpoint,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        run_id=run_id,
        steps=steps,
        batch_size=batch_size,
        resources=ResourceConfig.with_tpu(tpu, zone=zone, preemptible=False),
        sequence_length=sequence_length,
        context_parallel=context_parallel,
        seed=seed,
    )
    pipeline = build(config, sources)
    for name in ("tokenizer", "store", "train"):
        click.echo(f"{name}: {getattr(pipeline, name).output_path}")
    if run:
        StepRunner().run([pipeline.store if stage == "store" else pipeline.train], max_concurrent=1)


if __name__ == "__main__":
    main()
