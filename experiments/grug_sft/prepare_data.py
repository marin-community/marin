# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the per-source 262k Datakit SFT stores for the Grug data audit."""

import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.datakit.sft import SftInput, SftTokenStore, build_sft_store
from marin.datakit.sft_sources import all_sft_sources
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.cli import experiment_main
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
TOKENIZER_ID = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
CONTEXT_LENGTH = 262_144
SHUFFLE_SEED = 0
TARGET_TOKENS_PER_SHARD = 62_500_000


@dataclass(frozen=True)
class TokenizerConfig:
    output_path: str
    tokenizer_id: str
    revision: str
    chat_template: str


def export_tokenizer(config: TokenizerConfig) -> Artifact:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id, revision=config.revision)
    tokenizer.chat_template = config.chat_template
    expected = {
        "<|begin_of_text|>": 128000,
        "<|end_of_text|>": 128001,
        "<|start_think|>": 128002,
        "<|end_think|>": 128003,
        "<|eot_id|>": 128009,
    }
    for token, token_id in expected.items():
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Tokenizer changed the ID for {token}")
    if len(tokenizer) != 128256:
        raise ValueError("Tokenizer vocabulary does not match the Grug checkpoint")
    with tempfile.TemporaryDirectory() as directory:
        tokenizer.save_pretrained(directory)
        for path in Path(directory).iterdir():
            if path.is_file():
                StoragePath(prefix_join(config.output_path, path.name)).write_bytes(path.read_bytes())
    return Artifact(path=config.output_path)


@dataclass(frozen=True)
class SourceConfig:
    output_path: str
    name: str
    normalized_version: str
    tokenizer: str
    context_length: int
    seed: int
    shards: int
    workers: int


def prepare_source(config: SourceConfig) -> SftTokenStore:
    source = all_sft_sources()[config.name]
    if source.normalized.name_with_hash != config.normalized_version:
        raise ValueError(f"Source definition changed for {config.name}")
    StepRunner().run([source.normalized], max_concurrent=1)
    result = build_sft_store(
        [SftInput(config.name, prefix_join(source.normalized.output_path, "outputs/main"))],
        output_path=config.output_path,
        tokenizer=config.tokenizer,
        max_length=config.context_length,
        seed=config.seed,
        num_shards=config.shards,
        max_workers=config.workers,
    )
    logger.info("Prepared %s: %s", config.name, result.model_dump_json())
    return result


def build() -> dict[str, ArtifactStep[SftTokenStore]]:
    tokenizer_name = "grug_sft/tokenizer"
    tokenizer = ArtifactStep(
        name=tokenizer_name,
        version=resolve_version(tokenizer_name, None),
        artifact_type=Artifact,
        run=export_tokenizer,
        build_config=lambda ctx: TokenizerConfig(ctx.output_path, TOKENIZER_ID, TOKENIZER_REVISION, MARIN_CHAT_TEMPLATE),
    )
    handles = {}
    for name, source in all_sft_sources().items():
        artifact_name = f"grug_sft/tokenized/{name}"
        normalized_version = source.normalized.name_with_hash
        shards = max(1, math.ceil(source.rough_token_count_b * 1e9 / TARGET_TOKENS_PER_SHARD))

        def config(ctx: StepContext, name=name, normalized_version=normalized_version, shards=shards) -> SourceConfig:
            return SourceConfig(
                output_path=ctx.output_path,
                name=name,
                normalized_version=normalized_version,
                tokenizer=ctx.artifact_path(tokenizer),
                context_length=CONTEXT_LENGTH,
                seed=SHUFFLE_SEED,
                shards=shards,
                workers=ctx.runtime_arg("workers"),
            )

        handles[name] = ArtifactStep(
            name=artifact_name,
            version=resolve_version(artifact_name, None),
            artifact_type=SftTokenStore,
            run=prepare_source,
            build_config=config,
            deps=(tokenizer,),
            runtime_args={"workers": 32},
        )
    return handles


if __name__ == "__main__":
    experiment_main(build)()
