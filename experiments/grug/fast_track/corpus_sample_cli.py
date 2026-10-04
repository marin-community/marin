# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a fixed raw corpus pool for quality-classifier experiments."""

import math

import click
from levanter.tokenizers import tokenizer_content_hash
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.datakit.reference_pipeline import TokenizerSpec
from experiments.grug.fast_track.corpus_sample import (
    QUALITY_FRACTION,
    CorpusSampleSpec,
    RawCorpusPool,
    build_corpus_pool,
    corpus_sources,
)
from experiments.grug.fast_track.label_exclusion import read_label_exclusion
from experiments.grug.fast_track.launch import V16384_TOKENIZER, maximum_h100_ladder_tokens


@click.command()
@click.option("--source", "source_names", multiple=True, help="Registered source name. Omit to use every pinned source.")
@click.option("--max-training-tokens", type=click.IntRange(min=1), default=maximum_h100_ladder_tokens, show_default=True)
@click.option("--seed", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--label-exclusion-manifest", help="Fixed label groups to exclude from all classifier comparison pools.")
@build_options
def main(
    source_names: tuple[str, ...],
    max_training_tokens: int,
    seed: int,
    label_exclusion_manifest: str | None,
) -> ArtifactStep[RawCorpusPool]:
    sources = corpus_sources()
    if source_names:
        unknown = set(source_names) - {source.name for source in sources}
        if unknown:
            raise click.UsageError(f"unknown pinned corpus sources: {sorted(unknown)}")
        sources = tuple(source for source in sources if source.name in source_names)
    return build_corpus_pool(
        CorpusSampleSpec(
            sources=sources,
            tokenizer=TokenizerSpec(V16384_TOKENIZER, tokenizer_content_hash(V16384_TOKENIZER)),
            token_budget=math.ceil(max_training_tokens / QUALITY_FRACTION),
            seed=seed,
            label_exclusion=read_label_exclusion(label_exclusion_manifest) if label_exclusion_manifest else None,
        )
    )


if __name__ == "__main__":
    main()
