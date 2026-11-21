# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hero long-context mixture over virtual document-length buckets in the existing Harrier caches."""

from collections.abc import Mapping, Sequence

import click
from levanter.data.text.datasets import DatasetComponent, LengthSkewDatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.data.text.length_bucket import load_bucket_length_stats
from marin.execution.lazy import ArtifactStep, StepContext
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe.launch_datakit_moe_mix import _MIXTURE_BLOCK_SIZE
from experiments.grug.moe_hero_ep.harrier_length_stats import harrier_length_statistics
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import (
    _SPEC,
    HARRIER_MIX_2026_08_18_STORE,
    _shuffle_config,
    validation_components,
)
from experiments.marin_tokenizer import marin_tokenizer

LONG_CONTEXT_THRESHOLD = 65_536
LONG_CONTEXT_SKEW = 4
LONG_CONTEXT_TAG = f"harrier-long-context-skew-{LONG_CONTEXT_SKEW}"
REPORTED_TRAINING_TOKENS = 200_000_000_000
_COOLDOWN_WEIGHTS = {cell: weight for cell, weight in _SPEC.phase_weights[-1] if weight > 0}

LengthTokens = Mapping[str, tuple[int, int]]


def read_length_tokens(stats_path: str) -> dict[str, tuple[int, int]]:
    """Read short and long token totals from the length-statistics step output."""
    stats = load_bucket_length_stats(stats_path)
    return {
        cell: (
            row.total_tokens - row.tokens_above[LONG_CONTEXT_THRESHOLD],
            row.tokens_above[LONG_CONTEXT_THRESHOLD],
        )
        for cell, row in stats.buckets.items()
    }


def _bucket_component(store: str, cell: str, stats_path: str) -> LengthSkewDatasetComponent:
    return LengthSkewDatasetComponent(
        child=DatasetComponent(
            source=None,
            cache_dir=prefix_join(store, f"cluster={int(cell[1:3])}/quality={int(cell[4])}"),
            format=TextLmDatasetFormat(),
            tags=[cell],
            flat_cache=True,
        ),
        threshold=LONG_CONTEXT_THRESHOLD,
        long_weight=LONG_CONTEXT_SKEW,
        stats_path=stats_path,
        stats_key=cell,
    )


def harrier_long_context_data_config(
    *,
    ctx: StepContext,
    start_step: int,
    max_seq_len: int,
    validation: Sequence[ArtifactStep[TokenizedCache]],
    stats_path: str,
    store: str | None = None,
) -> LmDataConfig:
    """Train on cooldown weights over virtual length buckets from the existing Harrier store."""
    store = store or ctx.artifact_path(HARRIER_MIX_2026_08_18_STORE)
    components = {cell: _bucket_component(store, cell, stats_path) for cell in _COOLDOWN_WEIGHTS}
    val_components = validation_components(ctx, validation)
    if collisions := components.keys() & val_components.keys():
        raise ValueError(f"validation components collide with Harrier buckets: {sorted(collisions)}")
    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components={**components, **val_components},
        train_weights=[(0, {**_COOLDOWN_WEIGHTS, **{name: 0.0 for name in val_components}})],
        auto_build_caches=False,
        shuffle=_shuffle_config(max_seq_len),
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
        mixture_start_step=start_step,
    )


def long_document_share(length_tokens: LengthTokens, skew: int) -> float:
    """Return the fraction of training tokens from long documents, at the cooldown weights."""
    return sum(
        weight * skew * length_tokens[cell][1] / (length_tokens[cell][0] + skew * length_tokens[cell][1])
        for cell, weight in _COOLDOWN_WEIGHTS.items()
    )


def overall_long_partition_passes(length_tokens: LengthTokens, skew: int, training_tokens: int) -> float:
    """Return long tokens read per long token in the store."""
    long_tokens = sum(long for _, long in length_tokens.values())
    return long_document_share(length_tokens, skew) * training_tokens / long_tokens


@click.command()
def main() -> None:
    """Report the skew implied by the published Harrier length statistics."""
    configure_coreweave_s3()
    length_tokens = read_length_tokens(harrier_length_statistics().path())
    short_total = sum(short for short, _ in length_tokens.values())
    long_total = sum(long for _, long in length_tokens.values())
    click.echo(f"tokens: short {short_total:,}, long {long_total:,}")
    click.echo(f"long-document share: {long_document_share(length_tokens, LONG_CONTEXT_SKEW):.3%}")
    click.echo(
        f"{REPORTED_TRAINING_TOKENS / 1e9:.0f}B tokens: "
        f"{overall_long_partition_passes(length_tokens, LONG_CONTEXT_SKEW, REPORTED_TRAINING_TOKENS):.3f} "
        "long passes overall"
    )


if __name__ == "__main__":
    main()
