# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The hero's 262K extension: the Harrier cooldown weights on a store split by document length.

Each Harrier bucket concatenates its partition of short documents with ``LONG_CONTEXT_SKEW`` copies of its
partition of documents of 64K tokens or more, so the long documents gain mass inside the bucket while the
bucket keeps its cooldown weight. Sources differ from the Harrier store, so the Harrier context history cannot
replay here. The mixture instead starts at a chosen training step (``LmDataConfig.mixture_start_step``).

Run this module to read the store's token counts into ``harrier_long_context_mix.json`` and print the skewed
token share and the passes over the long partition.
"""

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheLedger
from marin.execution.lazy import ArtifactStep, StepContext
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe.launch_datakit_moe_mix import _MIXTURE_BLOCK_SIZE
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import _SPEC, _shuffle_config, validation_components
from experiments.marin_tokenizer import marin_tokenizer

# Provisional until the store lands. Refresh the token table after changing it.
LONG_CONTEXT_STORE_URI = "s3://marin-us-east-02a/marin/datakit/store_long_context_pending"
LONG_CONTEXT_SKEW = 4
LONG_CONTEXT_TAG = f"harrier-long-context-skew-{LONG_CONTEXT_SKEW}"
SHORT_PARTITION = "lte_64k"
LONG_PARTITION = "gt_64k"
# Training tokens the extension may read, which depends on the open learning-rate decision.
REPORTED_TRAINING_TOKENS = (50_000_000_000, 430_000_000_000, 490_000_000_000)

_TOKEN_TABLE_PATH = Path(__file__).with_name("harrier_long_context_mix.json")
_COOLDOWN_WEIGHTS = {cell: weight for cell, weight in _SPEC.phase_weights[-1] if weight > 0}

LengthTokens = Mapping[str, tuple[int, int]]
"""Short-partition and long-partition token counts by bucket."""


def _load_length_tokens() -> dict[str, tuple[int, int]]:
    return {cell: (short, long) for cell, (short, long) in json.loads(_TOKEN_TABLE_PATH.read_text()).items()}


def _partition_cache_dir(store: str, cell: str, partition: str) -> str:
    return prefix_join(store, f"cluster={int(cell[1:3])}/quality={int(cell[4])}/length={partition}")


def _bucket_component(
    store: str, cell: str, short_tokens: int, long_tokens: int, max_seq_len: int
) -> ConcatDatasetComponent:
    def child(partition: str) -> DatasetComponent:
        return DatasetComponent(
            source=None,
            cache_dir=_partition_cache_dir(store, cell, partition),
            format=TextLmDatasetFormat(),
            tags=[cell, partition],
            flat_cache=True,
        )

    # A partition shorter than one sequence serves nothing at this context.
    children = {f"{cell}-{SHORT_PARTITION}": child(SHORT_PARTITION)} if short_tokens >= max_seq_len else {}
    if long_tokens >= max_seq_len:
        children |= {f"{cell}-{LONG_PARTITION}-{copy}": child(LONG_PARTITION) for copy in range(LONG_CONTEXT_SKEW)}
    if not children:
        raise ValueError(f"Bucket {cell} has no {max_seq_len}-token sequence: {short_tokens=}, {long_tokens=}")
    return ConcatDatasetComponent(children=children, tags=[cell])


def _checked_length_tokens(length_tokens: LengthTokens) -> LengthTokens:
    if not length_tokens:
        raise ValueError(
            f"The long-context token table is empty. Set LONG_CONTEXT_STORE_URI, then run "
            f"`uv run python -m experiments.grug.moe_hero_ep.harrier_long_context_mix` to fill {_TOKEN_TABLE_PATH.name}"
        )
    if missing := _COOLDOWN_WEIGHTS.keys() - length_tokens.keys():
        raise ValueError(f"The long-context token table lacks weighted buckets: {sorted(missing)}")
    return length_tokens


def harrier_long_context_data_config(
    *,
    ctx: StepContext,
    start_step: int,
    max_seq_len: int,
    validation: Sequence[ArtifactStep[TokenizedCache]],
    store: str = LONG_CONTEXT_STORE_URI,
    length_tokens: LengthTokens | None = None,
) -> LmDataConfig:
    """Train on the cooldown weights over the skewed store, reading its first sequence at ``start_step``.

    The training sequence stream depends only on steps since ``start_step`` and the trainer seed, so arms on
    different rack counts read prefixes of one stream. The extension is short and expensive, thus it never uses
    simulated epoching.
    """
    tokens = _checked_length_tokens(_load_length_tokens() if length_tokens is None else length_tokens)
    components = {
        cell: _bucket_component(store, cell, *tokens[cell], max_seq_len=max_seq_len) for cell in _COOLDOWN_WEIGHTS
    }
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


def long_partition_passes(length_tokens: LengthTokens, skew: int, training_tokens: int) -> dict[str, float]:
    """Return how many times ``training_tokens`` of training reads each bucket's long partition."""
    return {
        cell: weight * training_tokens * skew / (length_tokens[cell][0] + skew * length_tokens[cell][1])
        for cell, weight in _COOLDOWN_WEIGHTS.items()
        if length_tokens[cell][1]
    }


def overall_long_partition_passes(length_tokens: LengthTokens, skew: int, training_tokens: int) -> float:
    """Return long tokens read per long token in the store."""
    long_tokens = sum(long for _, long in length_tokens.values())
    return long_document_share(length_tokens, skew) * training_tokens / long_tokens


def read_length_tokens(store: str, cells: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Read each bucket's partition token counts from the store's ledgers. A missing partition counts as empty."""

    def partition_tokens(cell_partition: tuple[str, str]) -> int:
        try:
            ledger = CacheLedger.load(_partition_cache_dir(store, *cell_partition))
        except FileNotFoundError:
            click.echo(f"no {cell_partition[1]} partition for {cell_partition[0]}; counting it as empty")
            return 0
        if not ledger.is_finished:
            raise ValueError(f"Unfinished partition: {cell_partition}")
        return ledger.field_counts["input_ids"]

    keys = [(cell, partition) for cell in cells for partition in (SHORT_PARTITION, LONG_PARTITION)]
    with ThreadPoolExecutor(max_workers=32) as pool:
        counts = dict(zip(keys, pool.map(partition_tokens, keys), strict=True))
    return {cell: (counts[cell, SHORT_PARTITION], counts[cell, LONG_PARTITION]) for cell in cells}


def _print_report(length_tokens: LengthTokens) -> None:
    short_total = sum(short for short, _ in length_tokens.values())
    long_total = sum(long for _, long in length_tokens.values())
    corpus_share = long_total / (short_total + long_total)
    click.echo(f"tokens: short {short_total:,}, long {long_total:,}, corpus long share {corpus_share:.3%}")
    click.echo(
        f"long-document share at the cooldown weights: 1x {long_document_share(length_tokens, 1):.3%}, "
        f"{LONG_CONTEXT_SKEW}x {long_document_share(length_tokens, LONG_CONTEXT_SKEW):.3%}"
    )
    for training_tokens in REPORTED_TRAINING_TOKENS:
        passes = long_partition_passes(length_tokens, LONG_CONTEXT_SKEW, training_tokens)
        overall = overall_long_partition_passes(length_tokens, LONG_CONTEXT_SKEW, training_tokens)
        most = sorted(passes.items(), key=lambda item: -item[1])[:5]
        click.echo(
            f"{training_tokens / 1e9:.0f}B tokens: {overall:.3f} long passes overall; most-read buckets "
            + ", ".join(f"{cell} {count:.2f}" for cell, count in most)
        )
        click.echo("per bucket: " + " ".join(f"{cell}={count:.3f}" for cell, count in sorted(passes.items())))


@click.command()
@click.option("--store", default=LONG_CONTEXT_STORE_URI, show_default=True, help="Store to read ledgers from.")
def main(store: str) -> None:
    """Write the store's token table next to this module and print its share and passes."""
    length_tokens = read_length_tokens(store, list(_COOLDOWN_WEIGHTS))
    rows = ",\n".join(f'"{cell}": [{short}, {long}]' for cell, (short, long) in sorted(length_tokens.items()))
    _TOKEN_TABLE_PATH.write_text("{\n" + rows + "\n}\n")
    _print_report(length_tokens)


if __name__ == "__main__":
    main()
