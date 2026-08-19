# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harrier data mixture on the fuzzy-deduplicated store built 2026.08.18.

The phase weights are the evaluated 2026.08.17.1 fit, reused unchanged. Only the store and its
per-cell token counts differ: ``store_4d2e363d`` rebuilds ``store_81e7e39a`` with 16 sources exempt
from fuzzy dedup instead of one (``dna/functional-regions``). The two builds are otherwise the same,
with 40 clusters, 5 quality levels, 200 cells, and 384 tasks. Every cell keeps or increases its
token count, 23.01T to 23.11T overall, and maximum cell exposure stays at 2.09 epochs, so the
eight-epoch cap still holds.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from levanter.data.text.datasets import BlockShuffleConfig, DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from marin.execution.lazy import ArtifactStep, StepContext
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe.launch_datakit_moe_mix import (
    _phase_1_start_step,
    _simulated_epoching_budgets,
    _two_phase_data_config,
    _val_component,
)
from experiments.marin_tokenizer import marin_tokenizer

PRETRAIN_TOKENS = 15_000_000_000_000
COOLDOWN_TOKENS = 3_750_000_000_000
TOTAL_TOKENS = PRETRAIN_TOKENS + COOLDOWN_TOKENS
HARRIER_MIX_2026_08_18_TAG = "harrier-mix-2026.08.18"
# Simulated epoching stretches a short run's mixture as if it were a larger budget. Above this analytic
# training-FLOP budget the run is expensive enough that we want maximally-real data over a simulated
# larger run, so it trains on the raw mixture instead.
SIMULATED_EPOCHING_MAX_FLOPS = 1e23
_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=8, perm_type="feistel")

HARRIER_MIX_2026_08_18_STORE = ArtifactStep.adopt(
    "datakit/store/harrier-all-sources-k40-q5-fuzzy-dedup-exempt16",
    "2026.08.18",
    source="s3://marin-us-east-02a/marin/datakit/store_4d2e363d",
)


@dataclass(frozen=True)
class _HarrierMixSpec:
    tokenizer: str
    candidate_store_uri: str
    phase_budgets: tuple[int, int]
    available_tokens: tuple[tuple[str, int], ...]
    phase_weights: tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]


def _load_spec() -> _HarrierMixSpec:
    raw = json.loads(Path(__file__).with_name("harrier_mix_2026_08_18.json").read_text())
    return _HarrierMixSpec(
        tokenizer=raw["tokenizer"],
        candidate_store_uri=raw["candidate_store_uri"],
        phase_budgets=tuple(raw["phase_budgets"]),
        available_tokens=tuple(raw["available_tokens"].items()),
        phase_weights=(tuple(raw["phase0_weights"].items()), tuple(raw["phase1_weights"].items())),
    )


_SPEC = _load_spec()


def _validate_spec(spec: _HarrierMixSpec) -> None:
    available_tokens = dict(spec.available_tokens)
    phase_weights = tuple(dict(weights) for weights in spec.phase_weights)
    cells = set(available_tokens)
    if spec.phase_budgets != (PRETRAIN_TOKENS, COOLDOWN_TOKENS):
        raise ValueError("Harrier 2026.08.18 must use the 15T/3.75T phase budgets")
    if spec.tokenizer != marin_tokenizer:
        raise ValueError("Harrier 2026.08.18 must use the Marin tokenizer")
    if spec.candidate_store_uri != HARRIER_MIX_2026_08_18_STORE.adopt_source:
        raise ValueError("Harrier 2026.08.18 store does not match its adopted artifact")
    for phase, weights in enumerate(phase_weights):
        if set(weights) != cells or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"Harrier 2026.08.18 phase {phase} is not a dense simplex")
    cumulative_epochs = {
        cell: (
            sum(tokens * weights[cell] for tokens, weights in zip(spec.phase_budgets, phase_weights, strict=True))
            / available_tokens[cell]
        )
        for cell in cells
    }
    if max(cumulative_epochs.values()) > 8.0 + 1e-8:
        raise ValueError("Harrier 2026.08.18 exceeds the eight-epoch cap")


_validate_spec(_SPEC)


def harrier_mix_2026_08_18_data_config(
    *,
    ctx: StepContext,
    total_steps: int,
    batch_size: int,
    max_seq_len: int,
    experiment_flops: float,
    validation: Sequence[ArtifactStep[TokenizedCache]],
) -> LmDataConfig:
    """Build the evaluated two-phase mixture.

    Simulated epoching is on by default; it is dropped once ``experiment_flops`` (the run's analytic
    training-FLOP budget) exceeds ``SIMULATED_EPOCHING_MAX_FLOPS``, so an expensive run trains on the
    raw mixture rather than a simulated larger budget.
    """
    available_tokens = dict(_SPEC.available_tokens)
    phase_weights = tuple(dict(weights) for weights in _SPEC.phase_weights)
    components = {
        cell: DatasetComponent(
            source=None,
            cache_dir=prefix_join(
                ctx.artifact_path(HARRIER_MIX_2026_08_18_STORE),
                f"cluster={int(cell[1:3])}/quality={int(cell[4])}",
            ),
            format=TextLmDatasetFormat(),
            tags=[cell],
            flat_cache=True,
        )
        for cell in available_tokens
    }
    if ctx.is_fingerprint:
        val_components = {item.name: _val_component(ctx.artifact_path(item)) for item in validation}
    else:
        val_components = {item.name: ctx.resolved(item).as_component() for item in validation}
    collisions = components.keys() & val_components.keys()
    if collisions:
        raise ValueError(f"validation components collide with Harrier buckets: {sorted(collisions)}")
    target_budget, experiment_budget = _simulated_epoching_budgets(
        total_steps=total_steps,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        target_budget=TOTAL_TOKENS,
        enable_simulated_epoching=experiment_flops <= SIMULATED_EPOCHING_MAX_FLOPS,
    )

    return dataclasses.replace(
        _two_phase_data_config(
            tokenizer=marin_tokenizer,
            components=components,
            phase_weights=phase_weights,
            phase_1_start=_phase_1_start_step(total_steps, batch_size),
            val_components=val_components,
            target_budget=target_budget,
            experiment_budget=experiment_budget,
        ),
        shuffle=_SHUFFLE,
    )
