# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harrier followed by the selected September mixture on the 2026.08.18 data store.

Training starts with the evaluated 2026.08.17.1 weights, then switches to the mixture in
https://github.com/marin-community/marin/issues/9126. The store and its
per-cell token counts differ: ``store_4d2e363d`` rebuilds ``store_81e7e39a`` with 16 sources exempt
from fuzzy dedup instead of one (``dna/functional-regions``). The two builds are otherwise the same,
with 40 clusters, 5 quality levels, 200 cells, and 384 tasks. Every cell keeps or increases its
token count, 23.01T to 23.11T overall. The original Harrier weights reached at most 2.09 epochs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from marin.execution.lazy import ArtifactStep, StepContext
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe.launch_datakit_moe_mix import (
    _MIXTURE_BLOCK_SIZE,
    _phase_1_start_step,
    _simulated_epoching_budgets,
    _val_component,
)
from experiments.marin_tokenizer import marin_tokenizer

PRETRAIN_TOKENS = 15_000_000_000_000
COOLDOWN_TOKENS = 3_750_000_000_000
TOTAL_TOKENS = PRETRAIN_TOKENS + COOLDOWN_TOKENS
HARRIER_MIX_2026_08_18_TAG = "harrier-mix-2026.08.18-to-996f4891"
# Simulated epoching stretches a short run's mixture as if it were a larger budget. Above this analytic
# training-FLOP budget the run is expensive enough that we want maximally-real data over a simulated
# larger run, so it trains on the raw mixture instead.
SIMULATED_EPOCHING_MAX_FLOPS = 1e23

HARRIER_MIX_2026_08_18_STORE = ArtifactStep.adopt(
    "datakit/store/harrier-all-sources-k40-q5-fuzzy-dedup-exempt16",
    "2026.08.18",
    source="s3://marin-us-east-02a/marin/datakit/store_4d2e363d",
)


@dataclass(frozen=True)
class _HarrierMixSpec:
    tokenizer: str
    candidate_store_uri: str
    available_tokens: tuple[tuple[str, int], ...]
    phase_weights: tuple[tuple[tuple[str, float], ...], ...]


def _load_spec() -> _HarrierMixSpec:
    raw = json.loads(Path(__file__).with_name("harrier_mix_2026_08_18.json").read_text())
    return _HarrierMixSpec(
        tokenizer=raw["tokenizer"],
        candidate_store_uri=raw["candidate_store_uri"],
        available_tokens=tuple(raw["available_tokens"].items()),
        phase_weights=tuple(tuple(phase["weights"].items()) for phase in raw["phases"]),
    )


_SPEC = _load_spec()
# Same relative switch for every rung; the 390251-step hero switches after checkpoint 108000.
_MIXTURE_SWITCH_FRACTION = 108_000 / 390_251


def _validate_spec(spec: _HarrierMixSpec) -> None:
    available_tokens = dict(spec.available_tokens)
    phase_weights = tuple(dict(weights) for weights in spec.phase_weights)
    cells = set(available_tokens)
    if len(phase_weights) != 3:
        raise ValueError("Harrier must have initial, main, and cooldown phases")
    initial_tokens = TOTAL_TOKENS * _MIXTURE_SWITCH_FRACTION
    phase_budgets = (initial_tokens, PRETRAIN_TOKENS - initial_tokens, COOLDOWN_TOKENS)
    if spec.tokenizer != marin_tokenizer:
        raise ValueError("Harrier 2026.08.18 must use the Marin tokenizer")
    if spec.candidate_store_uri != HARRIER_MIX_2026_08_18_STORE.adopt_source:
        raise ValueError("Harrier 2026.08.18 store does not match its adopted artifact")
    for phase, weights in enumerate(phase_weights):
        if set(weights) != cells or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"Harrier 2026.08.18 phase {phase} is not a dense simplex")
    cumulative_epochs = {
        cell: (
            sum(tokens * weights[cell] for tokens, weights in zip(phase_budgets, phase_weights, strict=True))
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
    """Start on Harrier, switch to the selected September mixture at 108000/390251 of training.

    Both transitions align to mixture blocks; the selected cooldown still starts around 80%.

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

    step_multiple = _MIXTURE_BLOCK_SIZE // math.gcd(_MIXTURE_BLOCK_SIZE, batch_size)
    switch_step = math.ceil(total_steps * _MIXTURE_SWITCH_FRACTION / step_multiple) * step_multiple
    cooldown_step = _phase_1_start_step(total_steps, batch_size)
    val_zero_weights = {name: 0.0 for name in val_components}
    # A short diagnostic can round both transitions to the same block; cooldown wins there.
    stages = {
        step: {**weights, **val_zero_weights}
        for step, weights in zip((0, switch_step, cooldown_step), phase_weights, strict=True)
    }
    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components={**components, **val_components},
        train_weights=sorted(stages.items()),
        auto_build_caches=False,
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
        target_budget=target_budget,
        experiment_budget=experiment_budget,
    )
