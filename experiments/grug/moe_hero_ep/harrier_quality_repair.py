# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harrier winner-quality-repair data mixture."""

from __future__ import annotations

import json
import math
from pathlib import Path

from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.storage_path import prefix_join

from experiments.marin_tokenizer import marin_tokenizer

MIXTURE_BLOCK_SIZE = 49_152
PRETRAIN_TOKENS = 15_000_000_000_000
COOLDOWN_TOKENS = 3_750_000_000_000
TOTAL_TOKENS = PRETRAIN_TOKENS + COOLDOWN_TOKENS

HARRIER_WINNER_QUALITY_REPAIR_STORE = ArtifactStep.adopt(
    "datakit/store/harrier-all-sources-k40-q5",
    "2026.08.16",
    source="s3://marin-us-east-02a/marin/datakit/store_0381a974",
)

_SPEC_PATH = Path(__file__).with_name("harrier_winner_quality_repair.json")
_SPEC = json.loads(_SPEC_PATH.read_text())
_AVAILABLE_TOKENS: dict[str, int] = _SPEC["available_tokens"]
_PHASE_WEIGHTS: tuple[dict[str, float], dict[str, float]] = (
    _SPEC["phase0_weights"],
    _SPEC["phase1_weights"],
)


def _validate_spec() -> None:
    cells = set(_AVAILABLE_TOKENS)
    if _SPEC["phase_budgets"] != [PRETRAIN_TOKENS, COOLDOWN_TOKENS]:
        raise ValueError("winner quality repair must use the 15T/3.75T phase budgets")
    if _SPEC["tokenizer"] != marin_tokenizer:
        raise ValueError("winner quality repair must use the Marin tokenizer")
    if _SPEC["candidate_store_uri"] != HARRIER_WINNER_QUALITY_REPAIR_STORE.adopt_source:
        raise ValueError("winner quality repair store does not match its adopted artifact")
    for phase, weights in enumerate(_PHASE_WEIGHTS):
        if set(weights) != cells or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"winner quality repair phase {phase} is not a dense simplex")
    cumulative_epochs = {
        cell: sum(tokens * weights[cell] for tokens, weights in zip(_SPEC["phase_budgets"], _PHASE_WEIGHTS))
        / _AVAILABLE_TOKENS[cell]
        for cell in cells
    }
    if max(cumulative_epochs.values()) > 8.0 + 1e-8:
        raise ValueError("winner quality repair exceeds the eight-epoch cap")


_validate_spec()


def _phase_1_start_step(total_steps: int, batch_size: int) -> int:
    requested = int(total_steps * PRETRAIN_TOKENS / TOTAL_TOKENS)
    step_multiple = MIXTURE_BLOCK_SIZE // math.gcd(MIXTURE_BLOCK_SIZE, batch_size)
    return max(step_multiple, requested // step_multiple * step_multiple)


def winner_quality_repair_data_config(
    *,
    store_path: str,
    total_steps: int,
    batch_size: int,
    max_seq_len: int,
    validation: dict[str, DatasetComponent | ConcatDatasetComponent],
) -> LmDataConfig:
    """Build the evaluated two-phase mixture for a simulated experiment budget."""
    components = {
        cell: DatasetComponent(
            source=None,
            cache_dir=prefix_join(
                store_path,
                f"cluster={int(cell[1:3])}/quality={int(cell[4])}",
            ),
            format=TextLmDatasetFormat(),
            tags=[cell],
            flat_cache=True,
        )
        for cell in _AVAILABLE_TOKENS
    }
    collisions = components.keys() & validation.keys()
    if collisions:
        raise ValueError(f"validation components collide with Harrier buckets: {sorted(collisions)}")
    components.update(validation)
    validation_weights = {name: 0.0 for name in validation}
    experiment_budget = total_steps * batch_size * max_seq_len
    if experiment_budget > TOTAL_TOKENS:
        raise ValueError(f"experiment budget {experiment_budget} exceeds target budget {TOTAL_TOKENS}")

    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=components,
        train_weights=[
            (0, {**_PHASE_WEIGHTS[0], **validation_weights}),
            (_phase_1_start_step(total_steps, batch_size), {**_PHASE_WEIGHTS[1], **validation_weights}),
        ],
        auto_build_caches=False,
        mixture_block_size=MIXTURE_BLOCK_SIZE,
        target_budget=TOTAL_TOKENS,
        experiment_budget=experiment_budget,
    )
