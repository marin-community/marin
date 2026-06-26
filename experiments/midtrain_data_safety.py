# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Slow runtime safety check for midtraining val carve-outs.

Hosts :func:`assert_val_train_disjoint`, the Layer 3 disjointness assertion
that fires once per sweep launch (under ``__main__``) before any training
task is dispatched. Reads from GCS, so this lives in its own module to keep
``experiments.midtraining_mixes`` import-fast for tests and dry-runs.

See :mod:`experiments.midtraining_mixes` for layers 1, 2, and 4 and
``.agents/logbooks/midtraining_delphi.md`` § "2026-05-01 21:00 UTC" for
the full design.
"""

import logging
from typing import Any

import jax
from haliax import Axis
from levanter.data.text import LmDataConfig

logger = logging.getLogger(__name__)

DEFAULT_DISJOINT_SAMPLE_SIZE: int = 5_000
DEFAULT_MIN_TRAIN_TO_VAL_RATIO: int = 100


def assert_val_train_disjoint(
    cfg: LmDataConfig,
    Pos: Axis,
    *,
    sample_train: int = DEFAULT_DISJOINT_SAMPLE_SIZE,
    min_train_to_val_ratio: int = DEFAULT_MIN_TRAIN_TO_VAL_RATIO,
) -> None:
    """Verify val is disjoint from the (block-shuffled) train stream.

    Hashes a stride-sample of train sequences and ALL val sequences (val is
    small, ~12.5k) for each component, then asserts empty intersection.

    Catches:
        * Levanter refactors that break the slice/shuffle composition
        * cache rebuilds that desynchronize ``train_sets`` and ``validation_sets``
        * silent overrides of ``shuffle_before_trainval_split``

    Slow: reads ``val_len + sample_train`` sequences per component from the
    cache. Run at sweep launch, not per training task.

    Args:
        cfg: the built :class:`LmDataConfig` to verify.
        Pos: position axis (e.g. ``Axis("position", 4096)``).
        sample_train: number of stride-sampled train sequences to hash per
            component. Larger values give stronger guarantees at proportional
            cost.
        min_train_to_val_ratio: train must be at least this many times larger
            than val per component; otherwise the carve-out is misconfigured.

    Raises:
        AssertionError: on empty val, ratio violation, or hash overlap.
    """
    if not cfg.num_validation_sequences:
        logger.info("no num_validation_sequences set; skipping disjointness check")
        return

    train_sets = cfg.train_sets(Pos, key=jax.random.PRNGKey(42))
    val_sets = cfg.validation_sets(Pos)

    for name in cfg.num_validation_sequences:
        if name not in train_sets:
            raise AssertionError(
                f"component {name!r} has a val carve-out but is not in train_sets " f"(maybe its train weight is 0?)"
            )
        if name not in val_sets:
            raise AssertionError(f"component {name!r} not in validation_sets")

        train_ds = train_sets[name].as_sync_dataset()
        val_ds = val_sets[name].as_sync_dataset()

        train_len, val_len = len(train_ds), len(val_ds)
        if val_len == 0:
            raise AssertionError(f"empty val set for {name!r}")
        if train_len < val_len * min_train_to_val_ratio:
            raise AssertionError(
                f"train < {min_train_to_val_ratio}x val for {name!r} "
                f"({train_len} vs {val_len}); likely misconfigured carve-out"
            )

        stride = max(1, train_len // sample_train)
        train_h = {_hash_seq(train_ds[i]) for i in range(0, train_len, stride)}
        val_h = {_hash_seq(val_ds[i]) for i in range(val_len)}

        overlap = train_h & val_h
        if overlap:
            raise AssertionError(
                f"VAL LEAKED INTO TRAIN for component {name!r}: {len(overlap)} "
                f"hash matches (val {len(val_h)} ∩ train {len(train_h)}). "
                f"Stop the sweep and investigate the slice/shuffle composition."
            )
        logger.info(
            "val/train disjoint for %s: |val|=%d, |train_sample|=%d, |∩|=0",
            name,
            len(val_h),
            len(train_h),
        )


def _hash_seq(example: Any) -> int:
    """Stable hash of a tokenized sequence example.

    Tries common attribute names exposed by Levanter's ``LmExample`` /
    ``GrugLmExample`` so this works across Levanter versions without
    pinning to one example type.
    """
    for attr in ("tokens", "input_ids"):
        if hasattr(example, attr):
            arr = getattr(example, attr)
            data = arr.tobytes() if hasattr(arr, "tobytes") else bytes(arr)
            return hash(data)
    raise AttributeError(
        f"don't know how to extract token bytes from {type(example).__name__}; " f"expected one of {{tokens, input_ids}}"
    )
