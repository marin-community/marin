# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable data mixtures and budget heuristic for midtraining experiments.

This module is the single source of truth for:

* The midtraining budget heuristic — :func:`midtrain_token_budget` applies
  the rule ``midtrain_tokens = pretrain_tokens * K`` with ``K = MIDTRAIN_BUDGET_FRACTION``
  uniformly across every model scale.
* The declarative mix spec — :class:`MidtrainMixSpec` composes a pretrain
  replay base (e.g. ``nemotron_mix``) with one or more midtrain components
  and auto-wires a held-out validation slice for each midtrain component.
* Layered safety assertions — :func:`validate_midtrain_spec`,
  :func:`assert_lm_data_config_safe`, :func:`log_partition_summary` catch
  config errors and silent regressions at config-/build-/launch-time.
* Backward-compatible names — the four registered mixes keep their stable
  string keys so existing callers (e.g. ``exp_delphi_math_10b_midtrain``)
  do not need to change.

Slow runtime checks (val/train disjointness by content hash, val partition
fingerprint) live in :mod:`experiments.midtrain_data_safety` to avoid
pulling jax / GCS reads into the import path of every consumer.

See ``.agents/logbooks/midtraining_delphi.md`` § "2026-05-01 21:00 UTC" for
the design rationale, the four-layer safety model, and the numbered
corner-case catalogue.
"""

import dataclasses
import logging
from dataclasses import dataclass

from haliax import Axis
from levanter.data.text import BlockShuffleConfig, LmDataConfig
from marin.execution.executor import ExecutorStep
from marin.processing.tokenize.data_configs import lm_mixture_data_config, step_to_lm_mixture_component

from experiments.midtraining_data_buckets import BUCKET_2
from experiments.pretraining_datasets.nemotron import nemotron_mix

logger = logging.getLogger(__name__)


# ─── 1. Heuristic constants (single source of truth) ────────────────────────

MIDTRAIN_BUDGET_FRACTION: float = 0.20
"""K — fraction of pretrain tokens spent on midtraining; same for every scale.

K = 0.20 corresponds to a 5/6-pretrain : 1/6-midtrain compute split, which
sits in Mantis-style cooldown territory (~10-20% of pretrain). This is the
single knob controlling the sweep budget; bumping it requires updating
this constant only. Override per launch via the ``MIDTRAIN_BUDGET_FRACTION``
environment variable consumed by ``exp_delphi_math_10b_midtrain.py``.
"""

DEFAULT_VAL_SEQUENCES_PER_MIDTRAIN_COMPONENT: int = 12_500
"""Default held-out val slice size per midtrain component.

12,500 sequences ≈ 51.2 M tokens at seq_len=4096 (~0.1% of a 52 B-token
component). Cheap enough for a fast eval pass every 200 train steps; large
enough that final-loss standard error is well below 0.001. Override
per-component via :attr:`MidtrainComponent.val_sequences`.
"""

HIGHQUALITY_NEMO_MATH_KEY: str = "nemotron_cc_math_v1/4plus"

# Backward-compatible registered mix names. Existing call sites import these
# strings directly; do NOT rename without updating call sites.
FULL_HIGHQUALITY_NEMO_MATH_NAME = "full_highquality_nemo_math"
PRETRAIN_70P_MATH_30P_HIGHQUALITY_NEMO_MATH_NAME = "70p_30m_highquality_nemo_math"
PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME = "67p_33m_highquality_nemo_math"
PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME = "33p_67m_highquality_nemo_math"


# ─── 2. Budget heuristic ────────────────────────────────────────────────────


def midtrain_token_budget(*, pretrain_tokens: int, fraction: float = MIDTRAIN_BUDGET_FRACTION) -> int:
    """Compute midtrain token budget under the global heuristic.

    Same rule for every Delphi scale; the only inputs are the base's own
    pretrain budget and the global ``fraction``. Truncates to an integer
    token count.

    Args:
        pretrain_tokens: tokens the base model was pretrained on.
        fraction: K — share of pretrain tokens spent on midtraining; in (0, 1].

    Raises:
        ValueError: if ``pretrain_tokens`` is non-positive or ``fraction`` is
            outside ``(0, 1]``.
    """
    if not isinstance(pretrain_tokens, int) or isinstance(pretrain_tokens, bool):
        raise TypeError(f"pretrain_tokens must be int, got {type(pretrain_tokens).__name__}")
    if pretrain_tokens <= 0:
        raise ValueError(f"pretrain_tokens must be positive, got {pretrain_tokens}")
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return int(pretrain_tokens * fraction)


# ─── 3. Declarative spec ────────────────────────────────────────────────────


@dataclass(frozen=True)
class MidtrainComponent:
    """A non-pretrain-replay dataset component participating in a midtrain mix.

    Each midtrain component automatically gets a held-out validation slice
    via Levanter's ``num_validation_sequences``. Set ``val_sequences=0`` to
    disable the carve-out for one specific component.
    """

    name: str
    """Registry key (e.g. ``"nemotron_cc_math_v1/4plus"``). Must not collide
    with any pretrain replay component name in the same mix."""

    step: ExecutorStep
    """Tokenized cache step (e.g. ``BUCKET_2["nemotron_cc_math_v1/4plus"]``)."""

    weight: float
    """Share of the total mix in (0, 1]. Sum of all midtrain weights plus
    :attr:`MidtrainMixSpec.pretrain_share` must equal 1.0."""

    val_sequences: int = DEFAULT_VAL_SEQUENCES_PER_MIDTRAIN_COMPONENT
    """Held-out val slice size; 0 disables the carve-out for this component."""


@dataclass(frozen=True)
class MidtrainMixSpec:
    """Declarative spec for a midtraining :class:`LmDataConfig`.

    The pretrain replay weights (from ``pretrain_base.train_weights``) are
    scaled to sum to ``pretrain_share``; midtrain components contribute the
    remaining ``1 - pretrain_share``. The build path
    (:func:`build_midtrain_lm_data_config`) preserves the pretrain base's
    other knobs (shuffle policy, mixture block size, etc.) while injecting
    the new components, normalized weights, and per-component val carve-outs.
    """

    name: str
    pretrain_base: LmDataConfig
    pretrain_share: float
    midtrain: tuple[MidtrainComponent, ...]

    def __post_init__(self):
        validate_midtrain_spec(self)


# ─── 4. Internal helpers ────────────────────────────────────────────────────


def _fixed_train_weights(config: LmDataConfig, *, name: str) -> dict[str, float]:
    weights = config.train_weights
    if not isinstance(weights, dict):
        raise ValueError(
            f"{name} must have fixed dict train_weights "
            f"(got {type(weights).__name__}); schedule-style weights are not "
            f"supported as a midtrain pretrain_base"
        )
    return weights


def _scale_weights(weights: dict[str, float], target_sum: float) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Cannot scale mixture weights with non-positive total")
    return {name: weight * target_sum / total for name, weight in weights.items()}


# ─── 5. Layer 1: config-time validation ─────────────────────────────────────


def validate_midtrain_spec(spec: MidtrainMixSpec) -> None:
    """Validate a :class:`MidtrainMixSpec` at construction time.

    Catches: out-of-range shares, empty midtrain tuple, duplicate midtrain
    names, name collisions with pretrain replay components, weights that
    don't sum to 1.0, and bad ``val_sequences`` values.

    Raises:
        ValueError on any failure.
    """
    if not (0 < spec.pretrain_share < 1):
        raise ValueError(f"pretrain_share must be in (0, 1), got {spec.pretrain_share}")
    if not spec.midtrain:
        raise ValueError("at least one midtrain component required")

    names = [c.name for c in spec.midtrain]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate midtrain component names: {names}")

    pretrain_names = set(spec.pretrain_base.components.keys())
    overlap = set(names) & pretrain_names
    if overlap:
        raise ValueError(
            f"midtrain components overlap with pretrain replay names: {sorted(overlap)}. "
            f"Rename the midtrain component or remove it from pretrain_base."
        )

    for c in spec.midtrain:
        if not (0 < c.weight <= 1):
            raise ValueError(f"weight for {c.name!r} must be in (0, 1], got {c.weight}")
        if c.val_sequences < 0:
            raise ValueError(f"val_sequences for {c.name!r} must be >= 0, got {c.val_sequences}")

    midtrain_share = sum(c.weight for c in spec.midtrain)
    total = spec.pretrain_share + midtrain_share
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"shares must sum to 1.0; got pretrain={spec.pretrain_share}, "
            f"midtrain_total={midtrain_share}, sum={total}"
        )


# ─── 6. Layer 2: built-config validation ────────────────────────────────────


def assert_lm_data_config_safe(cfg: LmDataConfig) -> None:
    """Verify safety invariants on a built :class:`LmDataConfig`.

    Catches dropped val carve-outs, weights that don't sum to 1.0, and
    shuffle-policy regressions (e.g. someone re-enabling full Feistel on
    the training stream — PR #5246 explicitly stopped that for I/O cost
    reasons).

    Raises:
        TypeError or ValueError on any failure.
    """
    weights = cfg.train_weights
    if not isinstance(weights, dict):
        raise TypeError(f"expected fixed dict train_weights, got {type(weights).__name__}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights sum to {total}, expected 1.0 (±1e-6)")

    if cfg.num_validation_sequences:
        for name, count in cfg.num_validation_sequences.items():
            if name not in cfg.components:
                raise ValueError(
                    f"val carve-out {name!r} is not a registered component " f"(components: {sorted(cfg.components)})"
                )
            if count <= 0:
                raise ValueError(f"val count for {name!r} must be positive, got {count}")

    if cfg.shuffle_before_trainval_split is not True:
        raise ValueError(
            "shuffle_before_trainval_split must be True so the val slice is a "
            "random sample, not the positional tail of the cache"
        )

    # Training shuffle: only block shuffle (default after #5246) or False.
    # Full Feistel on training reintroduces the I/O cost regression.
    if cfg.shuffle is True:
        raise ValueError(
            "Training shuffle=True (full Feistel) reintroduces the I/O cost "
            "regression that PR #5246 fixed. Use BlockShuffleConfig (default) or False."
        )
    if not (cfg.shuffle is False or isinstance(cfg.shuffle, BlockShuffleConfig)):
        raise ValueError(f"unsupported shuffle config type: {type(cfg.shuffle).__name__}")


# ─── 7. Builder ─────────────────────────────────────────────────────────────


def build_midtrain_lm_data_config(spec: MidtrainMixSpec) -> LmDataConfig:
    """Build an :class:`LmDataConfig` from a :class:`MidtrainMixSpec`.

    The pretrain replay weights are scaled to sum to ``spec.pretrain_share``;
    midtrain component weights are added on top. A held-out validation slice
    is carved out of each midtrain component with ``val_sequences > 0``.

    Inherits the pretrain base's training shuffle (Levanter's
    :class:`BlockShuffleConfig` default after PR #5246). Sets
    ``shuffle_before_trainval_split=True`` (full Feistel for the val carve-out)
    and ``auto_build_caches=False`` (a missing cache is a fast error rather
    than a silent rebuild that would desynchronize the val partition mid-sweep).

    The returned config is checked by :func:`assert_lm_data_config_safe` before
    being returned.
    """
    pretrain_weights = _scale_weights(
        _fixed_train_weights(spec.pretrain_base, name="pretrain_base"),
        spec.pretrain_share,
    )
    midtrain_weights = {c.name: c.weight for c in spec.midtrain}
    weights: dict[str, float] = {**pretrain_weights, **midtrain_weights}

    components = {
        **spec.pretrain_base.components,
        **{c.name: step_to_lm_mixture_component(c.step, include_raw_paths=True) for c in spec.midtrain},
    }

    existing_val = spec.pretrain_base.num_validation_sequences or {}
    midtrain_val = {c.name: c.val_sequences for c in spec.midtrain if c.val_sequences > 0}
    val_carveouts = {**existing_val, **midtrain_val}

    cfg = dataclasses.replace(
        spec.pretrain_base,
        components=components,
        train_weights=weights,
        num_validation_sequences=val_carveouts or None,
        shuffle_before_trainval_split=True,
        auto_build_caches=False,
    )
    assert_lm_data_config_safe(cfg)
    return cfg


# ─── 8. Layer 4: runtime instrumentation ────────────────────────────────────


def log_partition_summary(cfg: LmDataConfig, Pos: Axis) -> None:
    """Log the train/val partition shape for a built mix.

    Cheap; safe to call at startup of every training task. Records the val
    sequence counts (and approximate token counts) plus the train weights, so
    the partition is captured in stdout/W&B for every run and can be
    cross-referenced against the pinned fingerprint.

    Touches the cache to read ``len(validation_sets[name].as_sync_dataset())``;
    if ``auto_build_caches=False`` this only reads metadata, not data.
    """
    val_sets = cfg.validation_sets(Pos)
    for name, ds in val_sets.items():
        sync = ds.as_sync_dataset()
        n = len(sync)
        logger.info(
            "midtrain val[%s]: %d sequences (~%.1f M tokens at seq=%d)",
            name,
            n,
            n * Pos.size / 1e6,
            Pos.size,
        )
    weights = cfg.train_weights
    if isinstance(weights, dict):
        logger.info(
            "midtrain train weights (sum=%.6f): %s",
            sum(weights.values()),
            {k: round(v, 6) for k, v in weights.items()},
        )


# ─── 9. Registered specs and built mixes ────────────────────────────────────

_highquality_nemo_math_step = BUCKET_2[HIGHQUALITY_NEMO_MATH_KEY]


def _replay_math_spec(name: str, *, pretrain_share: float, math_share: float) -> MidtrainMixSpec:
    """Convenience: a Mantis-style mix with ``nemotron_mix`` replay + 4plus math."""
    return MidtrainMixSpec(
        name=name,
        pretrain_base=nemotron_mix,
        pretrain_share=pretrain_share,
        midtrain=(
            MidtrainComponent(
                name=HIGHQUALITY_NEMO_MATH_KEY,
                step=_highquality_nemo_math_step,
                weight=math_share,
            ),
        ),
    )


# 100% math mix has no pretrain replay base, so build it directly via
# lm_mixture_data_config (still gets the val carve-out and block shuffle defaults).
_full_math_lm_data_config = lm_mixture_data_config(
    components={HIGHQUALITY_NEMO_MATH_KEY: _highquality_nemo_math_step},
    weights={HIGHQUALITY_NEMO_MATH_KEY: 1.0},
    num_validation_sequences={HIGHQUALITY_NEMO_MATH_KEY: DEFAULT_VAL_SEQUENCES_PER_MIDTRAIN_COMPONENT},
    shuffle_before_trainval_split=True,
)
# auto_build_caches isn't a kwarg on lm_mixture_data_config; pin it via replace.
_full_math_lm_data_config = dataclasses.replace(_full_math_lm_data_config, auto_build_caches=False)
assert_lm_data_config_safe(_full_math_lm_data_config)


MIDTRAIN_SPECS: dict[str, MidtrainMixSpec] = {
    PRETRAIN_70P_MATH_30P_HIGHQUALITY_NEMO_MATH_NAME: _replay_math_spec(
        PRETRAIN_70P_MATH_30P_HIGHQUALITY_NEMO_MATH_NAME,
        pretrain_share=0.70,
        math_share=0.30,
    ),
    PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME: _replay_math_spec(
        PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME,
        pretrain_share=0.67,
        math_share=0.33,
    ),
    PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME: _replay_math_spec(
        PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME,
        pretrain_share=0.33,
        math_share=0.67,
    ),
}

MIDTRAINING_MIXES: dict[str, LmDataConfig] = {
    FULL_HIGHQUALITY_NEMO_MATH_NAME: _full_math_lm_data_config,
    **{name: build_midtrain_lm_data_config(spec) for name, spec in MIDTRAIN_SPECS.items()},
}

# Backward-compat module-level handles (legacy direct imports).
full_highquality_nemo_math = MIDTRAINING_MIXES[FULL_HIGHQUALITY_NEMO_MATH_NAME]
pretrain_70p_math_30p_highquality_nemo_math = MIDTRAINING_MIXES[PRETRAIN_70P_MATH_30P_HIGHQUALITY_NEMO_MATH_NAME]
pretrain_67p_math_33p_highquality_nemo_math = MIDTRAINING_MIXES[PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME]
pretrain_33p_math_67p_highquality_nemo_math = MIDTRAINING_MIXES[PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME]


def midtraining_mix_by_name(name: str) -> LmDataConfig:
    """Return a registered midtraining mixture by stable string name.

    Raises:
        ValueError if ``name`` is not a registered mix.
    """
    if name not in MIDTRAINING_MIXES:
        valid = ", ".join(sorted(MIDTRAINING_MIXES))
        raise ValueError(f"Unknown midtraining mix {name!r}; valid names: {valid}")
    return MIDTRAINING_MIXES[name]
