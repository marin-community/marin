# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The swarm design recipe: the run table of a new swarm.

A swarm trains one proxy run per row of ``swarm_table``: the baseline mixtures (the proportional mixture, which
calibrates the fit, UniMax and the stratified mixture), candidates drawn from a scrambled Sobol' sequence on the
simplex, the proportional mixture with each bucket removed in turn, and repeats of the proportional mixture. The
repeats are not fit rows; they anchor the floors. The paper's D-optimal selection among candidates is not part of this
recipe, so every candidate is a run; ``mariner design`` keeps each candidate at least ``MIN_DISTANCE`` in total
variation from the other candidates, the baselines and the removals. After training, one adds ``observation_id`` and the
``bpb__<task_id>`` outcomes, and ``mariner.data.swarm_from_runs`` reads the table.

The candidates port the research code's ``sobol_weight_configs`` for one phase, and UniMax its
``compute_unimax_weights``. The README's "Research code" section gives the research code's location: the research
branch at its release tag.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from mariner.data import (
    BASELINE,
    BASELINE_PREFIX,
    DELETION,
    DESIGN_KIND,
    FIT_INCLUSION,
    POOL_FRACTION_PREFIX,
    PROPORTIONAL,
    REPEAT,
    RUN_LABEL,
    SAMPLED,
    TRAINER_SEED,
    WEIGHT_PREFIX,
)

MIN_WEIGHT = 2e-4  # candidate weights below this become zero
MIN_DISTANCE = 0.001  # the least total variation distance of a candidate from another or from an existing mixture
SOBOL_CLIP = 1e-9  # keeps -log(u) finite and positive at the ends of the unit interval
POINTS_PER_CANDIDATE = 1_000  # Sobol' points tried per requested candidate before giving up
FIRST_BLOCK_EXPONENT = 3  # the first block holds 2**3 points
DEDUPLICATION_DECIMALS = 4

UNIMAX = "unimax"  # the recipe's other baselines, beside the calibrating PROPORTIONAL of mariner.data
STRATIFIED = "stratified"
SAMPLED_RUN_PREFIX = "run_"  # run labels run_00000, ..., deletion_<bucket> and proportional_repeat_00, ...
DELETION_RUN_PREFIX = "deletion_"
REPEAT_RUN_INFIX = "_repeat_"
DESIGN_TRAINER_SEED = 0  # the trainer seed of every run of a designed swarm


# ---------------------------------------------------------------------------------------------------------------
# Candidate mixtures
# ---------------------------------------------------------------------------------------------------------------


def sobol_mixtures(n_buckets: int, seed: int, min_weight: float) -> Iterator[np.ndarray]:
    """The mixtures of a scrambled Sobol' sequence in order, without skipping any.

    A point ``u`` becomes ``-log(u)`` normalized to sum to one, with weights below ``min_weight`` set to zero and the
    rest renormalized (the uniform mixture when every weight is below it).
    """
    # seed=<int> draws from default_rng(<int>) itself; rng=<int> would spawn a child of that generator first and
    # draw another sequence.
    sobol = qmc.Sobol(d=n_buckets, scramble=True, seed=seed)
    exponent = FIRST_BLOCK_EXPONENT
    drawn = 0
    while True:
        points = np.clip(sobol.random_base2(exponent), SOBOL_CLIP, 1.0 - SOBOL_CLIP)
        drawn += len(points)
        exponentials = -np.log(points)
        simplex = exponentials / exponentials.sum(axis=1, keepdims=True)
        kept = np.where(simplex >= min_weight, simplex, 0.0)
        total = kept.sum(axis=1, keepdims=True)
        mixtures = np.full_like(kept, 1.0 / n_buckets)
        np.divide(kept, total, out=mixtures, where=total > 0)
        yield from mixtures
        # random_base2 needs the points drawn so far to total a power of two, so each block doubles that total.
        exponent = int(math.log2(drawn))


def sobol_candidates(
    n: int,
    n_buckets: int,
    seed: int,
    min_weight: float = MIN_WEIGHT,
    existing: np.ndarray | None = None,
    min_distance: float = MIN_DISTANCE,
) -> np.ndarray:
    """``n`` candidate mixtures from a scrambled Sobol' sequence on the simplex, shape (n, n_buckets).

    The sequence's points are taken in order and each becomes a mixture (``sobol_mixtures``: ``-log(u)`` normalized,
    weights below ``min_weight`` zeroed). A mixture is skipped when its weights rounded to 4 decimals equal an
    accepted candidate's, or when its total variation distance to an accepted candidate or to a row of ``existing``
    (mixtures already in the swarm, shape (k, n_buckets)) is below ``min_distance``.

    Raises:
        ValueError: If ``n`` is negative, or fewer than ``n`` mixtures are accepted within 1,000 ``n`` points.
    """
    if n < 0:
        raise ValueError(f"the number of candidates must be nonnegative, got {n}")
    n_existing = 0 if existing is None else len(existing)
    pool = np.empty((n_existing + n, n_buckets))  # the existing mixtures, then the accepted candidates
    if existing is not None:
        pool[:n_existing] = existing
    filled = n_existing
    seen: set[tuple[float, ...]] = set()
    for mixture in itertools.islice(sobol_mixtures(n_buckets, seed, min_weight), POINTS_PER_CANDIDATE * n):
        key = tuple(np.round(mixture, DEDUPLICATION_DECIMALS))
        if key in seen:
            continue
        if filled and 0.5 * np.abs(pool[:filled] - mixture).sum(axis=1).min() < min_distance:
            continue
        seen.add(key)
        pool[filled] = mixture
        filled += 1
        if filled == len(pool):
            break
    if filled < len(pool):
        raise ValueError(
            f"only {filled - n_existing} of {n} candidates are distinct and at least {min_distance:g} apart"
            f" within {POINTS_PER_CANDIDATE * n} Sobol' points; ask for fewer or lower min_distance"
        )
    return pool[n_existing:]


# ---------------------------------------------------------------------------------------------------------------
# Baselines and removals
# ---------------------------------------------------------------------------------------------------------------


def deletion_label(bucket_id: str) -> str:
    """The run label of the mixture that removes ``bucket_id``, with the bucket id's ``/`` written ``_`` so that the
    label can name a folder: ``deletion_cc_art_and_design_high`` for ``cc/art_and_design_high``."""
    return f"{DELETION_RUN_PREFIX}{bucket_id.replace('/', '_')}"


def proportional(pool_tokens: np.ndarray) -> np.ndarray:
    """The mixture whose weights are proportional to the buckets' pool sizes."""
    return pool_tokens / pool_tokens.sum()


def unimax(pool_tokens: np.ndarray, budget: float, max_epochs: float) -> np.ndarray:
    """UniMax weights (Chung et al., 2023): ``budget`` tokens spread as evenly as possible over the buckets.

    Taking the buckets in ascending pool size (ties in bucket order), a bucket whose even share of the remaining
    budget exceeds ``max_epochs * pool`` is capped at that many tokens, and the buckets left split the remaining
    budget evenly.

    Raises:
        ValueError: If ``budget`` or ``max_epochs`` is not positive, or ``budget`` exceeds ``max_epochs`` times the
            pools' tokens, so that no bucket could stay within ``max_epochs``.
    """
    if not (budget > 0 and max_epochs > 0):
        raise ValueError(f"the budget and max_epochs must be positive, got {budget} and {max_epochs}")
    if budget > max_epochs * pool_tokens.sum():
        raise ValueError(
            f"a budget of {budget:g} tokens exceeds max_epochs ({max_epochs:g}) times the pools' {pool_tokens.sum():g}"
            " tokens"
        )
    allocation = np.zeros(len(pool_tokens))
    order = np.argsort(pool_tokens, kind="stable")
    remaining_budget = budget
    for visited, bucket in enumerate(order):
        share = remaining_budget / (len(order) - visited)
        cap = max_epochs * pool_tokens[bucket]
        if share <= cap:
            allocation[order[visited:]] = share
            break
        allocation[bucket] = cap
        remaining_budget -= cap
    return allocation / allocation.sum()


def stratified(n_buckets: int) -> np.ndarray:
    """The mixture that gives every bucket the same weight."""
    return np.full(n_buckets, 1.0 / n_buckets)


def removals(base: np.ndarray) -> np.ndarray:
    """One mixture per bucket, shape (buckets, buckets): row i is ``base`` with bucket i zeroed and renormalized.

    Raises:
        ValueError: If removing a bucket leaves no weight, as when ``base`` puts all of its weight on that bucket.
    """
    zeroed = base[None, :] * (1.0 - np.eye(len(base)))
    totals = zeroed.sum(axis=1, keepdims=True)
    empty = np.flatnonzero(totals[:, 0] <= 0.0)
    if len(empty):
        raise ValueError(f"the base mixture has no weight outside the bucket at position {empty[0]}")
    return zeroed / totals


# ---------------------------------------------------------------------------------------------------------------
# The run table
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunGroup:
    """Consecutive rows of a swarm table that share a design kind."""

    labels: list[str]
    kind: str
    fit_inclusion: bool
    weights: np.ndarray  # (runs, buckets)


def swarm_table(
    bucket_ids: Sequence[str],
    candidates: np.ndarray,
    baselines: Mapping[str, np.ndarray],
    removal_base: np.ndarray | None,
    repeats: int,
    data_seed_start: int,
) -> pd.DataFrame:
    """The run table of a new swarm, one row per training run, from mixtures over ``bucket_ids``.

    Rows come in this order:

    - one ``baseline_<name>`` row (``design_kind`` ``baseline``) for each of ``baselines``, which must contain
      ``proportional``;
    - ``run_00000``, ``run_00001``, ... (``sampled``) for each row of ``candidates``;
    - ``deletion_<bucket>`` (``deletion``) for each bucket, the bucket id with ``/`` written ``_``, the
      ``removal_base`` mixture with that bucket removed, unless ``removal_base`` is None;
    - ``repeats`` rows ``proportional_repeat_00``, ... (``repeat``) of the proportional baseline, which have
      ``fit_inclusion`` false; every other row has it true. The anchors need at least two, for a standard deviation.

    The columns are ``run_label``, ``design_kind``, ``fit_inclusion``, ``data_seed`` (``data_seed_start`` plus the
    row's position), ``trainer_seed`` (0), the weights ``w__<bucket_id>`` and the pool fractions ``pf__<bucket_id>``,
    which are all one. After training, one adds ``observation_id`` and the outcomes ``bpb__<task_id>``.

    Raises:
        ValueError: If ``baselines`` has no ``proportional`` mixture, or ``repeats`` is below two.
    """
    if PROPORTIONAL not in baselines:
        raise ValueError(f"the baselines must include '{PROPORTIONAL}', which calibrates the fit and is repeated")
    if repeats < 2:
        raise ValueError(f"the anchors need at least two proportional repeats for a standard deviation, got {repeats}")
    groups = [
        RunGroup(
            [f"{BASELINE_PREFIX}{name}" for name in baselines],
            BASELINE,
            fit_inclusion=True,
            weights=np.stack(list(baselines.values())),
        ),
        RunGroup(
            [f"{SAMPLED_RUN_PREFIX}{index:05d}" for index in range(len(candidates))],
            SAMPLED,
            fit_inclusion=True,
            weights=candidates,
        ),
    ]
    if removal_base is not None:
        groups.append(
            RunGroup(
                [deletion_label(bucket) for bucket in bucket_ids],
                DELETION,
                fit_inclusion=True,
                weights=removals(removal_base),
            )
        )
    groups.append(
        RunGroup(
            [f"{PROPORTIONAL}{REPEAT_RUN_INFIX}{index:02d}" for index in range(repeats)],
            REPEAT,
            fit_inclusion=False,
            weights=np.tile(baselines[PROPORTIONAL], (repeats, 1)),
        )
    )
    labels = [label for group in groups for label in group.labels]
    runs = pd.DataFrame(
        {
            RUN_LABEL: labels,
            DESIGN_KIND: [group.kind for group in groups for _ in group.labels],
            FIT_INCLUSION: [group.fit_inclusion for group in groups for _ in group.labels],
            "data_seed": data_seed_start + np.arange(len(labels)),
            TRAINER_SEED: DESIGN_TRAINER_SEED,
        }
    )
    weights = np.vstack([group.weights for group in groups])
    mixtures = pd.DataFrame(
        np.hstack([weights, np.ones_like(weights)]),
        columns=[f"{prefix}{bucket}" for prefix in (WEIGHT_PREFIX, POOL_FRACTION_PREFIX) for bucket in bucket_ids],
    )
    return pd.concat([runs, mixtures], axis=1)
