# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proportional-mixing builder for Datakit Testbed training runs.

Maps each testbed ``DatakitSource`` to a mixture weight proportional to its
``rough_token_count_b`` and wraps the resulting
``dict[source_name -> TokenizerStep]`` in an ``LmDataConfig`` via
``lm_mixture_data_config``.

Weights are raw; ``MixtureDataset`` normalizes them at sampling time (see
``levanter/data/text/datasets.py:701``), so there's no need to pre-normalize.

For simulated epoching, training callers set ``target_budget`` and
``experiment_budget`` on the returned config via ``dataclasses.replace``
(see ``experiments/defaults.py:321``). That slicing preserves the per-source
proportions over the shortened horizon.
"""

from __future__ import annotations

import logging

from levanter.data.text import LMMixtureDatasetConfig
from marin.processing.tokenize import lm_mixture_data_config
from marin.processing.tokenize.data_configs import TokenizerStep

from marin.datakit.sources import DatakitSource, all_sources

logger = logging.getLogger(__name__)


def weights_from_rough_counts(sources: list[DatakitSource]) -> dict[str, float]:
    """Use each source's ``rough_token_count_b`` as its mixture weight.

    Measured counts from the tokenize step's ``stats.json`` can replace
    this later.
    """
    return {src.name: src.rough_token_count_b for src in sources}


def build_testbed_mixture(
    tokenized_by_source: dict[str, TokenizerStep],
    *,
    weights: dict[str, float] | None = None,
    sources: list[DatakitSource] | None = None,
) -> LMMixtureDatasetConfig:
    """Build the proportional mixture over a set of tokenized caches.

    Args:
        tokenized_by_source: Mapping from ``DatakitSource.name`` to its
            ``tokenize`` step. Typically ``TestbedDAG.tokenized_by_source``.
        weights: Optional explicit mixture weights. Keys must match
            ``tokenized_by_source``. Raw values — not pre-normalized.
        sources: Optional source list to pull ``rough_token_count_b`` from
            when ``weights`` is not provided. Defaults to the full 102-entry
            set from :func:`all_sources`.

    Returns:
        An ``LMMixtureDatasetConfig`` ready to hand to ``default_train`` or
        ``simulated_epoching_train`` (after setting ``target_budget`` and
        ``experiment_budget``).

    Raises:
        ValueError: If ``tokenized_by_source`` is empty, or ``weights`` keys
            don't match ``tokenized_by_source`` keys exactly.
    """
    if not tokenized_by_source:
        raise ValueError("tokenized_by_source must be non-empty")

    if weights is None:
        resolved_sources = sources if sources is not None else list(all_sources().values())
        known = {s.name: s for s in resolved_sources}
        missing = set(tokenized_by_source) - set(known)
        if missing:
            raise ValueError(
                f"No DatakitSource metadata for tokenized components: {sorted(missing)}. "
                "Pass weights=... explicitly or extend sources=..."
            )
        selected = [known[name] for name in tokenized_by_source]
        weights = weights_from_rough_counts(selected)
    else:
        if set(weights) != set(tokenized_by_source):
            raise ValueError(
                f"weights keys {sorted(weights)} must match tokenized_by_source keys " f"{sorted(tokenized_by_source)}"
            )

    logger.info(
        "testbed mixture: %d components, total raw weight %.1fB",
        len(tokenized_by_source),
        sum(weights.values()),
    )

    return lm_mixture_data_config(
        components=tokenized_by_source,
        weights=weights,
    )
