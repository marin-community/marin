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

import json
import logging

from levanter.data.text import LMMixtureDatasetConfig
from rigging.filesystem import open_url

from marin.execution.executor import output_path_of
from marin.processing.tokenize import lm_mixture_data_config
from marin.processing.tokenize.data_configs import TokenizerStep

logger = logging.getLogger(__name__)


def weights_from_tokenized_stats(tokenized_by_source: dict[str, TokenizerStep]) -> dict[str, float]:
    """Read each tokenize step's on-disk ``train/.stats.json`` for mixture weights.

    The ``tokenize`` fn writes ``<cache_path>/train/.stats.json`` with
    ``total_tokens`` + ``total_elements``. We use ``total_tokens`` as the
    raw mixture weight — reflects what actually landed on disk after
    dedup/filter rather than the registry's ``rough_token_count_b``.

    Every component must have a ``.stats.json`` — raises otherwise. Run
    the tokenize steps before calling.
    """
    weights: dict[str, float] = {}
    for name, step in tokenized_by_source.items():
        stats_path = f"{output_path_of(step)}/train/.stats.json"
        with open_url(stats_path) as f:
            stats = json.load(f)
        weights[name] = float(stats["total_tokens"])
    return weights


def build_testbed_mixture(
    tokenized_by_source: dict[str, TokenizerStep],
    *,
    weights: dict[str, float] | None = None,
) -> LMMixtureDatasetConfig:
    """Build the proportional mixture over a set of tokenized caches.

    Args:
        tokenized_by_source: Mapping from ``DatakitSource.name`` to its
            tokenize :class:`TokenizerStep` (``ExecutorStep[TokenizeConfig]``).
            The step must already be materialized so
            ``train/.stats.json`` exists under its output path.
        weights: Optional explicit mixture weights. Keys must match
            ``tokenized_by_source``. When ``None`` (default), weights are
            read from each component's on-disk ``train/.stats.json``.

    Returns:
        An ``LMMixtureDatasetConfig`` ready to hand to ``default_train`` or
        ``simulated_epoching_train``.

    Raises:
        ValueError: If ``tokenized_by_source`` is empty, or ``weights`` keys
            don't match ``tokenized_by_source`` keys exactly.
    """
    if not tokenized_by_source:
        raise ValueError("tokenized_by_source must be non-empty")

    if weights is None:
        weights = weights_from_tokenized_stats(tokenized_by_source)
    elif set(weights) != set(tokenized_by_source):
        raise ValueError(
            f"weights keys {sorted(weights)} must match tokenized_by_source keys {sorted(tokenized_by_source)}"
        )

    logger.info(
        "testbed mixture: %d components, total raw weight %.2eB",
        len(tokenized_by_source),
        sum(weights.values()) / 1e9,
    )

    return lm_mixture_data_config(
        components=tokenized_by_source,
        weights=weights,
    )
