# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging

from rigging.filesystem import open_url

from marin.execution.executor import output_path_of
from marin.processing.tokenize.data_configs import TokenizerStep

logger = logging.getLogger(__name__)


def weights_from_tokenized_bucket_stats(tokenized_buckets: dict[str, TokenizerStep]) -> dict[str, float]:
    """Read each tokenize step's on-disk ``train/.stats.json`` for mixture weights"""
    weights: dict[str, float] = {}
    for name, step in tokenized_buckets.items():
        stats_path = f"{output_path_of(step)}/train/.stats.json"
        with open_url(stats_path) as f:
            stats = json.load(f)
        weights[name] = float(stats["total_tokens"])
    return weights
