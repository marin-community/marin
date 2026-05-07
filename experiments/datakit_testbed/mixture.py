# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging

from rigging.filesystem import open_url

logger = logging.getLogger(__name__)


def weights_from_tokenized_bucket_stats(tokenized_output_paths: dict[str, str]) -> dict[str, float]:
    """Read each tokenize output's on-disk ``train/.stats.json`` for mixture weights.

    Takes **resolved** output paths (e.g. from ``Executor.output_paths``),
    not ``TokenizerStep`` objects, because a bare ``ExecutorStep`` doesn't
    know its hashed output path outside an active executor context.
    """
    weights: dict[str, float] = {}
    for name, out_path in tokenized_output_paths.items():
        stats_path = f"{out_path}/train/.stats.json"
        with open_url(stats_path) as f:
            stats = json.load(f)
        weights[name] = float(stats["total_tokens"])
    return weights
