# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path
from marin.processing.tokenize.data_configs import TokenizerStep
from rigging.filesystem import open_url

logger = logging.getLogger(__name__)

_WEIGHTS_FILENAME = "weights.json"


@dataclass(frozen=True)
class TokenizedBucketWeightsConfig:
    """Inputs to ``compute_tokenized_bucket_weights``.

    ``tokenized_paths`` is keyed by bucket name and points at the resolved
    output path of each ``testbed_tokenize`` step. The values are
    ``InputName`` references at construction time and concrete strings at
    runtime, after the executor resolves dependencies.
    """

    tokenized_paths: dict[str, str | InputName]
    output_path: str


def compute_tokenized_bucket_weights(config: TokenizedBucketWeightsConfig) -> None:
    """Read ``train/.stats.json`` from each bucket and write aggregated weights."""
    weights: dict[str, float] = {}
    for name, out_path in config.tokenized_paths.items():
        stats_path = f"{out_path}/train/.stats.json"
        with open_url(stats_path) as f:
            stats = json.load(f)
        weights[name] = float(stats["total_tokens"])

    out = f"{config.output_path}/{_WEIGHTS_FILENAME}"
    with open_url(out, "w") as f:
        json.dump(weights, f)
    logger.info("Wrote bucket weights for %d buckets to %s", len(weights), out)


def read_bucket_weights(weights_dir: str) -> dict[str, float]:
    """Read the weights.json produced by ``compute_tokenized_bucket_weights``."""
    with open_url(f"{weights_dir}/{_WEIGHTS_FILENAME}") as f:
        return json.load(f)


def tokenized_bucket_weights_step(name: str, tokenized_buckets: dict[str, TokenizerStep]) -> ExecutorStep:
    """ExecutorStep that reads each bucket's tokenize stats and emits weights.json.

    Pass the resulting step to ``run_testbed_config`` as ``weights_step``; the
    executor resolves the dependency on each tokenize bucket automatically.
    """
    return ExecutorStep(
        name=f"data/datakit/weights/{name}",
        fn=compute_tokenized_bucket_weights,
        config=TokenizedBucketWeightsConfig(
            tokenized_paths={b: output_path_of(t) for b, t in tokenized_buckets.items()},
            output_path=this_output_path(),
        ),
    )
