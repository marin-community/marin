# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
import logging
from dataclasses import dataclass

from marin.execution.step_spec import StepSpec
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)

_WEIGHTS_FILENAME = "weights.json"


@dataclass(frozen=True)
class TokenizedBucketWeightsConfig:
    """Inputs to ``compute_tokenized_bucket_weights``.

    ``tokenized_paths`` is keyed by bucket name and points at the resolved output
    path of each ``testbed_tokenize`` step.
    """

    tokenized_paths: dict[str, str]
    output_path: str


def compute_tokenized_bucket_weights(config: TokenizedBucketWeightsConfig) -> None:
    """Read ``train/.stats.json`` from each bucket and write aggregated weights."""
    weights: dict[str, float] = {}
    for name, out_path in config.tokenized_paths.items():
        stats_path = f"{out_path}/train/.stats.json"
        stats = json.loads(StoragePath(stats_path).read_text())
        weights[name] = float(stats["total_tokens"])

    out = f"{config.output_path}/{_WEIGHTS_FILENAME}"
    StoragePath(out).write_text(json.dumps(weights))
    logger.info("Wrote bucket weights for %d buckets to %s", len(weights), out)


def read_bucket_weights(weights_dir: str) -> dict[str, float]:
    """Read the weights.json produced by ``compute_tokenized_bucket_weights``."""
    return json.loads(StoragePath(f"{weights_dir}/{_WEIGHTS_FILENAME}").read_text())


def tokenized_bucket_weights_step(name: str, tokenized_buckets: dict[str, StepSpec]) -> StepSpec:
    """A step that reads each bucket's tokenize stats and emits weights.json.

    Pass the resulting step to ``run_testbed_config`` as ``weights_step``; depending on
    every tokenize bucket lets the runner resolve each bucket's output path at run time.
    """
    buckets = dict(tokenized_buckets)

    def fn(output_path: str) -> None:
        compute_tokenized_bucket_weights(
            TokenizedBucketWeightsConfig(
                tokenized_paths={bucket: step.output_path for bucket, step in buckets.items()},
                output_path=output_path,
            )
        )

    return StepSpec(
        name=f"data/datakit/weights/{name}",
        deps=list(buckets.values()),
        fn=fn,
    )
