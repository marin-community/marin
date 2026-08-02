# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run hierarchy labeling inside a directly federated GLM server gang."""

import argparse
import logging

from glm_hierarchical_labels import DEFAULT_BATCH_SIZE, DEFAULT_CONCURRENCY, VARIANTS, hierarchy_launch_config

from experiments.rollout_data.glm52_vllm import serve_glm52

RAY_PORT = 6_379
HTTP_PORT = 8_000


def main() -> None:
    """Parse arguments and serve GLM with the hierarchy client on rank zero."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args()
    if args.batch_size < 1 or args.concurrency < 1:
        parser.error("--batch-size and --concurrency must be positive")
    logging.basicConfig(level=logging.INFO)
    launch = hierarchy_launch_config(
        args.run_id,
        [VARIANTS[name] for name in args.variants],
        args.batch_size,
        args.concurrency,
    )
    serve_glm52(launch, RAY_PORT, HTTP_PORT)


if __name__ == "__main__":
    main()
