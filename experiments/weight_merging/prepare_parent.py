# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Acquire a pinned parent in campaign-owned regional storage."""

import argparse
import logging

from levanter.model_cache import cache_hf_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cache_hf_model(args.output, args.repo, revision=args.revision)


if __name__ == "__main__":
    main()
