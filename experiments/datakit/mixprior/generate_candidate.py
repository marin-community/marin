# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate one candidate from a materialized campaign."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from experiments.datakit.mixprior.artifacts import CANDIDATE_ARTIFACT
from experiments.datakit.mixprior.search import DEFAULT_POOL_SIZE, DEFAULT_SEED, generate_candidate

DEPENDENCY_LOCK = Path(__file__).parents[3] / "uv.lock"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    payload = generate_candidate(
        campaign_manifest=args.campaign_manifest,
        output_dir=args.output_dir,
        pool_size=args.pool_size,
        seed=args.seed,
        device=device,
        dependency_lock=DEPENDENCY_LOCK,
    )
    print(f"Wrote candidate {payload['candidate_id']} to {args.output_dir / CANDIDATE_ARTIFACT}")


if __name__ == "__main__":
    main()
