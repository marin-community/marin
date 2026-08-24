# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download a public transfer campaign and generate one candidate bundle."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from experiments.datakit.mixprior.artifacts import write_cycle_record
from experiments.datakit.mixprior.campaign import download_campaign
from experiments.datakit.mixprior.search import DEFAULT_POOL_SIZE, DEFAULT_SEED, generate_candidate

DEPENDENCY_LOCK = Path(__file__).parents[3] / "uv.lock"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-uri", required=True)
    parser.add_argument("--campaign-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    with TemporaryDirectory(prefix="mixprior-") as temporary:
        campaign_dir = Path(temporary) / "campaign"
        manifest_path = download_campaign(args.campaign_uri, args.campaign_sha256, campaign_dir)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        candidate = generate_candidate(
            campaign_manifest=manifest_path,
            output_dir=args.output_dir,
            pool_size=args.pool_size,
            seed=args.seed,
            device=device,
            dependency_lock=DEPENDENCY_LOCK,
        )
        shutil.copytree(campaign_dir, args.output_dir / "campaign")

    cycle_path = write_cycle_record(args.output_dir, args.campaign_uri, candidate)
    print(f"Wrote candidate {candidate['candidate_id']} to {cycle_path}")


if __name__ == "__main__":
    main()
