# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download a pinned Hugging Face campaign and generate a candidate bundle."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.datakit.mixprior.artifacts import write_bundle_manifest
from experiments.datakit.mixprior.generate_candidate import generate_candidate
from experiments.datakit.mixprior.huggingface import download_campaign
from experiments.datakit.mixprior.search import (
    DEFAULT_ACQUISITION_SEED,
    DEFAULT_POOL_SEEDS,
    DEFAULT_POOL_SIZE,
    noisy_expected_improvement,
)
from experiments.datakit.mixprior.surrogate import default_device

DEPENDENCY_LOCK = Path(__file__).parents[3] / "uv.lock"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-uri", required=True)
    parser.add_argument("--campaign-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-size-per-seed", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--pool-seed", type=int, action="append")
    parser.add_argument("--acquisition-seed", type=int, default=DEFAULT_ACQUISITION_SEED)
    args = parser.parse_args()

    with TemporaryDirectory(prefix="mixprior-") as temporary:
        campaign_dir = Path(temporary) / "campaign"
        manifest_path = download_campaign(args.campaign_uri, args.campaign_sha256, campaign_dir)
        candidate = generate_candidate(
            campaign_manifest=manifest_path,
            output_dir=args.output_dir,
            dependency_lock=DEPENDENCY_LOCK,
            acquisition=noisy_expected_improvement(args.acquisition_seed),
            pool_size_per_seed=args.pool_size_per_seed,
            pool_seeds=tuple(args.pool_seed or DEFAULT_POOL_SEEDS),
            device=default_device(),
        )
        shutil.copytree(campaign_dir, args.output_dir / "campaign")

    manifest_path = write_bundle_manifest(args.output_dir, args.campaign_uri)
    print(f"Wrote candidate {candidate['candidate_id']} to {manifest_path}")


if __name__ == "__main__":
    main()
