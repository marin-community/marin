# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi scaling-ladder checkpoint roots."""

from __future__ import annotations

import argparse
import logging
import subprocess

from experiments.models import ModelConfig, download_model_step

logger = logging.getLogger(__name__)

DELPHI_CHECKPOINTS: dict[str, str] = {
    "3e18": "checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/hf",
    "9e18": "checkpoints/isoflop/isoflop-9e+18-d1152-L12-B16-adamh_scaling_v6/hf",
    "2e19": "checkpoints/isoflop/isoflop-2e+19-d1408-L15-B16-adamh_scaling_v6/hf",
    "3e19": "checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6/hf",
    "9e19": "checkpoints/isoflop/isoflop-9e+19-d1792-L18-B64-adamh_scaling_v6/hf",
    "2e20": "checkpoints/isoflop/isoflop-2e+20-d2048-L21-B64-adamh_scaling_v6/hf",
    "3e20": "checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/hf",
    "1e21": "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/hf",
    "1e22": "adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf",
    "1e23": "adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb/hf",
}

DELPHI_HF_REPOS: dict[str, str] = {
    "3e18": "marin-community/delphi-3e18-447Mparams-1.2Btokens",
    "9e18": "marin-community/delphi-9e18-550Mparams-2.9Btokens",
    "2e19": "marin-community/delphi-2e19-837Mparams-3.6Btokens",
    "3e19": "marin-community/delphi-3e19-998Mparams-5Btokens",
    "9e19": "marin-community/delphi-9e19-1.4Bparams-10.6Btokens",
    "2e20": "marin-community/delphi-2e20-1.9Bparams-14.8Btokens",
    "3e20": "marin-community/delphi-3e20-2.5Bparams-18.6Btokens",
    "1e21": "marin-community/delphi-1e21-3.4Bparams-46.3Btokens",
    "1e22": "marin-community/delphi-1e22-9.7Bparams-160Btokens",
    "1e23": "marin-community/delphi-1e23-25Bparams-628Btokens",
}

DELPHI_HF_DOWNLOADS = {
    slug: download_model_step(ModelConfig(hf_repo_id=repo, hf_revision="main")) for slug, repo in DELPHI_HF_REPOS.items()
}

# Region → marin GCS bucket. europe-west4's bucket is `marin-eu-west4`, not
# `marin-europe-west4`, so we can't just template `marin-{region}`.
_BUCKET_BY_REGION: dict[str, str] = {
    "us-central1": "marin-us-central1",
    "us-central2": "marin-us-central2",
    "us-east1": "marin-us-east1",
    "us-east5": "marin-us-east5",
    "us-west4": "marin-us-west4",
    "europe-west4": "marin-eu-west4",
}


def copy_checkpoints(
    src_region: str,
    dst_region: str,
    *,
    keys: list[str] | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Mirror Delphi HF checkpoints between marin GCS regions.

    Server-side `gcloud storage cp -r --no-clobber`; cross-region egress applies
    (~$0.01/GB US-US). ~1.1 TB total for all 10 keys; 1e23 alone is 745 GB.

    Args:
        src_region: source GCP region, e.g. "us-central2".
        dst_region: destination GCP region, e.g. "us-central1".
        keys: subset of `DELPHI_CHECKPOINTS` keys; None means all.
        dry_run: print commands without executing.

    Returns:
        Keys whose copy exited non-zero. Empty list on full success.
    """
    if src_region not in _BUCKET_BY_REGION:
        raise ValueError(f"unknown src_region {src_region!r}; known: {sorted(_BUCKET_BY_REGION)}")
    if dst_region not in _BUCKET_BY_REGION:
        raise ValueError(f"unknown dst_region {dst_region!r}; known: {sorted(_BUCKET_BY_REGION)}")
    src_bucket = _BUCKET_BY_REGION[src_region]
    dst_bucket = _BUCKET_BY_REGION[dst_region]

    requested = keys if keys is not None else list(DELPHI_CHECKPOINTS)
    unknown = set(requested) - set(DELPHI_CHECKPOINTS)
    if unknown:
        raise ValueError(f"unknown delphi keys: {sorted(unknown)}")

    failures: list[str] = []
    for key in requested:
        path = DELPHI_CHECKPOINTS[key]
        src = f"gs://{src_bucket}/{path}/"
        dst = f"gs://{dst_bucket}/{path}/"
        cmd = ["gcloud", "storage", "cp", "-r", "--no-clobber", "--continue-on-error", src, dst]
        logger.info("[%s] %s", key, " ".join(cmd))
        if dry_run:
            continue
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            logger.error("[%s] failed rc=%d", key, rc)
            failures.append(key)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror Delphi HF checkpoints between marin GCS regions.")
    parser.add_argument("--src", required=True, help="source region, e.g. us-central2")
    parser.add_argument("--dst", required=True, help="destination region, e.g. us-central1")
    parser.add_argument("--keys", nargs="+", help="subset of DELPHI_CHECKPOINTS keys (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures = copy_checkpoints(args.src, args.dst, keys=args.keys, dry_run=args.dry_run)
    if failures:
        raise RuntimeError(f"FAILED: {failures}")


if __name__ == "__main__":
    main()
