# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload a tier2 skewed-distribution synthetic dataset to HuggingFace Hub.

Reads the parquet shards staged at ``--source-path`` (a GCS prefix produced by
``scripts/datakit/generate_tier2_skewed.py``), copies them to a local scratch
directory, then uploads them to ``--repo-id`` on HuggingFace via
``HfApi.upload_folder``. Also writes a dataset-card README that propagates the
upstream FineWeb-Edu attribution and ODC-By license.

Token resolution order:
  1. ``--token`` CLI arg
  2. ``HF_TOKEN`` env var
  3. ``env.HF_TOKEN`` from the ``~/projects/marin/.marin.yaml`` (or
     ``--marin-yaml`` override) — the same file iris reads
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import tempfile
from pathlib import Path

import gcsfs
import yaml
from huggingface_hub import HfApi
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_PATH = "gs://marin-us-central2/tmp/ttl=3d/datakit-tier2-skew-v2/data"
DEFAULT_MARIN_YAML = "~/projects/marin/.marin.yaml"

DATASET_CARD_TEMPLATE = """\
---
license: odc-by
language:
  - en
size_categories:
  - 10B<n<100B
task_categories:
  - text-generation
pretty_name: Datakit Tier2 Skewed Synthetic
tags:
  - synthetic
  - testing
  - heavy-tail
  - datakit
  - marin
---

# Datakit Tier2 Skewed Synthetic

A synthetic, heavy-tailed-document dataset generated for stress-testing the
Marin datakit pipeline (normalize / minhash / fuzzy_dups / consolidate /
tokenize) against doc-length outliers up to 256 MB.

**This is a CI / pipeline-stress dataset, not a training corpus.** It exists
to exercise long-tail code paths that the FineWeb-Edu smoke ferry doesn't
cover. Don't use it for training without re-evaluating its distribution.

## Provenance

- **Source text**: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  `sample/10BT` split, revision `87f0914`. License: ODC-By 1.0.
- **Generator**: `scripts/datakit/generate_tier2_skewed.py` in
  [marin-community/marin](https://github.com/marin-community/marin).
- **Method**: per-doc target length sampled from a 3-mode mixture (normal
  log-normal, heavy Pareto, plus a deterministic injection of mega docs in
  the 128-256 MB band); content materialized by concatenating source text
  bytes from the FineWeb-Edu pool until the target length is reached.

## Schema

Parquet files with two columns:

- `id` — string, 16 hex chars, randomly assigned per doc
- `content` — string, UTF-8 text body

## License

Released under the [Open Data Commons Attribution License (ODC-By) 1.0](https://opendatacommons.org/licenses/by/1-0/),
inheriting from the upstream FineWeb-Edu license. When redistributing,
include attribution to:

> HuggingFaceFW/fineweb-edu (ODC-By 1.0)

## Generation parameters

{generation_params}
"""


def _resolve_token(args: argparse.Namespace) -> str:
    """Resolve the HF token from CLI / env / ~/.marin.yaml."""
    if args.token:
        return args.token
    env_tok = os.environ.get("HF_TOKEN")
    if env_tok:
        return env_tok
    path = Path(os.path.expanduser(args.marin_yaml))
    if not path.exists():
        raise RuntimeError(f"No HF token: --token not set, HF_TOKEN env empty, and {path} not found.")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    tok = (data.get("env") or {}).get("HF_TOKEN")
    if not tok:
        raise RuntimeError(f"No HF_TOKEN under 'env.HF_TOKEN' in {path}.")
    return tok


def _list_shards(source_path: str) -> tuple[gcsfs.GCSFileSystem, list[str]]:
    """Return ``(gcsfs, sorted_parquet_paths)`` under *source_path*."""
    fs = gcsfs.GCSFileSystem()
    src = source_path.removeprefix("gs://").rstrip("/")
    remote_paths = sorted(fs.glob(f"{src}/*.parquet"))
    if not remote_paths:
        raise FileNotFoundError(f"No parquet shards under {source_path}")
    return fs, remote_paths


def _build_card(source_path: str) -> bytes:
    """Build README.md dataset card content as UTF-8 bytes."""
    params = (
        f"- **Source**: `{source_path}`\n"
        "- **Distribution**: 70% log-normal (mean ~5 KB), 30% Pareto alpha=1.1 scale=2 KB, "
        "+100 mega docs uniformly in [128, 256] MB injected at random byte positions.\n"
        "- **Cap**: max doc 256 MB.\n"
    )
    return DATASET_CARD_TEMPLATE.format(generation_params=params).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. 'marin-community/datakit-tier2-skewed'.")
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--marin-yaml", default=DEFAULT_MARIN_YAML)
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true", help="Create the HF repo as private.")
    args = parser.parse_args()

    configure_logging()
    token = _resolve_token(args)
    api = HfApi(token=token)

    fs, remote_paths = _list_shards(args.source_path)
    logger.info("Found %d shards under %s", len(remote_paths), args.source_path)

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    # Upload one shard at a time: download to a single-file scratch path,
    # upload via ``HfApi.upload_file(path_or_fileobj=<str>)``, delete. Peak
    # disk = one shard (~470 MB); peak RAM = the gcsfs/copyfileobj buffers.
    # Cannot pass a gcsfs file-handle directly to ``upload_file`` — HF
    # requires ``io.BufferedIOBase`` specifically and gcsfs's file class is a
    # custom ``fsspec.AbstractBufferedFile``, not a BufferedIOBase subclass.
    scratch_dir = Path(tempfile.mkdtemp(prefix="tier2-upload-stage-"))
    try:
        for r in remote_paths:
            name = Path(r).name
            path_in_repo = f"data/{name}"
            local = scratch_dir / name
            with fs.open(r, "rb", block_size=8 * 1024 * 1024) as src_f, local.open("wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f, length=8 * 1024 * 1024)
            n = local.stat().st_size
            logger.info("uploading %s (%d bytes) -> %s", name, n, path_in_repo)
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Upload {name}",
            )
            local.unlink()
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    logger.info("uploading README.md")
    api.upload_file(
        path_or_fileobj=io.BytesIO(_build_card(args.source_path)),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    logger.info("Upload complete.")


if __name__ == "__main__":
    main()
