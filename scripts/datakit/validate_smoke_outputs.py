# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate datakit smoke ferry outputs.

Run after the iris job for the datakit smoke ferry has completed. Resolves
the output prefix via ``MARIN_PREFIX`` (falling back to
``marin_temp_bucket(ttl_days=1, prefix="datakit-smoke")`` — same default as
the ferry entrypoint) and asserts that each pipeline stage produced non-empty
output and that the tokenizer cache ledger is finished with rows.
"""

import logging
import os
import sys

from levanter.store.cache import CacheLedger
from rigging.filesystem import marin_temp_bucket, url_to_fs
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

STAGES = ("normalize", "dedup", "consolidate", "tokens")


def _assert_non_empty(path: str) -> None:
    fs, fs_path = url_to_fs(path)
    if not fs.exists(fs_path):
        raise SystemExit(f"Missing output directory: {path}")
    entries = fs.find(fs_path)
    if not entries:
        raise SystemExit(f"Output directory is empty: {path}")
    logger.info("OK %s (%d entries)", path, len(entries))


def main() -> None:
    configure_logging()
    prefix = os.environ.get("MARIN_PREFIX") or marin_temp_bucket(ttl_days=1)
    prefix = prefix.rstrip("/")
    run_id = os.environ["SMOKE_RUN_ID"]
    base = f"{prefix}/datakit-smoke/{run_id}"

    for stage in STAGES:
        _assert_non_empty(f"{base}/{stage}")

    train_dir = f"{base}/tokens/train"
    ledger = CacheLedger.load(train_dir)
    if not ledger.is_finished:
        raise SystemExit(f"Tokenizer cache ledger not finished: {train_dir}")
    if ledger.total_num_rows <= 0:
        raise SystemExit(f"Tokenizer cache ledger has 0 rows: {train_dir}")
    logger.info("Tokenizer cache OK: %d rows at %s", ledger.total_num_rows, train_dir)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
