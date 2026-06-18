# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris blog-metrics pipeline: TPU usage / users / tasks / FLOPS over time.

Three independently re-runnable steps, each reading/writing on-disk artifacts so
you can iterate on a later step without redoing an earlier one:

    fetch     mirror iris.worker/iris.task parquet (+ optional controller audit)
              to <data_dir>/raw
    extract   roll the raw parquet up into per-day CSVs in <data_dir>/daily
    charts    render SVG/PNG charts from the CSVs into <data_dir>/charts

Usage (run from lib/iris; matplotlib is only needed for `charts`):

    uv run python scripts/blog_metrics/pipeline.py fetch
    uv run python scripts/blog_metrics/pipeline.py extract
    uv run --with matplotlib python scripts/blog_metrics/pipeline.py charts
    uv run --with matplotlib python scripts/blog_metrics/pipeline.py all

The data source is the finelog `marin` deployment archived at
gs://marin-us-central2/finelog/marin. Structured rows only go back to the
finelog Rust migration (~2026-05-06), so the window is ~6 weeks, not months.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the sibling modules importable whether run as a script or with -m.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import extract
import fetch

logger = logging.getLogger("blog_metrics")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["fetch", "extract", "charts", "all"])
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"Root for raw/daily/charts artifacts (default: {config.DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--with-controller",
        action="store_true",
        help="In `fetch`, also extract /system/controller audit lines (slow ~33GB remote scan).",
    )
    parser.add_argument(
        "--no-wandb",
        dest="with_wandb",
        action="store_false",
        help="In `fetch`, skip the W&B long-history pull (needs `--with wandb` + WANDB_API_KEY).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="In `fetch`, refresh caches: rsync -d for stats parquet and re-pull the W&B history.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    paths = config.resolve_paths(args.data_dir)
    logger.info("data dir: %s", paths.data_dir)

    if args.step in ("fetch", "all"):
        fetch.run(paths, with_controller=args.with_controller, with_wandb=args.with_wandb, force=args.force)
    if args.step in ("extract", "all"):
        extract.run(paths)
    if args.step in ("charts", "all"):
        import charts  # noqa: PLC0415  # lazy: only `charts` needs matplotlib

        charts.run(paths)


if __name__ == "__main__":
    main()
