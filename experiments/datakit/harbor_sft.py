# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize SFT-ready records from pinned Harbor trace datasets.

The trace ``agent`` provenance selects a harness adapter per source. Terminus-2
conversations pass through validation and Arrow normalization. Installed
OpenCode harness exports are rebuilt from their literal prompt/completion token
columns with the exact teacher tokenizer. A manifest harness value is an
optional assertion, not a routing override.

Run the historical Grug reproduction on Iris::

    iris --cluster=cw-rno2a job run --job-name harbor-sft-grug \
      --priority interactive --cpu 1 --memory 2GB \
      -- python -m experiments.datakit.harbor_sft \
      --manifest experiments/datakit/manifests/grug_67b_a2b_agentic_sft.json \
      --max-concurrent 8

Use ``--only exp_rpt_curriculum-hard`` for the compact 22-row golden smoke.
"""

import argparse
import logging

from marin.datakit.download.harbor_sft import (
    HarborSftManifest,
    harbor_sft_steps,
    load_harbor_sft_manifest,
)
from marin.execution.step_runner import StepRunner
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Pinned Harbor SFT source manifest.")
    parser.add_argument(
        "--only",
        help="Comma-separated source names to materialize (default: every source).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Maximum source chains StepRunner may execute concurrently.",
    )
    return parser.parse_args(argv)


def select_sources(manifest: HarborSftManifest, only: str | None):
    if not only:
        return manifest.sources
    requested = [name.strip() for name in only.split(",") if name.strip()]
    by_name = {source.name: source for source in manifest.sources}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"unknown Harbor SFT sources {unknown}; available: {sorted(by_name)}")
    return tuple(by_name[name] for name in requested)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = load_harbor_sft_manifest(args.manifest)
    sources = select_sources(manifest, args.only)
    terminals = [harbor_sft_steps(source)[-1] for source in sources]
    logger.info(
        "Materializing %d/%d source(s) from Harbor SFT manifest %s",
        len(sources),
        len(manifest.sources),
        manifest.name,
    )
    StepRunner().run(terminals, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    configure_logging()
    main()
