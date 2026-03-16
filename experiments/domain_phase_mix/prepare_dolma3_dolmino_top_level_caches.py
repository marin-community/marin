# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare merged top-level tokenized caches for the Dolma 3 + Dolmino swarm."""

from __future__ import annotations

import argparse
import logging
import sys

from marin.execution.executor import executor_main

from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DOMAIN_NAMES,
    build_top_level_domain_steps,
)

logger = logging.getLogger("ray")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Prepare merged top-level Dolma 3 + Dolmino caches")
    parser.add_argument(
        "--domains",
        type=str,
        default="all",
        help="Comma-separated top-level domain names to prepare, or 'all'.",
    )
    return parser.parse_known_args()


def _selected_domain_names(domains_arg: str) -> tuple[str, ...]:
    if domains_arg == "all":
        return DOMAIN_NAMES

    selected = tuple(name.strip() for name in domains_arg.split(",") if name.strip())
    if not selected:
        raise ValueError("Expected at least one domain name or 'all'.")

    unknown = sorted(set(selected) - set(DOMAIN_NAMES))
    if unknown:
        raise ValueError(f"Unknown top-level domains: {', '.join(unknown)}")

    return selected


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    selected_domains = _selected_domain_names(args.domains)
    step_by_domain = build_top_level_domain_steps()
    steps = [step_by_domain[domain_name] for domain_name in selected_domains]

    logger.info("Preparing %d merged top-level caches", len(steps))
    for domain_name in selected_domains:
        logger.info("  %s", domain_name)

    executor_main(
        steps=steps,
        description="Prepare merged Dolma 3 + Dolmino top-level caches",
    )


if __name__ == "__main__":
    main()
