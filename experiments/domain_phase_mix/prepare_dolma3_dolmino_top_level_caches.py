# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare top-level tokenized runtime caches for the Dolma 3 + Dolmino swarm."""

from __future__ import annotations

import argparse
import logging
import sys

from marin.execution.context import executor_context
from marin.execution.executor import ExecutorStep, executor_main

from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DEFAULT_RUNTIME_CACHE_REGION,
    DOMAIN_NAMES,
    build_top_level_domains,
)

logger = logging.getLogger(__name__)

STACK_EDU_DOMAIN = "dolma3_stack_edu"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Prepare top-level Dolma 3 + Dolmino runtime caches")
    parser.add_argument(
        "--domains",
        type=str,
        default="all",
        help="Comma-separated top-level domain names to prepare, or 'all'.",
    )
    parser.add_argument(
        "--allow-stack-edu",
        action="store_true",
        help="Allow public Stack-Edu hydration when the regional tokenized cache is absent.",
    )
    parser.add_argument("--runtime-cache-region", default=DEFAULT_RUNTIME_CACHE_REGION)
    return parser.parse_known_args()


def _selected_domain_names(domains_arg: str, *, allow_stack_edu: bool) -> tuple[str, ...]:
    if domains_arg == "all":
        selected = DOMAIN_NAMES
    else:
        selected = tuple(name.strip() for name in domains_arg.split(",") if name.strip())
        if not selected:
            raise ValueError("Expected at least one domain name or 'all'.")

    selected_set = set(selected)
    unknown = sorted(selected_set - set(DOMAIN_NAMES))
    if unknown:
        raise ValueError(f"Unknown top-level domains: {', '.join(unknown)}")
    if STACK_EDU_DOMAIN in selected_set and not allow_stack_edu:
        raise ValueError(
            f"Preparing {STACK_EDU_DOMAIN} may hydrate the public Software Heritage corpus. "
            "Pass --allow-stack-edu only after explicitly approving that transfer."
        )

    return selected


def _prep_steps(selected_domain_names: tuple[str, ...], runtime_cache_region: str) -> list[ExecutorStep]:
    selected = set(selected_domain_names)
    steps_by_name: dict[str, ExecutorStep] = {}
    for domain in build_top_level_domains(
        runtime_cache_region=runtime_cache_region,
        require_prebuilt_complete=False,
    ):
        if domain.name not in selected:
            continue
        for component in domain.components:
            runtime_component = component.get_step()
            if isinstance(runtime_component, ExecutorStep):
                steps_by_name[runtime_component.name] = runtime_component
    return list(steps_by_name.values())


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    selected_domains = _selected_domain_names(args.domains, allow_stack_edu=args.allow_stack_edu)
    with executor_context():
        steps = _prep_steps(selected_domains, args.runtime_cache_region)
    if not steps:
        logger.info("Selected domains already use existing runtime caches; no new prep steps are required.")
        return

    logger.info("Preparing %d top-level runtime cache roots", len(steps))
    for step in steps:
        logger.info("  %s", step.name)

    executor_main(
        steps=steps,
        description=f"Prepare Dolma 3 + Dolmino top-level runtime caches ({args.runtime_cache_region})",
    )


if __name__ == "__main__":
    main()
