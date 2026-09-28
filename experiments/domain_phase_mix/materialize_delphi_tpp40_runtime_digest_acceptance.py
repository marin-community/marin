# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Materialize zero-exclusion digest jobs for the repaired TPP40 caches."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.domain_phase_mix.audit_delphi_tpp40_europe_runtime_caches import (
    DIGEST_EXPECTED_COUNTS,
    CachePair,
    digest_comparison_filename,
)
from experiments.domain_phase_mix.delphi_tpp40_europe_runtime_caches import (
    EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS,
)
from experiments.domain_phase_mix.digest_delphi_tpp40_runtime_cache import ALGORITHM
from experiments.domain_phase_mix.materialize_delphi_tpp40_runtime_digest_diagnostics import (
    COMPONENTS as DIAGNOSTIC_COMPONENTS,
)
from experiments.domain_phase_mix.materialize_delphi_tpp40_runtime_digest_diagnostics import (
    DATE,
    O4MINI_COMPONENT,
    O4MINI_PATHS,
    REGIONS,
    DigestJob,
    digest_job_command_tokens,
)


@dataclass(frozen=True)
class AcceptanceComponent:
    component: str
    slug: str
    east5_path: str
    europe_path: str
    memory: str = "8GB"
    europe_output_namespace: str = "delphi_tpp40_multiregion_runtime_digests_v4"
    job_name_suffix: str = ""


EAST5_PATHS = {component.component: component.east5_path for component in DIAGNOSTIC_COMPONENTS}
ACCEPTANCE_COMPONENTS = (
    AcceptanceComponent(
        component="finemath_3plus",
        slug="finemath3plus",
        east5_path=EAST5_PATHS["finemath_3plus"],
        europe_path=EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS["finemath_3plus"],
    ),
    AcceptanceComponent(
        component="dolmino_stem_heavy_crawl",
        slug="stem-heavy-crawl",
        east5_path=EAST5_PATHS["dolmino_stem_heavy_crawl"],
        europe_path=EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS["dolmino_stem_heavy_crawl"],
        europe_output_namespace="delphi_tpp40_multiregion_runtime_digests_v4_stem_metadata_repair",
    ),
    AcceptanceComponent(
        component="synth_instruction/dolmino_flan",
        slug="dolmino-flan",
        east5_path=EAST5_PATHS["synth_instruction/dolmino_flan"],
        europe_path=EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS["synth_instruction/dolmino_flan"],
        memory="14GB",
        job_name_suffix="-retry1",
    ),
    AcceptanceComponent(
        component="synth_math/dolmino_math",
        slug="dolmino-math",
        east5_path=EAST5_PATHS["synth_math/dolmino_math"],
        europe_path=EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS["synth_math/dolmino_math"],
    ),
    AcceptanceComponent(
        component="synth_qa/wiki_to_rcqa",
        slug="wiki-to-rcqa",
        east5_path=EAST5_PATHS["synth_qa/wiki_to_rcqa"],
        europe_path=EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS["synth_qa/wiki_to_rcqa"],
    ),
    AcceptanceComponent(
        component="synth_thinking/code_meta_reasoning",
        slug="code-meta-reasoning",
        east5_path=EAST5_PATHS["synth_thinking/code_meta_reasoning"],
        europe_path=EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS["synth_thinking/code_meta_reasoning"],
    ),
    AcceptanceComponent(
        component="synth_thinking/math_meta_reasoning",
        slug="math-meta-reasoning",
        east5_path=EAST5_PATHS["synth_thinking/math_meta_reasoning"],
        europe_path=EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS["synth_thinking/math_meta_reasoning"],
    ),
    AcceptanceComponent(
        component="synth_thinking/program_verifiable",
        slug="program-verifiable",
        east5_path=EAST5_PATHS["synth_thinking/program_verifiable"],
        europe_path=EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS["synth_thinking/program_verifiable"],
    ),
    AcceptanceComponent(
        component=O4MINI_COMPONENT,
        slug="o4mini",
        east5_path=O4MINI_PATHS["east5"],
        europe_path=O4MINI_PATHS["europe"],
    ),
)


def validate_acceptance_paths(pairs: tuple[CachePair, ...] | None = None) -> None:
    """Validate frozen East5 inputs and Europe repair outputs for the digest panel."""
    diagnostic_east5_paths = {component.component: component.east5_path for component in DIAGNOSTIC_COMPONENTS}
    diagnostic_east5_paths[O4MINI_COMPONENT] = O4MINI_PATHS["east5"]
    for component in ACCEPTANCE_COMPONENTS:
        if component.east5_path != diagnostic_east5_paths[component.component]:
            raise ValueError(f"Acceptance East5 path differs from the reviewed digest input for {component.component!r}")
        if component.component == O4MINI_COMPONENT:
            if component.europe_path != O4MINI_PATHS["europe"]:
                raise ValueError(
                    f"Acceptance Europe path differs from the reviewed digest input for {component.component!r}"
                )
            continue
        if component.europe_path != EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS[component.component]:
            raise ValueError(
                f"Acceptance Europe path differs from the reviewed repair output for {component.component!r}"
            )

    if pairs is None:
        return
    resolved = {pair.component: pair for pair in pairs if pair.component in DIGEST_EXPECTED_COUNTS}
    if set(resolved) != set(DIGEST_EXPECTED_COUNTS):
        raise ValueError(f"Production runtime pairs differ from the acceptance matrix: {sorted(resolved)}")

    for component in ACCEPTANCE_COMPONENTS:
        pair = resolved[component.component]
        if pair.east5_path != component.east5_path:
            raise ValueError(f"Acceptance East5 path differs from production for {component.component!r}")
        if component.component == O4MINI_COMPONENT and pair.europe_path != component.europe_path:
            raise ValueError(f"Acceptance Europe path differs from production for {component.component!r}")


def materialize_jobs() -> tuple[DigestJob, ...]:
    jobs: list[DigestJob] = []
    for component in ACCEPTANCE_COMPONENTS:
        expected_rows, expected_tokens = DIGEST_EXPECTED_COUNTS[component.component]
        for region_key in ("east5", "europe"):
            region, zone, marin_prefix = REGIONS[region_key]
            cache_path = component.east5_path if region_key == "east5" else component.europe_path
            output_namespace = (
                "delphi_tpp40_multiregion_runtime_digests_v4"
                if region_key == "east5"
                else component.europe_output_namespace
            )
            output = (
                f"{marin_prefix}/experiments/domain_phase_mix/"
                f"{output_namespace}/"
                f"{digest_comparison_filename(component.component)}"
            )
            job_name = (
                f"dm-delphi-tpp40-runtime-digest-{component.slug}-{region_key}-v4-acceptance"
                f"{component.job_name_suffix}-{DATE}"
            )
            command = shlex.join(
                digest_job_command_tokens(
                    job_name=job_name,
                    region=region,
                    zone=zone,
                    marin_prefix=marin_prefix,
                    cache_path=cache_path,
                    output=output,
                    expected_rows=expected_rows,
                    expected_tokens=expected_tokens,
                    excluded_shards=(),
                    memory=component.memory,
                )
            )
            jobs.append(
                DigestJob(
                    component=component.component,
                    region_key=region_key,
                    job_name=job_name,
                    region=region,
                    zone=zone,
                    marin_prefix=marin_prefix,
                    cache_path=cache_path,
                    output=output,
                    expected_rows=expected_rows,
                    expected_tokens=expected_tokens,
                    excluded_shards=(),
                    memory=component.memory,
                    command=command,
                )
            )
    return tuple(jobs)


def manifest() -> dict[str, object]:
    return {
        "algorithm": ALGORITHM,
        "mode": "acceptance",
        "acceptance_requires_zero_exclusions": True,
        "jobs": [asdict(job) for job in materialize_jobs()],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validate_acceptance_paths()
    payload = json.dumps(manifest(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
