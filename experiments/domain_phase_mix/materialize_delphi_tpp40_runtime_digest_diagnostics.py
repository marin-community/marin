# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Materialize reviewed region-local runtime-digest diagnostic jobs."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.domain_phase_mix.audit_delphi_tpp40_europe_runtime_caches import (
    DIGEST_EXPECTED_COUNTS,
    CachePair,
    cache_pairs,
    digest_comparison_filename,
)
from experiments.domain_phase_mix.digest_delphi_tpp40_runtime_cache import ALGORITHM

DATE = "20260830"
TIMEOUT_SECONDS = 43_200
WORKING_DIR_EXCLUDES = (
    ".agents/",
    ".github/",
    "docs/",
    "scripts/",
    "experiments/domain_phase_mix/exploratory/",
    "experiments/domain_phase_mix/manifests/",
    "checkpoints/",
    "tests/",
    "infra/grafana/",
    ".experiments/",
    ".experiments.zip",
)


@dataclass(frozen=True)
class ComponentDiagnostic:
    component: str
    slug: str
    east5_path: str
    europe_path: str
    europe_rows: int
    europe_tokens: int
    europe_excluded_shards: tuple[str, ...] = ()
    memory: str = "8GB"


@dataclass(frozen=True)
class DigestJob:
    component: str
    region_key: str
    job_name: str
    region: str
    zone: str
    marin_prefix: str
    cache_path: str
    output: str
    expected_rows: int
    expected_tokens: int
    excluded_shards: tuple[str, ...]
    memory: str
    command: str


COMPONENTS = (
    ComponentDiagnostic(
        component="finemath_3plus",
        slug="finemath3plus",
        east5_path="gs://marin-us-east5/tokenized/finemath_3_plus-a26b0f",
        europe_path="gs://marin-eu-west4/tokenized/finemath_3_plus-a26b0f",
        europe_rows=21_405_610,
        europe_tokens=34_001_855_255,
    ),
    ComponentDiagnostic(
        component="dolmino_stem_heavy_crawl",
        slug="stem-heavy-crawl",
        east5_path=("gs://marin-us-east5/tokenized/merged/dolma3_dolmino_top_level/" "dolmino_stem_heavy_crawl-e1ec3b"),
        europe_path=("gs://marin-eu-west4/tokenized/merged/dolma3_dolmino_top_level/" "dolmino_stem_heavy_crawl-e1ec3b"),
        europe_rows=5_160_830,
        europe_tokens=5_213_791_201,
    ),
    ComponentDiagnostic(
        component="synth_instruction/dolmino_flan",
        slug="dolmino-flan",
        east5_path="gs://marin-us-east5/tokenized/dolma3_dolmino_pool/synth_instruction_dolmino_flan-183f12",
        europe_path="gs://marin-eu-west4/tokenized/dolma3_dolmino_pool/synth_instruction_dolmino_flan-183f12",
        europe_rows=56_099_440,
        europe_tokens=16_442_407_197,
        europe_excluded_shards=("part-00064-of-00209", "part-00122-of-00209", "part-00163-of-00209"),
        memory="16GB",
    ),
    ComponentDiagnostic(
        component="synth_math/dolmino_math",
        slug="dolmino-math",
        east5_path="gs://marin-us-east5/tokenized/dolma3_dolmino_pool/synth_math_dolmino_math-6a90af",
        europe_path="gs://marin-eu-west4/tokenized/dolma3_dolmino_pool/synth_math_dolmino_math-6a90af",
        europe_rows=20_961_626,
        europe_tokens=10_708_625_130,
    ),
    ComponentDiagnostic(
        component="synth_qa/wiki_to_rcqa",
        slug="wiki-to-rcqa",
        east5_path="gs://marin-us-east5/tokenized/dolma3_dolmino_pool/synth_qa_wiki_to_rcqa-bd4afa",
        europe_path="gs://marin-eu-west4/tokenized/dolma3_dolmino_pool/synth_qa_wiki_to_rcqa-bd4afa",
        europe_rows=22_340_366,
        europe_tokens=4_254_057_998,
    ),
    ComponentDiagnostic(
        component="synth_thinking/code_meta_reasoning",
        slug="code-meta-reasoning",
        east5_path="gs://marin-us-east5/tokenized/dolma3_dolmino_pool/synth_thinking_code_meta_reasoning-89ea11",
        europe_path="gs://marin-eu-west4/tokenized/dolma3_dolmino_pool/synth_thinking_code_meta_reasoning-89ea11",
        europe_rows=910_921,
        europe_tokens=1_267_465_227,
    ),
    ComponentDiagnostic(
        component="synth_thinking/math_meta_reasoning",
        slug="math-meta-reasoning",
        east5_path="gs://marin-us-east5/tokenized/dolma3_dolmino_pool/synth_thinking_math_meta_reasoning-c0fdb1",
        europe_path="gs://marin-eu-west4/tokenized/dolma3_dolmino_pool/synth_thinking_math_meta_reasoning-c0fdb1",
        europe_rows=984_610,
        europe_tokens=1_051_508_851,
    ),
    ComponentDiagnostic(
        component="synth_thinking/program_verifiable",
        slug="program-verifiable",
        east5_path="gs://marin-us-east5/tokenized/dolma3_dolmino_pool/synth_thinking_program_verifiable-bc5995",
        europe_path="gs://marin-eu-west4/tokenized/dolma3_dolmino_pool/synth_thinking_program_verifiable-bc5995",
        europe_rows=273_431,
        europe_tokens=391_616_017,
    ),
)

REGIONS = {
    "east5": ("us-east5", "us-east5-a", "gs://marin-us-east5"),
    "europe": ("europe-west4", "europe-west4-b", "gs://marin-eu-west4"),
}
O4MINI_PATHS = {
    "east5": "gs://marin-us-east5/tokenized/dolma3_dolmino_pool/synth_math_verifiable_o4mini-2cbec0",
    "europe": "gs://marin-eu-west4/tokenized/dolma3_dolmino_pool/synth_math_verifiable_o4mini-2cbec0",
}
O4MINI_COMPONENT = "synth_math/verifiable_o4mini"


def digest_job_command_tokens(
    *,
    job_name: str,
    region: str,
    zone: str,
    marin_prefix: str,
    cache_path: str,
    output: str,
    expected_rows: int,
    expected_tokens: int,
    excluded_shards: tuple[str, ...],
    memory: str,
) -> list[str]:
    tokens = ["UV_FROZEN=1", "uv", "run", "python", "-m", "marin.run.iris_run", "--config", "lib/iris/config/marin.yaml"]
    for path in WORKING_DIR_EXCLUDES:
        tokens.extend(("--working-dir-exclude", path))
    tokens.extend(
        (
            "--",
            "--no-wait",
            "--no-preemptible",
            "--job-name",
            job_name,
            "--region",
            region,
            "--zone",
            zone,
            "--priority",
            "interactive",
            "--enable-extra-resources",
            "--cpu",
            "2",
            "--memory",
            memory,
            "--disk",
            "16GB",
            "--timeout",
            str(TIMEOUT_SECONDS),
            "--extra",
            "cpu",
            "-e",
            "MARIN_PREFIX",
            marin_prefix,
            "--",
            "python",
            "-m",
            "experiments.domain_phase_mix.digest_delphi_tpp40_runtime_cache",
            "--cache-path",
            cache_path,
            "--output",
            output,
            "--expect-rows",
            str(expected_rows),
            "--expect-tokens",
            str(expected_tokens),
        )
    )
    for shard in excluded_shards:
        tokens.extend(("--exclude-shard", shard))
    return tokens


def materialize_jobs() -> tuple[DigestJob, ...]:
    jobs: list[DigestJob] = []
    for component in COMPONENTS:
        east5_rows, east5_tokens = DIGEST_EXPECTED_COUNTS[component.component]
        for region_key in ("east5", "europe"):
            region, zone, marin_prefix = REGIONS[region_key]
            cache_path = component.east5_path if region_key == "east5" else component.europe_path
            expected_rows = east5_rows if region_key == "east5" else component.europe_rows
            expected_tokens = east5_tokens if region_key == "east5" else component.europe_tokens
            excluded_shards = () if region_key == "east5" else component.europe_excluded_shards
            job_name = f"dm-delphi-tpp40-runtime-digest-{component.slug}-{region_key}-v4-diagnostic-{DATE}"
            output_namespace = (
                "delphi_tpp40_multiregion_runtime_digests_v4"
                if region_key == "east5"
                else "delphi_tpp40_multiregion_runtime_digests_v4_diagnostic"
            )
            output = (
                f"{marin_prefix}/experiments/domain_phase_mix/{output_namespace}/"
                f"{digest_comparison_filename(component.component)}"
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
                    excluded_shards=excluded_shards,
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
                    excluded_shards=excluded_shards,
                    memory=component.memory,
                    command=command,
                )
            )
    return tuple(jobs)


def materialize_canary_jobs() -> tuple[DigestJob, ...]:
    expected_rows, expected_tokens = DIGEST_EXPECTED_COUNTS[O4MINI_COMPONENT]
    jobs: list[DigestJob] = []
    for region_key in ("east5", "europe"):
        region, zone, marin_prefix = REGIONS[region_key]
        job_name = f"dm-delphi-tpp40-runtime-digest-o4mini-{region_key}-v4-{DATE}"
        output = (
            f"{marin_prefix}/experiments/domain_phase_mix/"
            "delphi_tpp40_multiregion_runtime_digests_v4/synth_math_verifiable_o4mini.json"
        )
        command = shlex.join(
            digest_job_command_tokens(
                job_name=job_name,
                region=region,
                zone=zone,
                marin_prefix=marin_prefix,
                cache_path=O4MINI_PATHS[region_key],
                output=output,
                expected_rows=expected_rows,
                expected_tokens=expected_tokens,
                excluded_shards=(),
                memory="8GB",
            )
        )
        jobs.append(
            DigestJob(
                component=O4MINI_COMPONENT,
                region_key=region_key,
                job_name=job_name,
                region=region,
                zone=zone,
                marin_prefix=marin_prefix,
                cache_path=O4MINI_PATHS[region_key],
                output=output,
                expected_rows=expected_rows,
                expected_tokens=expected_tokens,
                excluded_shards=(),
                memory="8GB",
                command=command,
            )
        )
    return tuple(jobs)


def validate_runtime_paths(pairs: tuple[CachePair, ...] | None = None) -> None:
    pairs = cache_pairs() if pairs is None else pairs
    audited_components = {component.component for component in COMPONENTS} | {O4MINI_COMPONENT}
    resolved = {pair.component: pair for pair in pairs if pair.component in audited_components}
    if set(resolved) != audited_components:
        raise ValueError(f"Production runtime pairs differ from the digest matrix: {sorted(resolved)}")
    expected_paths = {
        component.component: {"east5": component.east5_path, "europe": component.europe_path} for component in COMPONENTS
    }
    expected_paths[O4MINI_COMPONENT] = O4MINI_PATHS
    for component, region_paths in expected_paths.items():
        pair = resolved[component]
        actual_paths = {"east5": pair.east5_path, "europe": pair.europe_path}
        if actual_paths != region_paths:
            raise ValueError(f"Digest paths differ from production runtime paths for {component!r}")


def manifest() -> dict[str, object]:
    jobs = materialize_jobs()
    canary_jobs = materialize_canary_jobs()
    return {
        "algorithm": ALGORITHM,
        "mode": "diagnostic_only",
        "acceptance_requires_zero_exclusions": True,
        "canary_jobs": [asdict(job) for job in canary_jobs],
        "jobs": [asdict(job) for job in jobs],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validate_runtime_paths()
    payload = json.dumps(manifest(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
