# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shlex
from pathlib import Path

from experiments.domain_phase_mix.audit_delphi_tpp40_europe_runtime_caches import DIGEST_EXPECTED_COUNTS
from experiments.domain_phase_mix.materialize_delphi_tpp40_runtime_digest_acceptance import (
    ACCEPTANCE_COMPONENTS,
    manifest,
    materialize_jobs,
    validate_acceptance_paths,
)


def test_acceptance_matrix_covers_exact_zero_exclusion_payloads_in_both_regions() -> None:
    jobs = materialize_jobs()

    assert len(jobs) == 2 * len(DIGEST_EXPECTED_COUNTS)
    assert {component.component for component in ACCEPTANCE_COMPONENTS} == set(DIGEST_EXPECTED_COUNTS)
    assert len({job.job_name for job in jobs}) == len(jobs)
    assert len({job.output for job in jobs}) == len(jobs)
    for job in jobs:
        tokens = shlex.split(job.command)
        assert (job.expected_rows, job.expected_tokens) == DIGEST_EXPECTED_COUNTS[job.component]
        assert not job.excluded_shards
        assert "--exclude-shard" not in tokens
        assert job.cache_path.startswith(job.marin_prefix + "/")
        assert job.output.startswith(job.marin_prefix + "/")
        assert "/delphi_tpp40_multiregion_runtime_digests_v4" in job.output
        assert "_diagnostic/" not in job.output
        assert tokens[tokens.index("--region") + 1] == job.region
        assert tokens[tokens.index("--zone") + 1] == job.zone


def test_acceptance_europe_jobs_bind_historical_repair_outputs() -> None:
    europe_jobs = {job.component: job for job in materialize_jobs() if job.region_key == "europe"}

    assert europe_jobs["finemath_3plus"].cache_path.endswith(
        "/tokenized/finemath_3_plus_historical_full_document_v1-244ece"
    )
    assert europe_jobs["dolmino_stem_heavy_crawl"].cache_path.endswith(
        "/tokenized/merged/dolma3_dolmino_top_level_historical_full_document_v1/" "dolmino_stem_heavy_crawl-4f736e"
    )
    assert (
        "/delphi_tpp40_multiregion_runtime_digests_v4_stem_metadata_repair/"
        in europe_jobs["dolmino_stem_heavy_crawl"].output
    )
    assert all(
        "/delphi_tpp40_multiregion_runtime_digests_v4/" in job.output
        for component, job in europe_jobs.items()
        if component != "dolmino_stem_heavy_crawl"
    )
    for component, job in europe_jobs.items():
        if component in {
            "finemath_3plus",
            "dolmino_stem_heavy_crawl",
            "synth_instruction/dolmino_flan",
            "synth_math/verifiable_o4mini",
        }:
            continue
        assert "/tokenized/dolma3_dolmino_pool_historical_full_document_v1/" in job.cache_path
    assert europe_jobs["synth_instruction/dolmino_flan"].cache_path.endswith(
        "/tokenized/dolma3_dolmino_pool_historical_full_document_east5_subset_v1/"
        "synth_instruction_dolmino_flan-985ec1"
    )
    assert europe_jobs["synth_instruction/dolmino_flan"].memory == "14GB"
    assert "-v4-acceptance-retry1-" in europe_jobs["synth_instruction/dolmino_flan"].job_name


def test_acceptance_paths_match_east5_production_and_europe_repairs() -> None:
    validate_acceptance_paths()


def test_frozen_acceptance_manifest_matches_generator() -> None:
    frozen_path = (
        Path(__file__).resolve().parents[1]
        / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "delphi_tpp40_europe_readiness_20260830/runtime_digest_acceptance_manifest_v4_retry1.json"
    )
    expected = json.dumps(manifest(), indent=2, sort_keys=True) + "\n"

    assert frozen_path.read_text() == expected
