# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shlex

import pytest

from experiments.domain_phase_mix.audit_delphi_tpp40_europe_runtime_caches import DIGEST_EXPECTED_COUNTS, CachePair
from experiments.domain_phase_mix.materialize_delphi_tpp40_runtime_digest_diagnostics import (
    COMPONENTS,
    O4MINI_COMPONENT,
    O4MINI_PATHS,
    materialize_canary_jobs,
    materialize_jobs,
    validate_runtime_paths,
)


def test_digest_diagnostic_matrix_is_region_local_and_count_bound() -> None:
    jobs = materialize_jobs()

    assert len(jobs) == 16
    assert len({job.job_name for job in jobs}) == len(jobs)
    assert len({job.output for job in jobs}) == len(jobs)
    for job in jobs:
        tokens = shlex.split(job.command)
        assert job.cache_path.startswith(job.marin_prefix + "/")
        assert job.output.startswith(job.marin_prefix + "/")
        assert tokens[tokens.index("--region") + 1] == job.region
        assert tokens[tokens.index("--zone") + 1] == job.zone
        assert tokens[tokens.index("--expect-rows") + 1] == str(job.expected_rows)
        assert tokens[tokens.index("--expect-tokens") + 1] == str(job.expected_tokens)
        if job.region_key == "east5":
            assert (job.expected_rows, job.expected_tokens) == DIGEST_EXPECTED_COUNTS[job.component]
            assert "/delphi_tpp40_multiregion_runtime_digests_v4/" in job.output
        else:
            assert "/delphi_tpp40_multiregion_runtime_digests_v4_diagnostic/" in job.output


def test_only_europe_flan_diagnostic_excludes_shards() -> None:
    jobs = materialize_jobs()
    excluded_jobs = [job for job in jobs if job.excluded_shards]

    assert len(excluded_jobs) == 1
    assert excluded_jobs[0].component == "synth_instruction/dolmino_flan"
    assert excluded_jobs[0].region_key == "europe"
    assert excluded_jobs[0].excluded_shards == (
        "part-00064-of-00209",
        "part-00122-of-00209",
        "part-00163-of-00209",
    )
    assert excluded_jobs[0].memory == "16GB"


def test_o4mini_canary_jobs_are_generated_under_acceptance_prefix() -> None:
    jobs = materialize_canary_jobs()

    assert len(jobs) == 2
    for job in jobs:
        assert job.component == O4MINI_COMPONENT
        assert job.cache_path == O4MINI_PATHS[job.region_key]
        assert (job.expected_rows, job.expected_tokens) == DIGEST_EXPECTED_COUNTS[O4MINI_COMPONENT]
        assert not job.excluded_shards
        assert "/delphi_tpp40_multiregion_runtime_digests_v4/" in job.output
        assert "_diagnostic/" not in job.output


def test_digest_paths_match_production_runtime_pairs() -> None:
    pairs = (
        *(
            CachePair(
                domain="domain",
                component=component.component,
                east5_path=component.east5_path,
                europe_path=component.europe_path,
            )
            for component in COMPONENTS
        ),
        CachePair(
            domain="domain",
            component=O4MINI_COMPONENT,
            east5_path=O4MINI_PATHS["east5"],
            europe_path=O4MINI_PATHS["europe"],
        ),
    )

    validate_runtime_paths(pairs)

    changed = list(pairs)
    changed[0] = CachePair(
        domain=changed[0].domain,
        component=changed[0].component,
        east5_path=changed[0].east5_path + "-different",
        europe_path=changed[0].europe_path,
    )
    with pytest.raises(ValueError, match="differ from production runtime paths"):
        validate_runtime_paths(tuple(changed))
