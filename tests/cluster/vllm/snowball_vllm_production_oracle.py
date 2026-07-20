# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed production-serving oracle for the Snowball TPU cell."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tests.cluster.vllm.snowball_vllm_production import (
    CONCURRENT_WAVES,
    SEQUENTIAL_REPEATS,
    ProductionBehaviorReport,
)

_RESOURCE_PATH = Path(__file__).parent / "resources" / "snowball_vllm_tpu_production_oracle_v2.json"
PRODUCTION_ORACLE_SHA256 = "9f97cef06367111f3a64627c6ea2912da112ab8e086f93b0179d7f22d97766a3"


@dataclass(frozen=True)
class ProductionBehaviorOracle:
    parameter_digest: str
    model_config_digest: str
    prompt_fixture_digest: str
    prefix_caching: bool
    max_num_seqs: int
    tensor_parallel_size: int
    data_parallel_size: int
    fork_source_revisions: tuple[tuple[str, str], ...]
    runtime_fingerprint_digest: str
    sequential_case_id: str
    sequential_continuation: tuple[int, ...]
    concurrent_first_tokens: tuple[tuple[str, int], ...]


def read_production_behavior_oracle() -> ProductionBehaviorOracle:
    payload_bytes = _RESOURCE_PATH.read_bytes()
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != PRODUCTION_ORACLE_SHA256:
        raise ValueError(f"Production oracle SHA-256 mismatch: expected {PRODUCTION_ORACLE_SHA256}, got {actual_sha256}")
    payload = json.loads(payload_bytes)
    if payload["schema_version"] != 1:
        raise ValueError(f"Unsupported production oracle schema {payload['schema_version']}")
    return ProductionBehaviorOracle(
        parameter_digest=payload["parameter_sha256"],
        model_config_digest=payload["model_config_digest"],
        prompt_fixture_digest=payload["prompt_fixture_sha256"],
        prefix_caching=bool(payload["prefix_caching"]),
        max_num_seqs=int(payload["max_num_seqs"]),
        tensor_parallel_size=int(payload["tensor_parallel_size"]),
        data_parallel_size=int(payload["data_parallel_size"]),
        fork_source_revisions=tuple(tuple(revision) for revision in payload["fork_source_revisions"]),
        runtime_fingerprint_digest=payload["runtime_fingerprint_digest"],
        sequential_case_id=payload["sequential_case_id"],
        sequential_continuation=tuple(int(token_id) for token_id in payload["sequential_continuation"]),
        concurrent_first_tokens=tuple(
            (str(case_id), int(token_id)) for case_id, token_id in sorted(payload["concurrent_first_tokens"].items())
        ),
    )


def assert_production_behavior_matches_oracle(
    report: ProductionBehaviorReport,
    oracle: ProductionBehaviorOracle,
) -> None:
    """Apply the frozen topology, provenance, cache, and greedy-token oracle."""
    assert report.parameter_digest == oracle.parameter_digest
    assert report.model_config_digest == oracle.model_config_digest
    assert report.prompt_fixture_digest == oracle.prompt_fixture_digest
    assert report.prefix_caching is oracle.prefix_caching
    assert report.max_num_seqs == oracle.max_num_seqs
    assert report.tensor_parallel_size == oracle.tensor_parallel_size
    assert report.data_parallel_size == oracle.data_parallel_size
    assert report.fork_source_revisions == oracle.fork_source_revisions
    assert report.runtime_fingerprint.digest() == oracle.runtime_fingerprint_digest

    assert len(report.sequential) == SEQUENTIAL_REPEATS
    assert all(completion.case_id == oracle.sequential_case_id for completion in report.sequential)
    assert {completion.wave for completion in report.sequential} == set(range(SEQUENTIAL_REPEATS))
    assert all(completion.token_ids == oracle.sequential_continuation for completion in report.sequential)
    assert report.sequential[0].cached_prompt_tokens == 0
    assert all(completion.cached_prompt_tokens > 0 for completion in report.sequential[1:])

    expected_first_tokens = dict(oracle.concurrent_first_tokens)
    assert len(report.concurrent) == CONCURRENT_WAVES * oracle.max_num_seqs
    for wave in range(CONCURRENT_WAVES):
        completions = [completion for completion in report.concurrent if completion.wave == wave]
        assert {completion.case_id for completion in completions} == expected_first_tokens.keys()
        for completion in completions:
            assert completion.token_ids == (expected_first_tokens[completion.case_id],)
