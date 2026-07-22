# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed production-serving oracle for the Snowball TPU cell."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tests.cluster.vllm.backend_parity import ParityContract
from tests.cluster.vllm.snowball_vllm_production import (
    CONCURRENT_WAVES,
    SEQUENTIAL_REPEATS,
    ProductionBehaviorReport,
)

_RESOURCE_PATH = Path(__file__).parent / "resources" / "snowball_vllm_tpu_production_oracle_v2.json"
PRODUCTION_ORACLE_SHA256 = "c4317974ff8ec8bae1eca30357ea7c5054ef42051f8824266ab8c940efbc1acf"


@dataclass(frozen=True)
class ProductionBehaviorOracle:
    parameter_digest: str
    model_config_digest: str
    prompt_fixture_digest: str
    code_digest: str
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
        code_digest=payload["code_digest"],
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
    numerical_contract: ParityContract,
) -> None:
    """Apply the frozen topology, provenance, cache, and greedy-token oracle."""
    failures = []
    fields = (
        ("parameter_digest", report.parameter_digest, oracle.parameter_digest),
        ("model_config_digest", report.model_config_digest, oracle.model_config_digest),
        ("prompt_fixture_digest", report.prompt_fixture_digest, oracle.prompt_fixture_digest),
        ("code_digest", report.code_digest, oracle.code_digest),
        ("prefix_caching", report.prefix_caching, oracle.prefix_caching),
        ("max_num_seqs", report.max_num_seqs, oracle.max_num_seqs),
        ("tensor_parallel_size", report.tensor_parallel_size, oracle.tensor_parallel_size),
        ("data_parallel_size", report.data_parallel_size, oracle.data_parallel_size),
        ("fork_source_revisions", report.fork_source_revisions, oracle.fork_source_revisions),
        ("runtime_fingerprint_digest", report.runtime_fingerprint.digest(), oracle.runtime_fingerprint_digest),
    )
    for name, actual, expected in fields:
        if actual != expected:
            failures.append(f"{name}: expected {expected!r}, got {actual!r}")

    if len(report.sequential) != SEQUENTIAL_REPEATS:
        failures.append(f"sequential count: expected {SEQUENTIAL_REPEATS}, got {len(report.sequential)}")
    for completion in report.sequential:
        if completion.case_id != oracle.sequential_case_id:
            failures.append(
                f"sequential wave {completion.wave} case: expected {oracle.sequential_case_id!r}, "
                f"got {completion.case_id!r}"
            )
        if completion.wave not in range(SEQUENTIAL_REPEATS):
            failures.append(f"unexpected sequential wave {completion.wave}")
        if completion.token_ids != oracle.sequential_continuation:
            failures.append(
                f"sequential wave {completion.wave} tokens: expected {oracle.sequential_continuation}, "
                f"got {completion.token_ids}"
            )
        if completion.wave == 0 and completion.cached_prompt_tokens != 0:
            failures.append(f"sequential cold request cached {completion.cached_prompt_tokens} prompt tokens")
        if completion.wave > 0 and completion.cached_prompt_tokens <= 0:
            failures.append(f"sequential hit wave {completion.wave} reported no cached prompt tokens")
        _append_parity_integrity_failure(failures, completion, label=f"sequential wave {completion.wave}")

    expected_first_tokens = dict(oracle.concurrent_first_tokens)
    expected_concurrent_count = CONCURRENT_WAVES * oracle.max_num_seqs
    if len(report.concurrent) != expected_concurrent_count:
        failures.append(f"concurrent count: expected {expected_concurrent_count}, got {len(report.concurrent)}")
    for wave in range(CONCURRENT_WAVES):
        completions = [completion for completion in report.concurrent if completion.wave == wave]
        actual_case_ids = {completion.case_id for completion in completions}
        if actual_case_ids != expected_first_tokens.keys():
            failures.append(
                f"concurrent wave {wave} cases: expected {sorted(expected_first_tokens)}, got {sorted(actual_case_ids)}"
            )
        for completion in completions:
            expected_token = expected_first_tokens.get(completion.case_id)
            if expected_token is not None and completion.token_ids != (expected_token,):
                _append_parity_integrity_failure(
                    failures,
                    completion,
                    label=f"concurrent wave {wave} case {completion.case_id}",
                )
                bucket_bounds = dict(numerical_contract.max_probability_error_by_bucket)
                bound = bucket_bounds.get(256, numerical_contract.max_probability_error)
                try:
                    completion.first_token_parity.assert_matches(max_probability_error=bound)
                except AssertionError:
                    failures.append(
                        f"concurrent wave {wave} case {completion.case_id} tokens: expected {(expected_token,)}, "
                        f"got {completion.token_ids}; alternate winner is outside the 256-token numerical contract: "
                        f"{completion.first_token_parity}"
                    )
            if wave > 0 and completion.cached_prompt_tokens <= 0:
                failures.append(
                    f"concurrent cache-hit wave {wave} case {completion.case_id} reported no cached prompt tokens"
                )
            if expected_token is not None and completion.token_ids == (expected_token,):
                _append_parity_integrity_failure(
                    failures,
                    completion,
                    label=f"concurrent wave {wave} case {completion.case_id}",
                )

    if failures:
        raise AssertionError("Production behavior mismatches:\n" + "\n".join(f"- {failure}" for failure in failures))


def _append_parity_integrity_failure(
    failures: list[str],
    completion,
    *,
    label: str,
) -> None:
    parity = completion.first_token_parity
    if parity.case_id != completion.case_id:
        failures.append(f"{label} parity case: expected {completion.case_id!r}, got {parity.case_id!r}")
    if not completion.token_ids or parity.greedy_token_id != completion.token_ids[0]:
        failures.append(f"{label} parity token does not match returned token ids")
