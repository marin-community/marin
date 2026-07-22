# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
from marin.inference.tpu_vllm_pins import TPU_INFERENCE_FORK_REV, VLLM_FORK_REV
from marin.inference.vllm_server import VllmRuntimeFingerprint

from tests.cluster.vllm.backend_parity import NextTokenParity
from tests.cluster.vllm.snowball import PROMPT_FIXTURE_SHA256, SNOWBALL, read_vllm_tpu_contract
from tests.cluster.vllm.snowball_vllm_production import (
    CONCURRENT_WAVES,
    MAX_NUM_SEQS,
    ORACLE_CASE_ID,
    SEQUENTIAL_REPEATS,
    ProductionBehaviorReport,
    ProductionCompletion,
)
from tests.cluster.vllm.snowball_vllm_production_oracle import (
    ProductionBehaviorOracle,
    assert_production_behavior_matches_oracle,
    read_production_behavior_oracle,
)

_EXPECTED_CONTINUATION = tuple(range(10, 18))
_EXPECTED_FIRST_TOKENS = tuple((f"case-{index}", 100 + index) for index in range(MAX_NUM_SEQS))


def _runtime_fingerprint() -> VllmRuntimeFingerprint:
    return VllmRuntimeFingerprint(
        packages=(),
        environment=(("LIBTPU_INIT_ARGS", "test"),),
        engine_args=("serve", "model"),
        launcher_args=("vllm",),
        python_version="3.12",
        platform="linux",
        libc=("glibc", "test"),
        os_release=(("ID", "test"),),
        cpu_affinity_count=1,
        isolation="container",
    )


def _parity(case_id: str, token_id: int) -> NextTokenParity:
    return NextTokenParity(
        case_id=case_id,
        backend_rank=0,
        greedy_token_id=token_id,
        golden_top_token_ids=(token_id,),
        golden_probability_gap_to_greedy=0.0,
        max_probability_error=0.01,
        top_probability_l1_error=0.01,
    )


def _report() -> ProductionBehaviorReport:
    report = ProductionBehaviorReport(
        parameter_digest=SNOWBALL.export_sha256,
        model_config_digest="config",
        prompt_fixture_digest=PROMPT_FIXTURE_SHA256,
        code_digest="code",
        prefix_caching=True,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=8,
        data_parallel_size=1,
        fork_source_revisions=(("vllm", VLLM_FORK_REV), ("tpu-inference", TPU_INFERENCE_FORK_REV)),
        runtime_fingerprint=_runtime_fingerprint(),
        sequential=tuple(
            ProductionCompletion(
                case_id=ORACLE_CASE_ID,
                wave=repeat,
                token_ids=_EXPECTED_CONTINUATION,
                cached_prompt_tokens=0 if repeat == 0 else 112,
                first_token_parity=_parity(ORACLE_CASE_ID, _EXPECTED_CONTINUATION[0]),
            )
            for repeat in range(SEQUENTIAL_REPEATS)
        ),
        concurrent=tuple(
            ProductionCompletion(
                case_id=case_id,
                wave=wave,
                token_ids=(token_id,),
                cached_prompt_tokens=0 if wave == 0 else 112,
                first_token_parity=_parity(case_id, token_id),
            )
            for wave in range(CONCURRENT_WAVES)
            for case_id, token_id in _EXPECTED_FIRST_TOKENS
        ),
    )
    return report


def test_production_behavior_round_trips_and_matches_oracles() -> None:
    report = _report()
    round_tripped = ProductionBehaviorReport.from_json_bytes(report.to_json_bytes())
    oracle = ProductionBehaviorOracle(
        parameter_digest=SNOWBALL.export_sha256,
        model_config_digest="config",
        prompt_fixture_digest=PROMPT_FIXTURE_SHA256,
        code_digest=report.code_digest,
        prefix_caching=True,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=8,
        data_parallel_size=1,
        fork_source_revisions=(("vllm", VLLM_FORK_REV), ("tpu-inference", TPU_INFERENCE_FORK_REV)),
        runtime_fingerprint_digest=_runtime_fingerprint().digest(),
        sequential_case_id=ORACLE_CASE_ID,
        sequential_continuation=_EXPECTED_CONTINUATION,
        concurrent_first_tokens=_EXPECTED_FIRST_TOKENS,
    )

    assert round_tripped == report
    contract = read_vllm_tpu_contract()
    assert_production_behavior_matches_oracle(round_tripped, oracle, contract)

    reference = round_tripped.concurrent[0]
    supported_alternate = dataclasses.replace(
        reference,
        token_ids=(999,),
        first_token_parity=dataclasses.replace(
            reference.first_token_parity,
            greedy_token_id=999,
            golden_top_token_ids=(reference.token_ids[0], 999),
            golden_probability_gap_to_greedy=0.01,
            max_probability_error=0.01,
        ),
    )
    supported_report = dataclasses.replace(
        round_tripped,
        concurrent=(supported_alternate, *round_tripped.concurrent[1:]),
    )
    assert_production_behavior_matches_oracle(supported_report, oracle, contract)

    unsupported_alternate = dataclasses.replace(
        supported_alternate,
        first_token_parity=dataclasses.replace(supported_alternate.first_token_parity, max_probability_error=0.04),
    )
    unsupported_report = dataclasses.replace(
        round_tripped,
        concurrent=(unsupported_alternate, *round_tripped.concurrent[1:]),
    )
    with pytest.raises(AssertionError, match="alternate winner is outside"):
        assert_production_behavior_matches_oracle(unsupported_report, oracle, contract)

    bad = dataclasses.replace(
        round_tripped,
        prefix_caching=False,
        sequential=(
            ProductionCompletion(
                case_id=ORACLE_CASE_ID,
                wave=0,
                token_ids=(999,),
                cached_prompt_tokens=0,
                first_token_parity=_parity(ORACLE_CASE_ID, 999),
            ),
            *round_tripped.sequential[1:],
        ),
    )
    with pytest.raises(AssertionError) as exc_info:
        assert_production_behavior_matches_oracle(bad, oracle, contract)
    assert "prefix_caching" in str(exc_info.value)
    assert "sequential wave 0 tokens" in str(exc_info.value)


def test_production_oracle_is_frozen_from_observed_cache_hits() -> None:
    oracle = read_production_behavior_oracle()

    assert oracle.prefix_caching is True
    assert oracle.runtime_fingerprint_digest == "9fd69b49defddd4d7115b644a9342fad95bdd1319e87ec52efa1d08ba10e5e7b"
    assert dict(oracle.concurrent_first_tokens)["instruction-ifeval-01"] == 3234
