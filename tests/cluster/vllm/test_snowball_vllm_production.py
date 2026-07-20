# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
from marin.inference.tpu_vllm_pins import TPU_INFERENCE_FORK_REV, VLLM_FORK_REV
from marin.inference.vllm_server import VllmRuntimeFingerprint

from tests.cluster.vllm.snowball import PROMPT_FIXTURE_SHA256, SNOWBALL
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
            )
            for repeat in range(SEQUENTIAL_REPEATS)
        ),
        concurrent=tuple(
            ProductionCompletion(case_id=case_id, wave=wave, token_ids=(token_id,), cached_prompt_tokens=0)
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
    assert_production_behavior_matches_oracle(round_tripped, oracle)

    bad = dataclasses.replace(
        round_tripped,
        sequential=(
            ProductionCompletion(
                case_id=ORACLE_CASE_ID,
                wave=0,
                token_ids=(999,),
                cached_prompt_tokens=0,
            ),
            *round_tripped.sequential[1:],
        ),
    )
    with pytest.raises(AssertionError):
        assert_production_behavior_matches_oracle(bad, oracle)


def test_production_oracle_is_frozen_from_observed_cache_hits() -> None:
    oracle = read_production_behavior_oracle()

    assert oracle.prefix_caching is True
    assert oracle.runtime_fingerprint_digest == "08a06735a23e90d382d2b6074214e8697143fbad57f7b1f4eb6aa61ab8c36c70"
    assert dict(oracle.concurrent_first_tokens)["instruction-ifeval-01"] == 3234
