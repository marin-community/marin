# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pytest

from tests.cluster.vllm.backend_parity import (
    GoldenTokenObservation,
    NextTokenObservation,
    ObservationReport,
    ParityContract,
    ParityDiscovery,
    RunProvenance,
    TokenScore,
)
from tests.cluster.vllm.snowball import (
    PromptBatch,
    RepresentativeCase,
    RepresentativeGolden,
)
from tests.cluster.vllm.snowball_levanter import (
    apply_canonical_top25,
    assert_native_tpu_contract,
    build_batch_observations,
)


def test_build_batch_observations_retains_boundary_ranks_and_router_diagnostics() -> None:
    case = RepresentativeCase(
        id="case",
        prompt_token_ids=(1, 2),
        top_logprobs=(
            TokenScore(token_id=7, logprob=-0.1),
            TokenScore(token_id=9, logprob=-0.2),
        ),
    )
    batch = PromptBatch(max_tokens=256, cases=(case,))

    (observation,) = build_batch_observations(
        batch,
        repeat_index=2,
        top_logprobs=np.asarray([[-0.11, -0.21]], dtype=np.float32),
        top_token_ids=np.asarray([[7, 8]], dtype=np.int32),
        golden_logprobs=np.asarray([[-0.11, -0.31]], dtype=np.float32),
        golden_ranks=np.asarray([[0, 3]], dtype=np.int32),
        capacity_overflow=np.asarray([0.0, 0.0], dtype=np.float32),
        has_nonfinite=np.asarray([False]),
    )

    assert observation.case_id == "case"
    assert observation.repeat_index == 2
    assert observation.greedy_token_id == 7
    assert [score.token_id for score in observation.top_logprobs] == [7, 8]
    assert [(token.token_id, token.rank) for token in observation.golden_tokens] == [(7, 0), (9, 3)]
    assert observation.capacity_overflow == (0.0, 0.0)
    assert not observation.has_nonfinite


def test_apply_canonical_top25_replaces_only_canonical_token_observations() -> None:
    logprobs, ranks, greedy = apply_canonical_top25(
        golden_token_ids=np.asarray([[7, 9, 11]], dtype=np.int32),
        diagnostic_logprobs=np.asarray([[-0.7, -0.9, -1.1]], dtype=np.float32),
        diagnostic_ranks=np.asarray([[3, 4, 5]], dtype=np.int32),
        canonical_logprobs=np.asarray([[-0.1, -0.2]], dtype=np.float32),
        canonical_token_ids=np.asarray([[9, 7]], dtype=np.int32),
    )

    np.testing.assert_array_equal(logprobs, np.asarray([[-0.2, -0.1, -1.1]], dtype=np.float32))
    np.testing.assert_array_equal(ranks, np.asarray([[1, 0, 5]], dtype=np.int32))
    np.testing.assert_array_equal(greedy, np.asarray([9], dtype=np.int32))


def test_assert_native_tpu_contract_checks_provenance_and_numerics() -> None:
    golden = RepresentativeGolden(id="case", top_logprobs=(TokenScore(token_id=7, logprob=-0.1),))
    contract = ParityContract(
        schema_version=1,
        name="test",
        backend="levanter-native",
        platform="tpu",
        max_probability_error=0.01,
        parameter_digest="parameters",
        model_config_digest="config",
        prompt_fixture_digest="prompts",
        canonical_golden_digest="golden",
        requested_attention="splash",
        effective_attention="splash",
        requested_moe="ring",
        effective_moe="scatter",
        mesh_shape=(("data", 8),),
        discovery=ParityDiscovery(0.001, 3, 0.009, 0.0, 3, "summary"),
    )
    report = ObservationReport(
        provenance=RunProvenance(
            backend="levanter-native",
            platform="tpu",
            process_id="process",
            code_digest="code",
            parameter_digest="parameters",
            model_config_digest="config",
            prompt_fixture_digest="prompts",
            requested_attention="splash",
            effective_attention="splash",
            requested_moe="ring",
            effective_moe="scatter",
            mesh_shape=(("data", 8),),
            device_kind="TPU v6e",
            golden_digest="golden",
        ),
        observations=(
            NextTokenObservation(
                case_id="case",
                bucket_max_tokens=256,
                repeat_index=0,
                backend_index=0,
                greedy_token_id=7,
                top_logprobs=(TokenScore(token_id=7, logprob=-0.1),),
                golden_tokens=(GoldenTokenObservation(token_id=7, logprob=-0.1, rank=0),),
                capacity_overflow=(0.0,),
                has_nonfinite=False,
            ),
        ),
    )

    assert_native_tpu_contract(report, (golden,), contract)

    mismatched = dataclasses.replace(
        report,
        provenance=dataclasses.replace(report.provenance, parameter_digest="wrong"),
    )
    with pytest.raises(AssertionError, match="parameter_digest"):
        assert_native_tpu_contract(mismatched, (golden,), contract)
