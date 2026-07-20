# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import pytest
from marin.inference.vllm_server import VllmRuntimeFingerprint

from tests.cluster.vllm.backend_parity import (
    GoldenTokenObservation,
    NextTokenObservation,
    NextTokenParity,
    ObservationReport,
    ParityContract,
    ParityDiscovery,
    RunProvenance,
    TokenScore,
    assert_report_matches_contract,
    assert_report_matches_exact_goldens,
    assert_report_matches_goldens,
    observation_from_completion_response,
    observations_bitwise_equal,
)


def _parity(*, greedy_token_id: int, gap: float, error: float) -> NextTokenParity:
    return NextTokenParity(
        case_id="case",
        backend_rank=0,
        greedy_token_id=greedy_token_id,
        golden_top_token_ids=(2, 3),
        golden_probability_gap_to_greedy=gap,
        max_probability_error=error,
        top_probability_l1_error=error,
    )


@pytest.mark.parametrize(
    "parity",
    [
        _parity(greedy_token_id=2, gap=0.0, error=0.0),
        _parity(greedy_token_id=3, gap=0.01, error=0.005),
    ],
)
def test_backend_distribution_contract_accepts_exact_and_error_explained_winners(parity: NextTokenParity) -> None:
    parity.assert_matches(max_probability_error=0.075)


@pytest.mark.parametrize(
    "parity",
    [
        _parity(greedy_token_id=3, gap=0.011, error=0.005),
        _parity(greedy_token_id=9, gap=0.01, error=0.01),
        _parity(greedy_token_id=2, gap=0.0, error=0.076),
    ],
)
def test_backend_distribution_contract_rejects_unexplained_outside_or_over_bound_winners(
    parity: NextTokenParity,
) -> None:
    with pytest.raises(AssertionError):
        parity.assert_matches(max_probability_error=0.075)


def _observation(logprob: float, *, repeat_index: int = 0) -> NextTokenObservation:
    return NextTokenObservation(
        case_id="case",
        bucket_max_tokens=256,
        repeat_index=repeat_index,
        backend_index=0,
        greedy_token_id=2,
        top_logprobs=(TokenScore(token_id=2, logprob=logprob),),
        golden_tokens=(GoldenTokenObservation(token_id=2, logprob=logprob, rank=0),),
        capacity_overflow=(0.0,),
        has_nonfinite=False,
    )


def _runtime_fingerprint() -> VllmRuntimeFingerprint:
    return VllmRuntimeFingerprint(
        packages=(),
        environment=(("LIBTPU_INIT_ARGS", "98304"),),
        engine_args=("serve", "model"),
        launcher_args=("uvx", "vllm"),
        python_version="3.12",
        platform="linux",
        libc=("glibc", "test"),
        os_release=(("ID", "test"),),
        cpu_affinity_count=160,
        isolation="container",
    )


def test_observation_report_json_round_trip_preserves_structured_diagnostics() -> None:
    report = ObservationReport(
        provenance=RunProvenance(
            backend="levanter-native",
            platform="tpu",
            process_id="process",
            code_digest="code",
            parameter_digest="parameters",
            model_config_digest="config",
            prompt_fixture_digest="prompt",
            requested_attention="tpu_splash",
            effective_attention="tpu_splash",
            requested_moe="ring",
            effective_moe="scatter",
            mesh_shape=(("data", 8), ("expert", 1)),
            device_kind="TPU v6e",
        ),
        observations=(_observation(-0.25),),
    )

    assert ObservationReport.from_json_bytes(report.to_json_bytes()) == report

    malformed = json.loads(report.to_json_bytes())
    del malformed["provenance"]["golden_digest"]
    with pytest.raises(KeyError, match="golden_digest"):
        ObservationReport.from_json_bytes(json.dumps(malformed).encode())


def test_observation_bitwise_equality_checks_float32_bits() -> None:
    assert observations_bitwise_equal(_observation(-0.25), _observation(-0.25, repeat_index=3))
    assert not observations_bitwise_equal(_observation(0.0), _observation(-0.0))


def test_completion_response_uses_the_shared_observation_schema() -> None:
    observation = observation_from_completion_response(
        {
            "choices": [
                {
                    "prompt_token_ids": [1, 2],
                    "token_ids": [7],
                    "logprobs": {
                        "top_logprobs": [
                            {
                                "token_id:8": -0.2,
                                "token_id:7": -0.1,
                            }
                        ]
                    },
                }
            ]
        },
        case_id="case",
        prompt_token_ids=(1, 2),
        expected_top_logprobs=(TokenScore(token_id=7, logprob=-0.11), TokenScore(token_id=9, logprob=-0.3)),
        bucket_max_tokens=256,
        repeat_index=2,
        backend_index=3,
    )

    assert observation.greedy_token_id == 7
    assert observation.top_logprobs == (
        TokenScore(token_id=7, logprob=-0.1),
        TokenScore(token_id=8, logprob=-0.2),
    )
    assert observation.golden_tokens == (
        GoldenTokenObservation(token_id=7, logprob=-0.1, rank=0),
        GoldenTokenObservation(token_id=9, logprob=None, rank=None),
    )


def test_observation_report_uses_the_shared_distribution_contract() -> None:
    report = ObservationReport(
        provenance=RunProvenance(
            backend="levanter-native",
            platform="tpu",
            process_id="process",
            code_digest="code",
            parameter_digest="parameters",
            model_config_digest="config",
            prompt_fixture_digest="prompt",
            requested_attention="tpu_splash",
            effective_attention="tpu_splash",
            requested_moe="ring",
            effective_moe="scatter",
            mesh_shape=(("data", 8),),
            device_kind="TPU v6e",
        ),
        observations=(_observation(-0.25), _observation(-0.25, repeat_index=1)),
    )
    expected = {"case": (TokenScore(token_id=2, logprob=-0.2),)}

    parities = assert_report_matches_goldens(report, expected, max_probability_error=0.05)

    assert len(parities) == 2
    assert parities[0].max_probability_error < 0.05


def test_shared_contract_checks_cell_provenance_topology_and_numerics() -> None:
    report = ObservationReport(
        provenance=RunProvenance(
            backend="levanter-exported",
            platform="tpu",
            process_id="process",
            code_digest="code",
            parameter_digest="parameters",
            model_config_digest="config",
            prompt_fixture_digest="prompt",
            requested_attention="tpu_splash",
            effective_attention="tpu_splash",
            requested_moe="ring",
            effective_moe="scatter",
            mesh_shape=(("data", 8), ("model", 1)),
            device_kind="TPU v6e",
            golden_digest="golden",
            fork_source_revisions=(("vllm", "a" * 40), ("tpu-inference", "b" * 40)),
            runtime_fingerprint=_runtime_fingerprint(),
        ),
        observations=(_observation(-0.25),),
    )
    contract = ParityContract(
        schema_version=1,
        name="cell",
        backend="levanter-exported",
        platform="tpu",
        max_probability_error=0.05,
        parameter_digest="parameters",
        model_config_digest="config",
        prompt_fixture_digest="prompt",
        canonical_golden_digest="golden",
        requested_attention="tpu_splash",
        effective_attention="tpu_splash",
        requested_moe="ring",
        effective_moe="scatter",
        mesh_shape=(("data", 8),),
        discovery=ParityDiscovery(0.04, 3, 0.01, 0.0, 3, "summary"),
        fork_source_revisions=(("vllm", "a" * 40), ("tpu-inference", "b" * 40)),
        runtime_fingerprint_digest=_runtime_fingerprint().digest(),
    )
    expected = {"case": (TokenScore(token_id=2, logprob=-0.2),)}

    assert_report_matches_contract(report, expected, contract)

    with pytest.raises(AssertionError, match="parameter_digest"):
        assert_report_matches_contract(
            dataclasses.replace(
                report,
                provenance=dataclasses.replace(report.provenance, parameter_digest="wrong"),
            ),
            expected,
            contract,
        )

    with pytest.raises(AssertionError, match="fork_source_revisions"):
        assert_report_matches_contract(
            dataclasses.replace(
                report,
                provenance=dataclasses.replace(
                    report.provenance,
                    fork_source_revisions=(("vllm", "c" * 40), ("tpu-inference", "b" * 40)),
                ),
            ),
            expected,
            contract,
        )

    with pytest.raises(AssertionError, match="runtime_fingerprint_digest"):
        assert_report_matches_contract(
            dataclasses.replace(
                report,
                provenance=dataclasses.replace(
                    report.provenance,
                    runtime_fingerprint=dataclasses.replace(_runtime_fingerprint(), isolation="host"),
                ),
            ),
            expected,
            contract,
        )


def test_observation_report_reports_every_bucket_before_failing() -> None:
    short = dataclasses.replace(_observation(-1.0), case_id="short", bucket_max_tokens=256)
    medium = dataclasses.replace(_observation(-1.0), case_id="medium", bucket_max_tokens=4096)
    report = ObservationReport(
        provenance=RunProvenance(
            backend="backend",
            platform="platform",
            process_id="process",
            code_digest="code",
            parameter_digest="parameters",
            model_config_digest="config",
            prompt_fixture_digest="prompt",
            requested_attention="attention",
            effective_attention="attention",
            requested_moe="moe",
            effective_moe="moe",
            mesh_shape=(),
            device_kind="device",
            golden_digest="golden",
        ),
        observations=(short, medium),
    )
    contract = ParityContract(
        schema_version=1,
        name="bucket-diagnostics",
        backend="backend",
        platform="platform",
        max_probability_error=0.05,
        parameter_digest="parameters",
        model_config_digest="config",
        prompt_fixture_digest="prompt",
        canonical_golden_digest="golden",
        requested_attention="attention",
        effective_attention="attention",
        requested_moe="moe",
        effective_moe="moe",
        mesh_shape=(),
        discovery=ParityDiscovery(0.01, 3, 0.01, 0.0, 3, "summary"),
        max_probability_error_by_bucket=((4096, 0.04),),
    )
    expected = {
        "short": (TokenScore(token_id=2, logprob=0.0),),
        "medium": (TokenScore(token_id=2, logprob=0.0),),
    }

    with pytest.raises(AssertionError) as error:
        assert_report_matches_contract(report, expected, contract)

    message = str(error.value)
    assert "bucket=256" in message
    assert "bucket=4096" in message
    assert "case=short" in message
    assert "case=medium" in message


def test_observation_report_applies_reviewed_bucket_bound() -> None:
    report = ObservationReport(
        provenance=RunProvenance(
            backend="levanter-exported",
            platform="tpu",
            process_id="process",
            code_digest="code",
            parameter_digest="parameters",
            model_config_digest="config",
            prompt_fixture_digest="prompt",
            requested_attention="attention",
            effective_attention="attention",
            requested_moe="moe",
            effective_moe="moe",
            mesh_shape=(("data", 8),),
            device_kind="TPU v6e",
            golden_digest="golden",
        ),
        observations=(
            dataclasses.replace(
                _observation(-0.1),
                bucket_max_tokens=32768,
            ),
        ),
    )
    contract = ParityContract(
        schema_version=1,
        name="length-aware-cell",
        backend="levanter-exported",
        platform="tpu",
        max_probability_error=0.05,
        parameter_digest="parameters",
        model_config_digest="config",
        prompt_fixture_digest="prompt",
        canonical_golden_digest="golden",
        requested_attention="attention",
        effective_attention="attention",
        requested_moe="moe",
        effective_moe="moe",
        mesh_shape=(("data", 8),),
        discovery=ParityDiscovery(0.1, 3, 0.1, 0.0, 3, "summary"),
        max_probability_error_by_bucket=((32768, 0.2),),
    )
    expected = {"case": (TokenScore(token_id=2, logprob=0.0),)}

    assert_report_matches_contract(report, expected, contract)

    with pytest.raises(AssertionError):
        assert_report_matches_contract(
            report,
            expected,
            dataclasses.replace(contract, max_probability_error_by_bucket=()),
        )


@pytest.mark.parametrize("score_source", ["top_logprobs", "canonical_tokens"])
def test_observation_report_uses_shared_exact_golden_comparison(score_source: str) -> None:
    report = ObservationReport(
        provenance=RunProvenance(
            backend="backend",
            platform="platform",
            process_id="process",
            code_digest="code",
            parameter_digest="parameters",
            model_config_digest="config",
            prompt_fixture_digest="prompt",
            requested_attention="attention",
            effective_attention="attention",
            requested_moe="moe",
            effective_moe="moe",
            mesh_shape=(),
            device_kind="device",
        ),
        observations=(_observation(-0.25),),
    )
    expected = {"case": (TokenScore(token_id=2, logprob=-0.25),)}

    assert_report_matches_exact_goldens(report, expected, score_source=score_source)

    with pytest.raises(AssertionError, match="case"):
        assert_report_matches_exact_goldens(
            report,
            {"case": (TokenScore(token_id=2, logprob=-0.2),)},
            score_source=score_source,
        )


@pytest.mark.parametrize("mutation", ["missing", "nonfinite", "overflow"])
def test_observation_report_rejects_incomplete_or_unhealthy_results(mutation: str) -> None:
    observation = _observation(-0.25)
    expected = {"case": (TokenScore(token_id=2, logprob=-0.2),)}
    if mutation == "missing":
        observation = dataclasses.replace(observation, golden_tokens=())
    elif mutation == "nonfinite":
        observation = dataclasses.replace(observation, has_nonfinite=True)
    else:
        observation = dataclasses.replace(observation, capacity_overflow=(1.0,))
    report = ObservationReport(
        provenance=RunProvenance(
            backend="backend",
            platform="platform",
            process_id="process",
            code_digest="code",
            parameter_digest="parameters",
            model_config_digest="config",
            prompt_fixture_digest="prompt",
            requested_attention="attention",
            effective_attention="attention",
            requested_moe="moe",
            effective_moe="moe",
            mesh_shape=(),
            device_kind="device",
        ),
        observations=(observation,),
    )

    with pytest.raises(AssertionError):
        assert_report_matches_goldens(report, expected, max_probability_error=0.05)
