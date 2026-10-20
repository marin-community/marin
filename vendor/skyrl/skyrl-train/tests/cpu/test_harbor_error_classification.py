import importlib

from omegaconf import OmegaConf
import pytest

from skyrl_train.config.objective_spec import rollout_logprobs_required
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.harbor_errors import (
    ErrorHandlingConfig,
    ErrorTreatment,
    PASSTHROUGH_WITHOUT_LOGPROBS_ERROR,
    classify_exception_type,
    error_category,
    known_error_types,
    passthrough_logprob_error_type,
    treatment_excludes_from_baseline,
)


def test_local_taxonomy_matches_pinned_harbor_runtime():
    pytest.importorskip("harbor")
    harbor_taxonomy = importlib.import_module("harbor_config.errors")
    names = known_error_types() | harbor_taxonomy.known_error_types()

    assert {name: error_category(name).value for name in names} == {
        name: harbor_taxonomy.error_category(name).value for name in names
    }


def test_error_handling_mapping_is_validated_at_config_boundary():
    config = ErrorHandlingConfig.from_mapping(
        {
            "enable_error_classification": True,
            "mask_exceptions": ["CampaignInfrastructureError"],
            "zero_exceptions": "FirstAgentError, SecondAgentError",
            "default_error_treatment": "passthrough",
        }
    )

    assert config.enable_error_classification is True
    assert config.mask_exceptions == frozenset({"CampaignInfrastructureError"})
    assert config.zero_exceptions == frozenset({"FirstAgentError", "SecondAgentError"})
    assert config.default_error_treatment is ErrorTreatment.PASSTHROUGH


def test_context_management_failure_is_masked_from_persisted_exception_type():
    treatment = classify_exception_type(
        "ContextManagementInfrastructureError",
        ErrorHandlingConfig(default_error_treatment=ErrorTreatment.ZERO),
    )

    assert treatment is ErrorTreatment.MASK


def test_shared_taxonomy_maps_agent_and_passthrough_failures():
    config = ErrorHandlingConfig()
    assert classify_exception_type("AgentTimeoutError", config) is ErrorTreatment.ZERO
    assert classify_exception_type("OutputLengthExceededError", config) is ErrorTreatment.PASSTHROUGH


def test_campaign_override_takes_precedence_over_shared_taxonomy():
    treatment = classify_exception_type(
        "ContextManagementInfrastructureError",
        ErrorHandlingConfig(zero_exceptions=frozenset({"ContextManagementInfrastructureError"})),
    )

    assert treatment is ErrorTreatment.ZERO


def test_unknown_error_uses_explicit_fallback_treatment():
    treatment = classify_exception_type(
        "FutureHarborError",
        ErrorHandlingConfig(default_error_treatment=ErrorTreatment.MASK),
    )

    assert treatment is ErrorTreatment.MASK


def test_passthrough_requires_a_verifier_result_to_remain_in_baseline():
    assert treatment_excludes_from_baseline(ErrorTreatment.PASSTHROUGH, verifier_available=False) is True
    assert treatment_excludes_from_baseline(ErrorTreatment.PASSTHROUGH, verifier_available=True) is False
    assert treatment_excludes_from_baseline(ErrorTreatment.MASK, verifier_available=True) is True
    assert treatment_excludes_from_baseline(ErrorTreatment.ZERO, verifier_available=False) is False


@pytest.mark.parametrize(
    "algorithm_config",
    [
        {"off_policy_correction": "tis", "policy_loss_type": "regular"},
        {"off_policy_correction": "none", "policy_loss_type": "behavior_clip"},
    ],
)
def test_behavior_referenced_passthrough_without_logprobs_gets_named_error(algorithm_config):
    cfg = get_default_config()
    cfg.trainer.algorithm = OmegaConf.merge(cfg.trainer.algorithm, algorithm_config)

    error_type = passthrough_logprob_error_type(
        ErrorTreatment.PASSTHROUGH,
        has_rollout_logprobs=False,
        rollout_logprobs_required=rollout_logprobs_required(cfg.trainer.algorithm),
    )

    assert error_type == PASSTHROUGH_WITHOUT_LOGPROBS_ERROR


def test_passthrough_without_logprobs_is_unchanged_when_behavior_reference_is_not_required():
    assert (
        passthrough_logprob_error_type(
            ErrorTreatment.PASSTHROUGH,
            has_rollout_logprobs=False,
            rollout_logprobs_required=False,
        )
        is None
    )


def test_passthrough_with_logprobs_remains_trainable():
    assert (
        passthrough_logprob_error_type(
            ErrorTreatment.PASSTHROUGH,
            has_rollout_logprobs=True,
            rollout_logprobs_required=True,
        )
        is None
    )
