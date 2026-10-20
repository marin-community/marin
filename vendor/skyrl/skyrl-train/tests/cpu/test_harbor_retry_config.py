"""Harbor retry policy preserves terminal results."""

from omegaconf import OmegaConf
import pytest
from rolloutengine.contracts import RolloutFailure

from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings


@pytest.mark.parametrize(
    "name,overrides,expected_delay",
    [
        ("NonZeroAgentExitCodeError", {}, None),
        ("UnknownPolicyFailure", {}, None),
        ("AgentTimeoutError", {"passthrough_exceptions": ["AgentTimeoutError"]}, None),
        ("OutputLengthExceededError", {}, None),
        ("TurnCapExhaustedError", {"zero_exceptions": ["TurnCapExhaustedError"]}, None),
        ("EnvironmentStartTimeoutError", {}, 1.0),
        ("ModelServerError", {"mask_exceptions": ["ModelServerError"]}, 1.0),
        ("VerifierTimeoutError", {"mask_exceptions": ["VerifierTimeoutError"]}, None),
        ("EnvironmentStartTimeoutError", {"exclude_exceptions": ["EnvironmentStartTimeoutError"]}, None),
        ("OutputLengthExceededError", {"enable_error_classification": False}, 1.0),
    ],
)
def test_harbor_retries_only_masked_failures(name, overrides, expected_delay):
    cfg = OmegaConf.create(
        {
            "harbor": {
                "enable_error_classification": True,
                "default_error_treatment": "zero",
                "exclude_exceptions": ["VerifierTimeoutError"],
                "max_retries": 2,
                **overrides,
            }
        }
    )
    settings = HarborTaskSettings.from_config(cfg)
    assert settings.retry_delay(RolloutFailure(name), retries=0) == expected_delay
    assert settings.retry_delay(RolloutFailure(name), retries=2) is None
