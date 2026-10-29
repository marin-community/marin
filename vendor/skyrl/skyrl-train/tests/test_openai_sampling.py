import pytest

from skyrl_train.inference_engines.vllm.utils import apply_openai_sampling


@pytest.mark.parametrize("logprobs", [None, False])
def test_greedy_eval_request_overrides_training_defaults(logprobs: bool | None) -> None:
    request = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "seed": 20261020,
        "logprobs": logprobs,
    }
    training = {"temperature": 1.0, "top_p": 0.9, "top_k": 20, "min_p": 0.1}

    apply_openai_sampling(request, training, validate_rollout_logprob_sampling=True)

    assert request == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "seed": 20261020,
        "logprobs": logprobs,
    }


def test_training_request_inherits_configured_sampling() -> None:
    request = {"logprobs": 0}
    training = {"temperature": 0.8, "top_p": 1.0, "top_k": -1, "min_p": 0.0}

    apply_openai_sampling(request, training, validate_rollout_logprob_sampling=True)

    assert request == {"logprobs": 0, "temperature": 0.8, "top_p": 1.0, "top_k": -1, "min_p": 0.0}


@pytest.mark.parametrize("override", [{"temperature": 0.7}, {"top_p": 0.5}])
def test_training_request_rejects_incompatible_probability_distribution(override: dict[str, float]) -> None:
    request = {"logprobs": 0, **override}
    training = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}

    with pytest.raises(ValueError, match="Behavior-logprob"):
        apply_openai_sampling(request, training, validate_rollout_logprob_sampling=True)
