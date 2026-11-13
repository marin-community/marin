from skyrl_train.inference_engines.vllm.utils import apply_openai_sampling


def test_training_request_inherits_configured_sampling_without_mutation() -> None:
    request = {"logprobs": 0}
    training = {"temperature": 0.8, "top_p": 1.0, "top_k": -1, "min_p": 0.0}

    body = apply_openai_sampling(request, training, validate_rollout_logprob_sampling=True)

    assert body == {"logprobs": 0, "temperature": 0.8, "top_p": 1.0, "top_k": -1, "min_p": 0.0}
    assert request == {"logprobs": 0}
    assert training == {"temperature": 0.8, "top_p": 1.0, "top_k": -1, "min_p": 0.0}
