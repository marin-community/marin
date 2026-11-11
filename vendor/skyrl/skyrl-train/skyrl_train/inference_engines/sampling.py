"""CPU-safe native sampling request contract and backend adapters.

Request keys include the pinned serving API parameters and native length aliases;
accepting a key does not add a default value to a request.
"""

from typing import Any, Dict

from omegaconf import DictConfig, ListConfig

# Public SamplingParams fields in the pinned marin.39e62869693c serving API,
# together with the SGLang adapter's request fields and native length alias.
SAMPLING_PARAMETER_KEYS = frozenset(
    {
        'allowed_token_ids',
        'bad_words',
        'custom_params',
        'detokenize',
        'ebnf',
        'extra_args',
        'flat_logprobs',
        'frequency_penalty',
        'ignore_eos',
        'include_stop_str_in_output',
        'json_schema',
        'logit_bias',
        'logprob_token_ids',
        'logprobs',
        'max_generate_length',
        'max_new_tokens',
        'max_tokens',
        'min_new_tokens',
        'min_p',
        'min_tokens',
        'n',
        'no_stop_trim',
        'output_kind',
        'output_text_buffer_length',
        'presence_penalty',
        'prompt_logprobs',
        'regex',
        'repetition_detection',
        'repetition_penalty',
        'routed_experts_prompt_start',
        'sampling_seed',
        'seed',
        'skip_clone',
        'skip_reading_prefix_cache',
        'skip_special_tokens',
        'spaces_between_special_tokens',
        'stop',
        'stop_regex',
        'stop_token_ids',
        'stream_interval',
        'structured_outputs',
        'temperature',
        'thinking_token_budget',
        'top_k',
        'top_p',
        'trace_decode_token_ids',
        'watermarking',
    }
)


def get_vllm_sampling_params(sampling_params: DictConfig) -> Dict[str, Any]:
    stop_val = sampling_params.get("stop", None)
    vllm_sampling_params = {
        "min_tokens": sampling_params.get("min_tokens", 1),
        "skip_special_tokens": True,
        "include_stop_str_in_output": True,
        "max_tokens": sampling_params.max_generate_length,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "logprobs": sampling_params.logprobs,
        "stop": list(stop_val) if stop_val is not None else None,
    }
    exclude_keys = ["max_generate_length"]
    for key, value in sampling_params.items():
        if key not in vllm_sampling_params and key not in exclude_keys:
            # Convert OmegaConf ListConfig to regular list if needed
            if isinstance(value, ListConfig):
                value = list(value)
            vllm_sampling_params[key] = value
    return vllm_sampling_params


def get_sglang_sampling_params(sampling_params: DictConfig) -> Dict[str, Any]:
    # `min_tokens` in vllm is equivalent to `min_new_tokens` in sglang. However `min_new_tokens` and
    # `stop` are not supported when `skip_tokenizer_init` is True, which we need for token-in-token-out.
    # See this issue for more: https://github.com/sgl-project/sglang/issues/9039#issuecomment-3218331087
    sglang_sampling_params = {
        "skip_special_tokens": True,
        "no_stop_trim": True,  # equivalent to include_stop_str_in_output=True
        "max_new_tokens": sampling_params.max_generate_length,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
    }
    # logprobs not supported with sglang for now
    exclude_keys = ["max_generate_length", "logprobs"]
    for key, value in sampling_params.items():
        if key not in sglang_sampling_params and key not in exclude_keys:
            # Convert OmegaConf ListConfig to regular list if needed
            if isinstance(value, ListConfig):
                value = list(value)
            sglang_sampling_params[key] = value
    return sglang_sampling_params


def get_sampling_params_for_backend(backend: str, sampling_params: DictConfig) -> Dict[str, Any]:
    if backend == "vllm":
        return get_vllm_sampling_params(sampling_params)
    elif backend == "sglang":
        return get_sglang_sampling_params(sampling_params)
    else:
        raise ValueError(f"Unsupported generation backend: {backend}")


