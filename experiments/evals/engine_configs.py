"""Engine configuration for vLLM used for evals."""

DEFAULT_VLLM_ENGINE_KWARGS = {"max_model_len": 4096}

DEFAULT_LM_EVAL_MODEL_KWARGS = {**DEFAULT_VLLM_ENGINE_KWARGS, "max_gen_toks": 4096}
