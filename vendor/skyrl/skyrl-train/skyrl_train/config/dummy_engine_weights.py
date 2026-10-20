from omegaconf import DictConfig


def dummy_engine_weights_requested(load_format: str | None) -> bool:
    return load_format == "dummy"


def validate_dummy_engine_weights(cfg: DictConfig) -> None:
    engine_kwargs = cfg.generator.engine_init_kwargs
    if cfg.generator.backend == "vllm" and engine_kwargs.get("load_format") is not None:
        engine_kwargs["load_format"] = engine_kwargs["load_format"].lower()
    if dummy_engine_weights_requested(engine_kwargs.get("load_format")):
        failures = []
        if cfg.generator.weight_sync_transport != "expert_block":
            failures.append("generator.weight_sync_transport=expert_block")
        if not cfg.generator.run_engines_locally:
            failures.append("generator.run_engines_locally=true")
        if cfg.generator.get("speculative_decoding") or engine_kwargs.get("speculative_config"):
            failures.append("no speculative decoding")
        if engine_kwargs.get("kv_cache_dtype", "auto") != "auto":
            failures.append("kv_cache_dtype=auto")
        if failures:
            raise ValueError("Dummy engine weights require: " + "; ".join(failures))
