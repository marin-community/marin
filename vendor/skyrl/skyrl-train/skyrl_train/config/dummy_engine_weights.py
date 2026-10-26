from omegaconf import DictConfig

from marinskyrl.resource_locator import RolloutModelLoading
from marinskyrl.runtime_options import WeightSyncTransport


def dummy_engine_weights_requested(cfg: DictConfig) -> bool:
    return cfg.generator.engine_init_kwargs.get("load_format") == "dummy"


def validate_dummy_engine_weights(cfg: DictConfig) -> None:
    engine_kwargs = cfg.generator.engine_init_kwargs
    if engine_kwargs.get("load_format") is not None:
        engine_kwargs["load_format"] = engine_kwargs["load_format"].lower()
    if dummy_engine_weights_requested(cfg):
        failures = []
        if cfg.generator.weight_sync_transport != WeightSyncTransport.EXPERT_BLOCK:
            failures.append("generator.weight_sync_transport=expert_block")
        if cfg.trainer.placement.colocate_all:
            failures.append("trainer.placement.colocate_all=false")
        if not cfg.generator.run_engines_locally:
            failures.append("generator.run_engines_locally=true")
        if cfg.generator.get("speculative_decoding") or engine_kwargs.get("speculative_config"):
            failures.append("no speculative decoding")
        if engine_kwargs.get("kv_cache_dtype", "auto") != "auto":
            failures.append("kv_cache_dtype=auto")
        if cfg.generator.model_loading == RolloutModelLoading.STAGE_LOCAL:
            failures.append("generator.model_loading must not be stage_local")
        if failures:
            raise ValueError("Dummy engine weights require: " + "; ".join(failures))
