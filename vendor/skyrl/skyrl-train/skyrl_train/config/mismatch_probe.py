"""CPU-only launch contract for the synchronous mismatch probe."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from skyrl_train.config.behavior_logprobs import (
    ROLLOUT_LOGPROB_ENGINE_OPTIONS,
    validate_behavior_logprob_sampling,
)
from skyrl_train.mismatch_probe.modes import (
    NATIVE_MODE,
    REPEAT_MODE,
    TRAINER_MODES,
)

CACHE_OFF = "off"
CACHE_ON = "on"
CACHE_BOTH = "both"
PROBE_CACHE_MODES = frozenset({CACHE_OFF, CACHE_ON, CACHE_BOTH})
GENERATION_SCORER = "vllm.generate"
RESCORE_SCORER = "vllm.rescore"
TRAINER_SCORER = "trainer"


def rescore_label(update: int, cache_mode: str) -> str:
    """Name a vLLM score group by update and prefix-cache mode."""
    return f"{RESCORE_SCORER}@{update}" if cache_mode == CACHE_OFF else f"{RESCORE_SCORER}@{update}:{cache_mode}"


def trainer_label(update: int, mode: str) -> str:
    """Name a Megatron score group by update and routing mode."""
    return f"{TRAINER_SCORER}@{update}:{mode}"


def validate_mismatch_probe_config(
    skyrl: Mapping,
    *,
    synchronous: bool = True,
) -> None:
    """Reject invalid probe recipes before worker allocation and at runtime."""
    trainer = skyrl.get("trainer", {})
    probe = trainer.get("mismatch_probe") or {}
    if not probe.get("enabled", False):
        return
    if not synchronous or trainer.get("rollout_buffer", {}).get("max_staleness_steps", 0) != 0:
        raise ValueError("trainer.mismatch_probe requires the synchronous trainer")
    if trainer.get("step_wise_training", False):
        raise ValueError(
            "trainer.mismatch_probe requires whole-trajectory token identity; step-wise probing is unsupported"
        )
    prompts = probe.get("prompts") or {}
    for field in ("count", "samples_per_prompt"):
        value = prompts.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"trainer.mismatch_probe.prompts.{field} must be a positive integer")
    if prompts["count"] * prompts["samples_per_prompt"] < 2:
        raise ValueError("trainer.mismatch_probe requires at least two samples to change repeat packing")
    seed = probe.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("trainer.mismatch_probe.seed must be a non-negative integer")
    updates = probe.get("updates")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("trainer.mismatch_probe.updates must be a non-negative integer")
    modes = probe.get("extra_trainer_modes") or []
    if (
        not isinstance(modes, Sequence)
        or isinstance(modes, str)
        or len(modes) != len(set(modes))
        or any(mode not in TRAINER_MODES or mode in {NATIVE_MODE, REPEAT_MODE} for mode in modes)
    ):
        raise ValueError("trainer.mismatch_probe.extra_trainer_modes contains an unsupported or repeated mode")
    cache_mode = probe.get("rescore_prefix_cache", CACHE_OFF)
    if cache_mode not in PROBE_CACHE_MODES:
        raise ValueError("trainer.mismatch_probe.rescore_prefix_cache must be off, on or both")
    reuse = probe.get("reuse_probe")
    if reuse is not None and (not isinstance(reuse, str) or not reuse):
        raise ValueError("trainer.mismatch_probe.reuse_probe must be a non-empty archive URI or null")
    archive_uri = probe.get("archive_uri")
    if not isinstance(archive_uri, str) or not archive_uri:
        raise ValueError("trainer.mismatch_probe.archive_uri must be a non-empty durable URI")

    generator = skyrl.get("generator", {})
    if generator.get("backend") != "vllm":
        raise ValueError("trainer.mismatch_probe requires a vLLM generator")
    sampling = generator.get("sampling_params") or {}
    validate_behavior_logprob_sampling(sampling)
    if sampling.get("temperature") != 1.0:
        raise ValueError("trainer.mismatch_probe requires generator.sampling_params.temperature=1")
    if sampling.get("logprobs") is None:
        raise ValueError("trainer.mismatch_probe requires processed generation logprobs")
    engine_options = generator.get("engine_init_kwargs") or {}
    for name, required in ROLLOUT_LOGPROB_ENGINE_OPTIONS.items():
        if engine_options.get(name) != required:
            raise ValueError(f"trainer.mismatch_probe requires generator.engine_init_kwargs.{name}={required!r}")
    if engine_options.get("override_generation_config") or engine_options.get("logits_processors"):
        raise ValueError("trainer.mismatch_probe rejects engine generation overrides and logits processors")

    if any(TRAINER_MODES[mode].requires_routes for mode in modes):
        policy = trainer.get("policy") or {}
        megatron = policy.get("megatron_config") or {}
        capture = generator.get("enable_return_routed_experts") or engine_options.get("enable_return_routed_experts")
        if not capture or not megatron.get("moe_router_replay"):
            raise ValueError("trainer.mismatch_probe replay modes require route capture and Megatron route consumption")
        topk = megatron.get("moe_router_topk")
        if topk is not None and topk < 2:
            raise ValueError("trainer.mismatch_probe replay modes require an MoE router with top-k >= 2")
    if any(TRAINER_MODES[mode].requires_keep_fraction for mode in modes):
        fraction = (probe.get("filtered_replay") or {}).get("keep_fraction")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ValueError("trainer.mismatch_probe.filtered_replay.keep_fraction must be explicit and in [0, 1]")
