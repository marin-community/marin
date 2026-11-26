# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure small-model coverage on the admitted realistic sampler mixture."""

from dataclasses import replace

from iris.rpc import job_pb2
from rolloutengine.models import MachineRuntimeSpec
from shellbox.machine import Backend, NetworkPolicy
from skyrlrecipe import ContextBudget, sections

from experiments.post_training import rl
from experiments.post_training.qwen_topic_control import RESPONSE_PROFILES, initial_probe
from experiments.post_training.rl_presets import ModelPresets, Preset
from experiments.post_training.task_curation.qwen_realistic_sampler import MAX_PROMPT_TOKENS, NAME, VERSION

READINESS_REVISION = "2026.11.26.2"


def readiness(panel: str, copies: int, mode: str) -> Preset:
    (profile,) = (p for p in RESPONSE_PROFILES if p.name == mode)
    if mode == "native":
        profile = replace(profile, max_response=32768 - MAX_PROMPT_TOKENS)
    preset = initial_probe(panel, copies, profile)
    response = profile.max_response
    verifier = MachineRuntimeSpec(
        backend=Backend.LOCAL,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=600,
        cleanup_timeout=60,
    )
    config = preset.recipe.config.replace(
        environment=sections.Environment(
            task_sessions={"local": {}, "task_spec": {"package_verifier": verifier.model_dump(mode="json")}}
        ),
        context_budget=ContextBudget(
            request_window_tokens=MAX_PROMPT_TOKENS + response,
            serving_window_tokens=MAX_PROMPT_TOKENS + response,
            max_new_tokens_per_turn=response,
            max_turns=1,
        ),
    )
    return replace(
        preset,
        name=f"realistic-{mode}-{panel}",
        revision=READINESS_REVISION,
        tasks=f"{NAME}@{VERSION}",
        held_out_tasks=f"{NAME}-{panel}@{VERSION}",
        recipe=replace(
            preset.recipe,
            config=config,
            layout=replace(preset.recipe.layout, container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED),
        ),
    )


QWEN_REALISTIC = ModelPresets(
    model="qwen-realistic-sampler",
    namespace="qwen-realistic-sampler",
    control_pins={},
    presets=tuple(readiness("dev", 4, mode) for mode in ("direct", "native")),
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_REALISTIC))
