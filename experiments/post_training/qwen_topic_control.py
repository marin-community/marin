# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure Qwen's initial difficulty and realized response caps on the topic control."""

from dataclasses import dataclass, replace

from marin.rl.skyrl import SkyRLGenerationPlan
from skyrlrecipe import ContextBudget, RecipePatch, sections

from experiments.post_training import rl
from experiments.post_training.qwen_sampler import EVAL, LAYOUT, sampler_preset
from experiments.post_training.rl_presets import Generation, ModelPresets, Preset
from experiments.post_training.task_curation.qwen_topic_control import NAME, VERSION

CAP_REVISION = "2026.11.26.2"
INPUT_TOKENS = 512
NATIVE_WINDOW = 32768
YARN_WINDOW = 131072


@dataclass(frozen=True)
class ResponseProfile:
    name: str
    max_response: int
    thinking: bool
    yarn: bool = False


RESPONSE_PROFILES = (
    ResponseProfile("direct", 8192, thinking=False),
    ResponseProfile("native", NATIVE_WINDOW - INPUT_TOKENS, thinking=True),
    ResponseProfile("yarn", 65536, thinking=True, yarn=True),
)


def initial_probe(panel: str, samples: int, profile: ResponseProfile) -> Preset:
    parameters = EVAL.profile.sampling_params
    assert parameters is not None
    temperature = 0.6 if profile.thinking else (0.7 if samples > 1 else 0.0)
    context_tokens = INPUT_TOKENS + profile.max_response
    evaluation = replace(
        EVAL,
        name=panel,
        profile=EVAL.profile.replace(
            n_samples_per_prompt=samples,
            sampling_params=parameters.replace(
                temperature=temperature,
                max_generate_length=profile.max_response,
                top_p=0.95 if profile.thinking else 0.8,
                top_k=20,
                seed=26605,
            ),
        ),
    )
    layout = replace(
        LAYOUT,
        name="qwen-h100x1",
        job_timeout_seconds=3600,
        workload=Generation(
            SkyRLGenerationPlan(
                gpus_per_node=1,
                num_inference_engines=1,
                inference_engine_tensor_parallel_size=1,
                inference_engine_pipeline_parallel_size=1,
                inference_engine_data_parallel_size=1,
                inference_engine_expert_parallel_size=1,
                n_samples_per_prompt=samples,
            )
        ),
    )
    preset = sampler_preset(
        f"topic-{profile.name}-{panel}",
        0,
        "adaptive-source",
        1.0,
        26605,
        layout=layout,
        revision=CAP_REVISION,
        evaluation_interval=None,
    )
    engine_kwargs = {}
    if profile.yarn:
        engine_kwargs = {
            "hf_overrides": {
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": 1000000,
                    "factor": 4.0,
                    "original_max_position_embeddings": NATIVE_WINDOW,
                },
                "max_position_embeddings": YARN_WINDOW,
            },
        }
    config = preset.recipe.config.merge(
        RecipePatch(
            context_budget=ContextBudget(
                request_window_tokens=context_tokens,
                serving_window_tokens=context_tokens,
                max_new_tokens_per_turn=profile.max_response,
                max_turns=1,
            ),
            generator=sections.Generator(
                chat_template_kwargs={"enable_thinking": profile.thinking},
                engine_init_kwargs=engine_kwargs,
                gpu_memory_utilization=0.65,
                max_num_seqs=64,
            ),
        )
    )
    return replace(
        preset,
        tasks=f"{NAME}@{VERSION}",
        held_out_tasks=f"{NAME}-{panel}@{VERSION}",
        recipe=replace(preset.recipe, config=config),
        evaluations=(evaluation,),
        measured=range(1),
    )


QWEN_TOPIC_CONTROL = ModelPresets(
    model="qwen-topic-control",
    namespace="qwen-topic-control",
    control_pins={},
    presets=tuple(
        initial_probe(panel, samples, profile)
        for profile in RESPONSE_PROFILES
        for panel, samples in (("probe", 4), ("dev", 1))
    ),
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_TOPIC_CONTROL))
