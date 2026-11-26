# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure Qwen's initial difficulty and realized response caps on the topic control."""

from dataclasses import replace

from marin.rl.skyrl import SkyRLGenerationPlan
from skyrlrecipe import ContextBudget, RecipePatch

from experiments.post_training import rl
from experiments.post_training.qwen_sampler import EVAL, LAYOUT, sampler_preset
from experiments.post_training.rl_presets import Generation, ModelPresets, Preset
from experiments.post_training.task_curation.qwen_topic_control import NAME, VERSION

RESPONSE_TOKENS = 8192
CONTEXT_TOKENS = 512 + RESPONSE_TOKENS


def initial_probe(panel: str, samples: int, temperature: float) -> Preset:
    evaluation = replace(
        EVAL,
        name=panel,
        profile=EVAL.profile.replace(
            n_samples_per_prompt=samples,
            sampling_params=EVAL.profile.sampling_params.replace(
                temperature=temperature,
                max_generate_length=RESPONSE_TOKENS,
            ),
        ),
    )
    layout = replace(
        LAYOUT,
        name="qwen-h100x1",
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
        f"topic-control-{panel}",
        0,
        "adaptive-source",
        1.0,
        26605,
        layout=layout,
        revision=VERSION,
        evaluation_interval=None,
    )
    config = preset.recipe.config.merge(
        RecipePatch(
            context_budget=ContextBudget(
                request_window_tokens=CONTEXT_TOKENS,
                serving_window_tokens=CONTEXT_TOKENS,
                max_new_tokens_per_turn=RESPONSE_TOKENS,
                max_turns=1,
            )
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
    presets=(initial_probe("probe", 4, 1.0), initial_probe("dev", 1, 0.0)),
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_TOPIC_CONTROL))
