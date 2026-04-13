# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small curriculum examples for Reasoning Gym-backed RL lessons.

These helpers are intentionally minimal. They show the expected `EnvConfig`
shape for `ReasoningGymEnv` without introducing another full launcher script.
"""

from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.rl_experiment_utils import RLExperimentConfig

DEFAULT_REASONING_GYM_EVAL_N_EXAMPLES = 128
REASONING_GYM_LEG_COUNTING_SEED = 42


def build_leg_counting_curriculum(
    run_id: str,
    config: RLExperimentConfig,
    eval_frequency: int,
) -> CurriculumConfig:
    """Build a minimal single-lesson Reasoning Gym curriculum example."""
    sampling_params = SamplingParams(
        temperature=1.0,
        n_prompts=config.n_prompts,
        n_generations_per_prompt=config.n_generations_per_prompt,
        max_output_tokens=config.max_output_tokens,
        top_k=config.inference_top_k,
        stop_tokens=None,
    )

    return CurriculumConfig(
        lessons={
            "rg_leg_counting": LessonConfig(
                lesson_id="rg_leg_counting",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.reasoning_gym_env.ReasoningGymEnv",
                    env_args={
                        "dataset_name": "leg_counting",
                        "train_dataset_args": {
                            "seed": REASONING_GYM_LEG_COUNTING_SEED,
                            "size": 10_000,
                            "min_animals": 2,
                            "max_animals": 4,
                        },
                        "eval_dataset_args": {
                            "seed": REASONING_GYM_LEG_COUNTING_SEED + 1,
                            "size": DEFAULT_REASONING_GYM_EVAL_N_EXAMPLES,
                            "min_animals": 2,
                            "max_animals": 4,
                        },
                        "success_threshold": 1.0,
                        "prompt_template": "{question}",
                    },
                ),
                dependencies=[],
                sampling_params=sampling_params,
            ),
        },
        eval_frequency=eval_frequency,
        micro_eval_frequency=None,
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=DEFAULT_REASONING_GYM_EVAL_N_EXAMPLES,
        max_seq_len=config.max_input_tokens + config.max_output_tokens,
    )
