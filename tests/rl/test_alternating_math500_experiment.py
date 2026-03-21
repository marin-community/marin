# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from experiments.alternating_rl_math500 import (
    ALTERNATING_WANDB_PROJECT,
    _experiment_config_from_args,
    _default_experiment_config,
    build_parser,
    create_math_curriculum,
)


def test_experiment_overrides_support_small_smoke_profile():
    config = _experiment_config_from_args(
        SimpleNamespace(
            train_batch_size=16,
            max_input_tokens=512,
            max_output_tokens=256,
            n_prompts=4,
            n_generations_per_prompt=4,
            inference_gpu_memory_utilization=0.95,
        )
    )

    assert config.train_batch_size == 16
    assert config.max_input_tokens == 512
    assert config.max_output_tokens == 256
    assert config.n_prompts == 4
    assert config.n_generations_per_prompt == 4
    assert config.inference_gpu_memory_utilization == 0.95


def test_math_curriculum_uses_eval_override_and_sequence_length():
    config = _experiment_config_from_args(
        SimpleNamespace(
            train_batch_size=None,
            max_input_tokens=512,
            max_output_tokens=256,
            n_prompts=4,
            n_generations_per_prompt=4,
            inference_gpu_memory_utilization=None,
        )
    )

    curriculum = create_math_curriculum(
        "alt-math500-test",
        config,
        seed=123,
        eval_n_examples=8,
    )

    lesson = curriculum.lessons["math_full"]
    assert curriculum.eval_n_examples == 8
    assert curriculum.max_seq_len == 768
    assert lesson.sampling_params.n_prompts == 4
    assert lesson.sampling_params.n_generations_per_prompt == 4


def test_controller_parser_accepts_smoke_override_flags():
    parser = build_parser()

    args = parser.parse_args(
        [
            "controller",
            "--run-id",
            "alt-tpu-001-smoke",
            "--shared-root",
            "gs://example/alternating-rl",
            "--image",
            "example/image:tag",
            "--tpu-name",
            "alt-v5p-probe-001",
            "--tpu-type",
            "v5p-8",
            "--zone",
            "us-east5-a",
            "--num-hosts",
            "1",
            "--steps-per-phase",
            "1",
            "--num-train-steps",
            "1",
            "--train-batch-size",
            "16",
            "--n-prompts",
            "4",
            "--n-generations-per-prompt",
            "4",
            "--eval-examples-per-lesson",
            "8",
            "--max-input-tokens",
            "512",
            "--max-output-tokens",
            "256",
            "--inference-gpu-memory-utilization",
            "0.95",
        ]
    )

    assert args.train_batch_size == 16
    assert args.n_prompts == 4
    assert args.n_generations_per_prompt == 4
    assert args.eval_examples_per_lesson == 8
    assert args.max_input_tokens == 512
    assert args.max_output_tokens == 256
    assert args.inference_gpu_memory_utilization == 0.95


def test_alternating_math500_uses_dedicated_wandb_project():
    config = _default_experiment_config()

    assert config.project_name == ALTERNATING_WANDB_PROJECT
