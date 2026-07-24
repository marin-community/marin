# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolved serving contracts for the Grug agentic model profile."""

from dataclasses import replace

from experiments.evaluation.hardware import Platform
from experiments.evaluation.launch import LaunchSpec, plan_runs
from experiments.evals.evalchemy.serve_and_eval import _shared_inference_config


def test_grug_agentic_profile_resolves_validated_h100_serving_contract():
    plan = plan_runs(
        LaunchSpec(
            model="grug-agentic-s3-step1903",
            evals=("aime-harbor",),
            platform=Platform.GPU,
            accelerator=None,
            limit=None,
            records_prefix=None,
            cluster="marin",
        )
    )[0]

    assert plan.accel.label == "H100x8"
    assert plan.accel.target_cluster == "cw-rno2a"
    assert plan.serve.max_model_len == 65536
    assert plan.serve.max_num_batched_tokens == 7168
    assert plan.serve.max_num_seqs == 32
    assert plan.model.agentic_n_concurrent == 256
    assert plan.suite.harbor is not None
    assert plan.suite.harbor.n_concurrent == 256
    assert plan.serve.tensor_parallel_size == 1
    assert plan.serve.serve_cpu == 48.0
    assert plan.serve.serve_memory == "1024g"
    assert plan.serve.serve_disk == "512g"
    assert plan.serve.vllm_extra_args == (
        "--data-parallel-size",
        "8",
        "--enable-expert-parallel",
        "--model-loader-extra-config",
        '{"distributed":true}',
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    )


def test_grug_agentic_profile_materializes_one_sequence_limit_in_the_vllm_engine():
    plan = plan_runs(
        LaunchSpec(
            model="grug-agentic-s3-step1903",
            evals=("aime-harbor",),
            platform=Platform.GPU,
            accelerator=None,
            limit=None,
            records_prefix=None,
            cluster="marin",
        )
    )[0]

    inference = _shared_inference_config(
        plan.model.location,
        plan.model.tokenizer or plan.model.location,
        replace(plan.serve, auto_overrides=False),
    )
    engine_args = inference.engine.extra_args

    assert engine_args.count("--max-num-seqs") == 1
    assert engine_args[engine_args.index("--max-num-seqs") + 1] == "32"
