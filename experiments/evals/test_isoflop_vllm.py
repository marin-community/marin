# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test vLLM inference on an isoflop model.

Monkey-patches transformers.AutoConfig to fix the architectures mismatch
(isoflop checkpoints have model_type=qwen3 but architectures=["LlamaForCausalLM"]).
"""

import argparse
import sys
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path

from experiments.evals.exp_isoflop_hf_math500 import get_isoflop_hf_model
from experiments.isoflop_sweep import MARIN_SCALING_SUITES


@dataclass(frozen=True)
class TestIsoFlopVllmConfig:
    model_path: str
    output_path: str


def run_test_isoflop_vllm(config: TestIsoFlopVllmConfig):
    import transformers

    _original_from_pretrained = transformers.AutoConfig.from_pretrained

    def _patched_from_pretrained(*args, **kwargs):
        cfg = _original_from_pretrained(*args, **kwargs)
        if (
            getattr(cfg, "model_type", None) == "qwen3"
            and getattr(cfg, "architectures", None) == ["LlamaForCausalLM"]
        ):
            cfg.architectures = ["Qwen3ForCausalLM"]
        return cfg

    transformers.AutoConfig.from_pretrained = _patched_from_pretrained

    from vllm import LLM, SamplingParams

    print(f"Loading model from: {config.model_path}")
    llm = LLM(model=config.model_path, trust_remote_code=True, load_format="runai_streamer")

    prompts = [
        "What is 2 + 2?",
        "Explain the Pythagorean theorem.",
        "Write a haiku about math.",
    ]

    sampling_params = SamplingParams(max_tokens=256, temperature=0.7)
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        generated = output.outputs[0].text
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt}")
        print(f"Response: {generated}")


DEFAULT_TPU_TYPE = "v5p-8"

isoflop_steps, candidates = MARIN_SCALING_SUITES["nemotron"]
checkpoint_path = get_isoflop_hf_model(isoflop_steps[0], prefix="gs://marin-us-central1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vLLM inference on an isoflop model.")
    parser.add_argument(
        "--tpu-type",
        type=str,
        default=DEFAULT_TPU_TYPE,
        help=f"TPU type for ResourceConfig.with_tpu (default {DEFAULT_TPU_TYPE}).",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    test_step = ExecutorStep(
        name="rohith-debug/test_isoflop_vllm",
        fn=run_test_isoflop_vllm,
        config=TestIsoFlopVllmConfig(
            model_path=checkpoint_path,
            output_path=this_output_path(),
        ),
        resources=ResourceConfig.with_tpu(args.tpu_type),
        pip_dependency_groups=["vllm"],
    )
    executor_main(steps=[test_step], description="Test vLLM inference on an isoflop model.")
