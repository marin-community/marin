# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quick single-prompt test for SWE-bench Multilingual evaluation using vLLM on TPU.

Runs the entrypoint on the head node, then dispatches vLLM inference to a TPU worker
via ray.remote. This follows the standard Marin pattern for TPU-based vLLM inference.

Usage:
    # Submit to us-central1 cluster (runs on v5p-8 TPU worker)
    RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py \\
        --env_vars HF_TOKEN ${HF_TOKEN} \\
        --cluster us-central1 \\
        -- python scripts/test_single_prompt_sweb_eval.py \\
            --model AlienKevin/swe-smith-rs-base-qwen2.5-coder-7b-instruct-teacher-glm-4.6

    # With custom instance and parameters
    RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py \\
        --env_vars HF_TOKEN ${HF_TOKEN} \\
        --cluster us-central1 \\
        -- python scripts/test_single_prompt_sweb_eval.py \\
            --model AlienKevin/swe-smith-rs-base-qwen3-8b-teacher-glm-4.6 \\
            --problem-statement "Fix the bug in the parser" \\
            --max-tokens 2000
"""

import argparse
import json
import logging

import os
from contextlib import contextmanager

import ray

from fray.cluster import ResourceConfig
from fray.v1.cluster.ray.resources import as_remote_kwargs
from fray.v1.cluster.ray.tpu import run_on_pod_ray

logger = logging.getLogger(__name__)


@contextmanager
def _remove_tpu_lockfile():
    """Remove the TPU lockfile on exit so the next Ray task can use the TPU."""
    try:
        yield
    finally:
        try:
            os.unlink("/tmp/libtpu_lockfile")
        except (FileNotFoundError, PermissionError):
            pass


SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. \
Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>"""

INSTANCE_TEMPLATE = """\
We're currently solving the following issue within our repository. Here's the issue text:
ISSUE:
{problem_statement}

INSTRUCTIONS:
Now, you're going to solve this issue on your own. Your terminal session has started and \
you're in the repository's root directory. You can use any bash commands or the special interface \
to help you. Edit all the files you need to and run any checks or tests that you can to verify \
your fix. When you're satisfied with all the changes, you can submit your changes."""

TOOL_DEF = json.dumps(
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
                "required": ["command"],
            },
        },
    }
)


def build_qwen3_prompt(system_msg: str, user_msg: str, tool_def: str) -> str:
    """Build a Qwen3-style chat prompt with tool definitions."""
    return (
        f"<|im_start|>system\n"
        f"{system_msg}\n\n"
        f"# Tools\n\n"
        f"You may call one or more functions to assist with the user query.\n\n"
        f"You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tool_def}\n</tools>\n\n"
        f"For each function call, return a json object with function name and arguments "
        f"within <tool_call></tool_call> XML tags:\n"
        f"<tool_call>\n"
        f'{{"name": <function-name>, "arguments": <args-json-object>}}\n'
        f"</tool_call><|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_inference(prompt: str, model: str, max_tokens: int, temperature: float, max_model_len: int):
    """Run vLLM inference inside a TPU-allocated task."""
    from vllm import LLM, SamplingParams

    logger.info(f"Initializing vLLM with model: {model}")
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.info(f"Running inference (prompt length: {len(prompt)} chars)...")
    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0]

    return output.outputs[0].text, output.outputs[0].finish_reason


def main():
    parser = argparse.ArgumentParser(description="Single-prompt SWE-bench eval via vLLM on TPU")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--instance-id",
        default="sharkdp__bat-562",
        help="SWE-bench instance ID (for display only, default: sharkdp__bat-562)",
    )
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max tokens for completion")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max model context length")
    parser.add_argument("--tpu-type", default="v5p-8", help="TPU type for inference (default: v5p-8)")
    parser.add_argument(
        "--problem-statement",
        default=None,
        help="Problem statement text (required; or loads from HF dataset if omitted)",
    )
    args = parser.parse_args()

    # Get problem statement
    if args.problem_statement:
        problem_statement = args.problem_statement
    else:
        from datasets import load_dataset

        ds = load_dataset("SWE-bench/SWE-bench_Multilingual", split="test")
        instances = [x for x in ds if x["instance_id"] == args.instance_id]
        if not instances:
            print(f"Instance {args.instance_id} not found in SWE-bench_Multilingual")
            return
        problem_statement = instances[0]["problem_statement"]

    # Build prompt
    system_msg = SYSTEM_TEMPLATE.strip()
    user_msg = INSTANCE_TEMPLATE.format(problem_statement=problem_statement)
    prompt = build_qwen3_prompt(system_msg, user_msg, TOOL_DEF)

    print(f"Instance: {args.instance_id}")
    print(f"Model: {args.model}")
    print(f"TPU: {args.tpu_type}")
    print(f"Prompt length: {len(prompt)} chars")

    # Setup env vars for TPU worker
    env = {}
    env["HF_DATASETS_TRUST_REMOTE_CODE"] = os.environ.get("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    env["EQX_ON_ERROR"] = "nan"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["SKIP_JAX_PRECOMPILE"] = "1"

    # Configure TPU resources
    inference_resources = ResourceConfig.with_tpu(args.tpu_type)
    inference_kwargs = dict(max_calls=1, **as_remote_kwargs(inference_resources, env_vars=env))

    # Capture args for the remote task
    model = args.model
    max_tokens = args.max_tokens
    temperature = args.temperature
    max_model_len = args.max_model_len

    @ray.remote(**inference_kwargs)
    def inference_task():
        with _remove_tpu_lockfile():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            return run_inference(prompt, model, max_tokens, temperature, max_model_len)

    # Dispatch to TPU worker via run_on_pod_ray
    task = run_on_pod_ray.remote(
        inference_task,
        args.tpu_type,
        num_slices=1,
        max_retries_failure=3,
        max_retries_preemption=100,
    )

    # run_on_pod_ray returns a list of results (one per slice)
    results = ray.get(task)
    text, finish = results[0]

    print(f"\nfinish_reason: {finish}")
    print(f"output length: {len(text)} chars")
    print("\n=== RAW OUTPUT ===")
    print(text)
    print("\n=== REPR ===")
    print(repr(text))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
