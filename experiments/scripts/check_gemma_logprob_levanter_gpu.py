"""GPU entry point for Gemma log-prob parity using Levanter + Marin executor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Configure platform before importing JAX/Levanter.
os.environ.setdefault("JAX_PLATFORMS", "cuda")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from marin.execution.executor import ExecutorStep, executor_main
from marin.resources import GpuConfig

from _levanter_logprob_runner import GemmaLevanterLogProbConfig, run_gemma_levanter_logprob
from gemma_logprob_utils import DEFAULT_PROMPT

STEP_NAME = "checks/gemma/logprob/levanter-gpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Gemma log probabilities on GPU via Marin executor + Levanter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default="google/gemma-2-9b", help="HuggingFace model identifier.")
    parser.add_argument("--revision", default="main", help="Optional HuggingFace revision or commit.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text to evaluate.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="Computation dtype for Levanter forward pass.",
    )
    parser.add_argument("--reference", help="Optional JSON reference to compare against.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-5,
        help="Maximum allowed absolute diff when --reference is provided.",
    )
    parser.add_argument(
        "--output-name",
        default="logprob.json",
        help="Filename (within the executor output directory) used to store results.",
    )
    parser.add_argument(
        "--override-output-path",
        dest="override_output_path",
        help="Optional absolute or cloud path to force the executor output directory.",
    )
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs requested for the step.")
    parser.add_argument(
        "--accelerator-type",
        default=None,
        help="Optional Ray accelerator type (e.g. A100-80G) to pin the GPU selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = GemmaLevanterLogProbConfig(
        backend="levanter-gpu",
        model_id=args.model_id,
        revision=args.revision,
        prompt=args.prompt,
        dtype=args.dtype,
        reference_path=args.reference,
        tolerance=args.tolerance,
        output_filename=args.output_name,
        resource_config=GpuConfig(gpu_count=args.gpu_count, accelerator_type=args.accelerator_type),
    )

    step = ExecutorStep(
        name=STEP_NAME,
        fn=run_gemma_levanter_logprob,
        config=config,
    )

    if args.override_output_path:
        step = step.with_output_path(args.override_output_path)

    executor_main(
        steps=[step],
        description="Gemma log-prob consistency check on GPU (Levanter).",
    )


if __name__ == "__main__":
    main()
