# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Submit a SWE-ZERO MVP job to the Iris cluster.

This script:
  1. Starts a vLLM-tpu server on a TPU worker
  2. Runs the rollout generation against it
  3. Saves results to GCS

Usage:
    # Step 1: Prototype tool calling (single TPU)
    uv run python experiments/swe_zero/submit_iris_job.py \
        --step 1 --model google/gemma-4-E2B-it --tpu_type v6e-4

    # Step 3-6: Generate rollouts
    uv run python experiments/swe_zero/submit_iris_job.py \
        --step 3 --model google/gemma-4-E2B-it --tpu_type v6e-4 \
        --output_dir gs://marin-us-central2/experiments/swe_zero_mvp
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_ray_command(
    step: int,
    model: str,
    tpu_type: str,
    output_dir: str,
    tp_size: int,
) -> list[str]:
    """Build the Ray job submission command for a given step."""

    if step == 1:
        # Step 1: Just run the prototype script
        script = "experiments/swe_zero/prototype_tool_calling.py"
        script_args = [
            "--api_base",
            "http://localhost:8000/v1",
            "--model",
            model,
        ]
    else:
        # Steps 3-6: Run the MVP pipeline
        script = "experiments/swe_zero/run_swe_zero_mvp.py"
        script_args = [
            "--api_base",
            "http://localhost:8000/v1",
            "--model",
            model,
            "--step",
            str(step),
            "--output_dir",
            output_dir,
        ]

    # The job needs to:
    # 1. Start vLLM server in background
    # 2. Wait for it to be ready
    # 3. Run the script
    # We wrap this in a shell script
    vllm_cmd = (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {model} "
        f"--tensor-parallel-size {tp_size} "
        f"--max-model-len 16384 "
        f"--port 8000 "
        f"--trust-remote-code "
        f"--enable-auto-tool-choice "
        f"--tool-call-parser hermes "
        f"--chat-template auto "
        f"--dtype bfloat16 "
        f"--max-num-seqs 32 "
        f"--gpu-memory-utilization 0.95"
    )

    # Create a wrapper script that starts vLLM and runs the experiment
    wrapper_script = f"""
set -e
echo "Starting vLLM server for {model}..."
{vllm_cmd} &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server process died!"
        exit 1
    fi
    sleep 5
done

echo "Running SWE-ZERO step {step}..."
python {script} {" ".join(script_args)}
EXIT_CODE=$?

echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

exit $EXIT_CODE
"""

    return ["bash", "-c", wrapper_script]


def main():
    parser = argparse.ArgumentParser(description="Submit SWE-ZERO job to Iris cluster")
    parser.add_argument("--step", type=int, required=True, choices=[1, 3, 4, 5, 6])
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--tpu_type", default="v6e-4", help="TPU type (v6e-4, v5p-8, etc.)")
    parser.add_argument("--output_dir", default="gs://marin-us-central2/experiments/swe_zero_mvp")
    parser.add_argument("--dry_run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    # Determine tensor parallel size from TPU type
    tp_map = {
        "v6e-4": 4,
        "v6e-8": 8,
        "v5p-8": 8,
        "v5litepod-4": 4,
        "v5litepod-8": 8,
    }
    tp_size = tp_map.get(args.tpu_type, 4)

    logger.info("Submitting SWE-ZERO step %d job", args.step)
    logger.info("  Model: %s", args.model)
    logger.info("  TPU type: %s (TP=%d)", args.tpu_type, tp_size)
    logger.info("  Output: %s", args.output_dir)

    cmd = build_ray_command(
        step=args.step,
        model=args.model,
        tpu_type=args.tpu_type,
        output_dir=args.output_dir,
        tp_size=tp_size,
    )

    if args.dry_run:
        print("Would run:", " ".join(cmd))
        return

    # Submit via Iris CLI
    # For now, print the command for manual submission
    print("\n=== Submit this job to the Iris cluster ===")
    print(f"TPU type: {args.tpu_type}")
    print(f"Model: {args.model}")
    print()
    print("Option 1 — Run directly on a TPU VM:")
    print(f"  bash experiments/swe_zero/serve_vllm_tpu.sh {args.model} {tp_size}")
    print("  # In another terminal:")
    if args.step == 1:
        print(
            f"  python experiments/swe_zero/prototype_tool_calling.py"
            f" --api_base http://localhost:8000/v1 --model {args.model}"
        )
    else:
        print(
            f"  python experiments/swe_zero/run_swe_zero_mvp.py"
            f" --api_base http://localhost:8000/v1 --model {args.model}"
            f" --step {args.step} --output_dir {args.output_dir}"
        )
    print()
    print("Option 2 — Submit via Ray (from a connected terminal):")
    print(f"  uv run lib/marin/src/marin/run/ray_run.py --no_wait -- {cmd[0]} {cmd[1]} '{cmd[2]}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
