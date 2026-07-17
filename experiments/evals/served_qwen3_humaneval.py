# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run HumanEval against Qwen3 through brokered vLLM serving.

By default, the evaluator runs through an isolated ``uv run`` environment in a
non-preemptible Iris CPU parent, with inference in TPU child jobs connected
through Marin's inference broker. ``--local`` instead runs the broker, proxy,
and worker in the current process.

\b
Examples:
  uv run python experiments/evals/served_qwen3_humaneval.py
  uv run python experiments/evals/served_qwen3_humaneval.py --priority production
  uv run python experiments/evals/served_qwen3_humaneval.py --local
"""

from experiments.evals.served_lm_eval import ServedLmEvalBenchmark, served_lm_eval_command

main = served_lm_eval_command(
    __doc__,
    ServedLmEvalBenchmark(
        tasks=("humaneval",),
        output_path="/tmp/served-qwen3-humaneval",
        job_name="served-qwen3-humaneval",
        confirm_run_unsafe_code=True,
        parent_env_vars={"HF_ALLOW_CODE_EVAL": "1"},
    ),
)

if __name__ == "__main__":
    main()
