# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate broker-served Qwen3 on HumanEval and save metrics and samples.

\b
Examples:
  uv run python experiments/evals/served_qwen3_humaneval.py
  uv run python experiments/evals/served_qwen3_humaneval.py --priority production
  uv run python experiments/evals/served_qwen3_humaneval.py --launcher local
"""

from experiments.evals.evals import HUMANEVAL_ENV
from experiments.evals.served_lm_eval import ServedLmEvalBenchmark, served_lm_eval_command

main = served_lm_eval_command(
    __doc__,
    ServedLmEvalBenchmark(
        tasks=("humaneval",),
        output_path="/tmp/served-qwen3-humaneval",
        job_name="served-qwen3-humaneval",
        confirm_run_unsafe_code=True,
        parent_env_vars=HUMANEVAL_ENV,
    ),
)

if __name__ == "__main__":
    main()
