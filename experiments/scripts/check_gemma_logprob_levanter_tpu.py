"""TPU entry point for Gemma log-prob parity using Levanter + Marin executor."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Configure platform before importing JAX/Levanter.
os.environ.setdefault("JAX_PLATFORMS", "tpu")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from marin.execution.executor import ExecutorStep, executor_main

from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL
from _levanter_logprob_runner import GemmaLevanterLogProbConfig, run_gemma_levanter_logprob
from gemma_logprob_utils import DEFAULT_PROMPT

STEP_NAME = "checks/gemma/logprob/levanter-tpu"
MODEL_ID = "google/gemma-2-27b"
MODEL_REVISION: str | None = "main"
PROMPT = DEFAULT_PROMPT
DTYPE = "bfloat16"
REFERENCE_PATH: str | None = None
TOLERANCE = 5e-5
OUTPUT_NAME = "logprob.json"


def main() -> None:
    os.environ["HF_HOME"] = "/opt/gcsfuse_mount"
    step = ExecutorStep(
        name=STEP_NAME,
        fn=run_gemma_levanter_logprob,
        config=GemmaLevanterLogProbConfig(
            backend="levanter-tpu",
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            prompt=PROMPT,
            dtype=DTYPE,
            reference_path=REFERENCE_PATH,
            tolerance=TOLERANCE,
            output_filename=OUTPUT_NAME,
            resource_config=SINGLE_TPU_V5p_8_FULL,
        ),
    )

    executor_main(
        steps=[step],
        description="Gemma log-prob consistency check on TPU (Levanter).",
    )


if __name__ == "__main__":
    main()
