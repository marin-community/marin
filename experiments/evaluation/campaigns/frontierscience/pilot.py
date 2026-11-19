# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the 159-row pilot inside a time-limited CoreWeave Iris root job."""

import os
import signal
import time
from pathlib import Path

from evalrecords.records import EvaluationOverrides, EvaluationSelection, JudgeIdentity
from iris.rpc import job_pb2
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.task_worker import JUDGE_API_KEY
from rigging.secrets import require_secret

from experiments.evaluation.launch import LaunchSpec, prepare_evaluation_batch, run_evaluation_group


def main() -> None:
    remaining = int(os.environ["FRONTIERSCIENCE_DEADLINE"]) - int(time.time())
    if remaining <= 0:
        raise TimeoutError("FrontierScience pilot deadline has passed")
    signal.alarm(remaining)
    judge_model = "openai/gpt-oss-120b"
    run = EvaluationOverrides(
        judge=JudgeIdentity(
            deployment="together", models={judge_model: f"provider-managed:{judge_model}"}, default_model=judge_model
        ),
        judge_endpoint="https://api.together.xyz/v1",
        judge_api_key=("env:TOGETHER_API_KEY",),
        max_tokens=81920,
    )
    research_rows = tuple(f"research/test.jsonl:{i}" for i in range(60) if i != 58)
    spec = LaunchSpec(
        model=load_model_config(Path(__file__).with_name("Qwen-Qwen3.6-35B-A3B.yaml")),
        evals=(
            EvaluationSelection(benchmark="frontierscience-olympiad", run=run),
            EvaluationSelection(
                benchmark="frontierscience-research",
                label="frontierscience-research-59-of-60",
                run=run.model_copy(update={"source_rows": research_rows}),
            ),
        ),
        platform=Platform.GPU,
        accelerator="H100x4",
        limit=None,
        records_prefix=None,
        submission_cluster="marin",
        federated_cluster="cw-us-east-02a",
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        version="dev",
        description="FrontierScience pilot: Olympiad 100/100 and Research 59/60, omitting one-based row 59.",
    )
    batch = prepare_evaluation_batch(spec)
    os.environ[JUDGE_API_KEY] = require_secret("TOGETHER_API_KEY")
    for evaluation in batch.evaluations:
        print(evaluation.identity.run_id, evaluation.identity.output_dir, flush=True)
    run_evaluation_group(batch)


if __name__ == "__main__":
    main()
