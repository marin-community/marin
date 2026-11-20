# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the FrontierScience pilots inside a time-limited CoreWeave Iris root job."""

import argparse
import os
import signal
import time
from dataclasses import replace
from pathlib import Path

from evalrecords.records import EvaluationOverrides, EvaluationSelection, JudgeIdentity
from iris.rpc import job_pb2
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.task_worker import JUDGE_API_KEY
from rigging.secrets import require_secret
from taskcompendium.importers.evalchemy.frontierscience import JUDGE_MODEL

from experiments.evaluation.launch import LaunchSpec, prepare_evaluation_batch, run_evaluation_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("qwen", "gemma", "gpt-oss-20b"), default="qwen")
    solver = parser.parse_args().solver
    directory = Path(__file__).parent
    protocol = "evalchemy"
    excluded = {50}
    if solver == "qwen":
        model = load_model_config(directory / "Qwen-Qwen3.6-35B-A3B.yaml")
    else:
        configs = directory.parent / "eval-campaign-09-25-unlabeled" / "model-configs"
        if solver == "gemma":
            model = load_model_config(configs / "google-gemma-4-26B-A4B-it.yaml")
            context = 262144
            template = {"enable_thinking": True}
            sampling = {"temperature": "1.0", "top_p": "0.95", "top_k": "64"}
        else:
            model = load_model_config(configs / "openai-gpt-oss-20b.yaml")
            context = 131072
            template = {"reasoning_effort": "high"}
            sampling = {"temperature": "1.0", "reasoning_effort": '"high"'}
        model = replace(
            model,
            serve=replace(model.serve, max_model_len=context),
            generation=replace(
                model.generation, max_gen_toks=context, chat_template_kwargs=template, extra_gen_kwargs=sampling
            ),
        )
        protocol = "remaining-context"
    remaining = int(os.environ["FRONTIERSCIENCE_DEADLINE"]) - int(time.time())
    if remaining <= 0:
        raise TimeoutError("FrontierScience pilot deadline has passed")
    signal.alarm(remaining)
    run = EvaluationOverrides(
        judge=JudgeIdentity(
            deployment="together", models={JUDGE_MODEL: f"provider-managed:{JUDGE_MODEL}"}, default_model=JUDGE_MODEL
        ),
        judge_endpoint="https://api.together.xyz/v1",
        judge_api_key=("env:TOGETHER_API_KEY",),
        max_tokens=model.generation.max_gen_toks,
    )
    research_rows = tuple(f"research/test.jsonl:{i}" for i in range(60) if i + 1 not in excluded)
    spec = LaunchSpec(
        model=model,
        evals=(
            EvaluationSelection(benchmark="frontierscience-olympiad", protocol=protocol, run=run),
            EvaluationSelection(
                benchmark="frontierscience-research",
                protocol=protocol,
                label=f"frontierscience-research-{len(research_rows)}-of-60",
                run=run.model_copy(update={"source_rows": research_rows}),
            ),
        ),
        platform=Platform.GPU,
        accelerator=f"H100x{model.resource_hint.gpu['H100']}",
        limit=None,
        records_prefix=None,
        submission_cluster="marin",
        federated_cluster="cw-us-east-02a",
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        version="dev",
        description=f"FrontierScience {solver} pilot, Olympiad 100/100 and Research {len(research_rows)}/60.",
    )
    batch = prepare_evaluation_batch(spec)
    os.environ[JUDGE_API_KEY] = require_secret("TOGETHER_API_KEY")
    for evaluation in batch.evaluations:
        print(evaluation.identity.run_id, evaluation.identity.output_dir, flush=True)
    run_evaluation_group(batch)


if __name__ == "__main__":
    main()
