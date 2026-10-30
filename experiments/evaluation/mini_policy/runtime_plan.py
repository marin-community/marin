# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export canonical task argv for the source-identity audit and timed subsets."""

import json
import os
from pathlib import Path

from marin.evaluation.evalchemy.client import build_command
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import _run_config_json
from marin.evaluation.model_config import load_model_config
from marin.inference.types import OpenAIEndpoint, RunningModel

from experiments.evaluation.evals import EvalchemyDefinition

CAMPAIGN = Path("experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled")


def main():
    model_path = CAMPAIGN / "model-configs/Qwen-Qwen3.6-35B-A3B.yaml"
    config = load_model_config(model_path)
    running = RunningModel(OpenAIEndpoint("http://not-used.invalid/v1", config.location), tokenizer=config.location)
    cohort = json.loads(Path("experiments/evaluation/mini_policy/cohort.json").read_text())
    inventory = json.loads(Path("experiments/evaluation/mini_policy/inventory.json").read_text())
    excluded = set(cohort["predictor_exclusions"])
    benchmarks = sorted({row["benchmark"] for row in inventory if row["benchmark"] not in excluded})
    plans = []
    for benchmark in benchmarks:
        path = CAMPAIGN / "evalchemy-configs" / f"{benchmark.lower()}.yaml"
        if not path.exists():
            continue  # Harbor task-name selections are prepared through its native DatasetConfig.
        run = EvalchemyDefinition(benchmark, path).config_for(load_evalchemy_config(path), config, None)
        document = json.loads(
            _run_config_json(
                running, run, "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/source-audit/no-generations"
            )
        )
        available = config.serve.max_model_len - (0 if document.get("prompt_budget_preflight", False) else 64)
        maximum = min(available, document["max_length"]) if document.get("max_length") is not None else available
        commands = [build_command(document, task, "/runtime/python", maximum)[1:] for task in document["tasks"]]
        plans.append(
            {
                "benchmark": benchmark,
                "config_path": str(path),
                "config_text": path.read_text(),
                "commands": commands,
                "chat_template_kwargs": document["chat_template_kwargs"],
            }
        )
    output = {"model_config": str(model_path), "model": config.location, "revision": config.revision, "plans": plans}
    path = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp")) / "canonical-source-plan.json"
    path.write_text(json.dumps(output, indent=2))
    print(path)


if __name__ == "__main__":
    main()
