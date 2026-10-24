# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the existing dtype A/B endpoints with the standard Evalchemy runner."""

import argparse
import json
from dataclasses import asdict

import requests
from marin.evaluation.evalchemy.runner import EvalchemyRunConfig, EvalchemyRuntimeConfig, run_evalchemy
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.inference.types import InferenceBackendState, OpenAIEndpoint, RunningModel
from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import require_secret

MODEL = "open-athena/Marin-Snowball-67B-A2B-v0.1.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    args = parser.parse_args()
    base = require_secret("IDENTITY_ENDPOINT").rstrip("/")
    model = RunningModel(OpenAIEndpoint(base_url=base + "/v1", model=MODEL), tokenizer=MODEL)

    def backend_state() -> InferenceBackendState:
        response = requests.get(base + "/health", timeout=30)
        response.raise_for_status()
        return InferenceBackendState.READY

    for name, alias, thinking, count in (("gsm8k", "gsm8k_0shot", False, 1319), ("MATH500", "math500", True, 500)):
        output = f"{args.output.rstrip('/')}/{alias}"
        fs, path = filesystem_for(output + "/done.json")
        config = EvalchemyRunConfig(
            name=alias,
            tasks=(EvalTaskConfig(name=name, num_fewshot=0, task_alias=alias, generation=True),),
            apply_chat_template=True,
            max_gen_toks=8192,
            max_length=32768,
            num_concurrent=8,
            seed=0,
            extra_gen_kwargs={"temperature": "0"},
            chat_template_kwargs={"enable_thinking": thinking},
            runtime=EvalchemyRuntimeConfig(extras=("serve-eval", "math500")),
        )
        provenance = {
            "dtype": args.dtype,
            "revision": "59852d6f07261722bbec1edb6ab148317156b58e",
            "config": asdict(config),
        }
        # Normalize tuples to JSON arrays for equality after a restart.
        provenance = json.loads(json.dumps(provenance))
        if fs.exists(path):
            assert json.loads(fs.cat_file(path))["provenance"] == provenance
            print(f"Skipping completed {alias}", flush=True)
            continue
        outcome = run_evalchemy(model, config, output, env_vars={}, backend_state=backend_state)
        coverage = list(outcome.coverage.values())
        assert len(coverage) == 1 and coverage[0].n_scored == count, f"Incomplete coverage: {coverage}"
        payload = json.dumps({"provenance": provenance, "outcome": asdict(outcome)}, indent=2).encode()
        fs.pipe_file(path, payload)
        assert fs.cat_file(path) == payload
        print(f"Completed {alias}: {outcome.metrics}", flush=True)


if __name__ == "__main__":
    main()
