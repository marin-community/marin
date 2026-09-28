# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Compare frozen tasks across installed harnesses without TPU or model downloads.

Run in isolated task-only harness environments with the same dependency versions.
Deterministic synthetic responses isolate task/request/scoring behavior; this is
not a model-quality evaluation or a substitute for the real inference test.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.evaluator_utils import get_task_list
from lm_eval.tasks import TaskManager


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def encoded(value):
    return json.dumps(value, default=json_default, sort_keys=True, separators=(",", ":")).encode()


class ResponseProbe(LM):
    def __init__(self):
        super().__init__()
        self.requests_hash = hashlib.sha256()
        self.request_count = 0

    def loglikelihood(self, requests):
        responses = []
        for request in requests:
            payload = encoded([request.request_type, request.args])
            digest = hashlib.sha256(payload).digest()
            self.requests_hash.update(digest)
            self.request_count += 1
            responses.append((-1.0 - int.from_bytes(digest[:4], "big") / 2**30, digest[4] % 2 == 0))
        return responses

    def loglikelihood_rolling(self, requests):
        raise AssertionError("Unexpected rolling request in the frozen accuracy suite")

    def generate_until(self, requests):
        raise AssertionError("Unexpected generation request in the frozen accuracy suite")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    manager = TaskManager()
    tasks = []
    for spec in config["tasks"]:
        name, alias = spec["task"], spec["task_alias"]
        if args.upstream:
            from levanter.eval_harness_metrics import task_config_with_smooth_metrics  # noqa: PLC0415

            spec = task_config_with_smooth_metrics(spec)
        leaves = get_task_list(manager.load_config(dict(spec)))
        for leaf in leaves:
            leaf.task_name = (
                alias + leaf.task_name[len(name) :] if leaf.task_name.startswith(name) else alias + "_" + leaf.task_name
            )
            # Levanter appends the alias suffix to the leaf, not the family prefix.
            if alias.startswith(name) and leaf.task_name.startswith(alias):
                leaf.task_name = name + leaf.task_name[len(alias) :] + alias[len(name) :]
            leaf.task.config.task = leaf.task_name
        tasks.extend(leaves)
    assert {item.task_name for item in tasks} == set(config["expected_tasks"])
    report = {}
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002 - reproduce the harness's global RNG contract
    for item in sorted(tasks, key=lambda value: value.task_name):
        name, task = item.task_name, item.task
        task.set_fewshot_seed(seed=0)
        probe = ResponseProbe()
        result = evaluator.evaluate(
            lm=probe,
            task_dict={name: task},
            bootstrap_iters=0,
            log_samples=True,
            apply_chat_template=False,
        )
        metrics = result["results"][name]
        metric_names = {key.split(",")[0] for key in metrics if "," in key and "_stderr" not in key}
        sample_keys = {"doc_id", "doc", "target", "arguments", "resps", "filtered_resps"} | metric_names
        samples_hash = hashlib.sha256()
        for sample in result["samples"][name]:
            samples_hash.update(hashlib.sha256(encoded({key: sample[key] for key in sample_keys})).digest())
        report[name] = {
            "documents": result["n-samples"][name],
            "num_fewshot": result["configs"][name]["num_fewshot"],
            "metrics": metrics,
            "requests": probe.request_count,
            "requests_sha256": probe.requests_hash.hexdigest(),
            "samples_sha256": samples_hash.hexdigest(),
        }
        args.output.write_text(json.dumps(report, indent=2, default=json_default, sort_keys=True) + "\n")
        print(f"{name}: {report[name]['documents']}, {probe.request_count} requests", flush=True)
    assert sum(row["documents"]["effective"] for row in report.values()) == 44248


if __name__ == "__main__":
    main()
