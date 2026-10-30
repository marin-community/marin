# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render full canonical sources without issuing candidate or judge requests."""

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from eval.contracts.preflight import prepare_requested_tasks
from eval.eval import DEFAULT_LM_EVAL_INCLUDE_DIR, resolve_task_routes, setup_custom_parser
from eval.execution.driver import source_plans
from eval.execution.run_spec import RunSpec
from eval.limits import resolve_evaluation_limits
from eval.source_selection import source_key
from eval.task import TaskManager
from lm_eval import utils
from lm_eval.tasks import TaskManager as PretrainTaskManager
from rigging.filesystem.buckets import filesystem_for
from transformers import AutoTokenizer

from experiments.evaluation.mini_policy.identities import document_identity

DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/source-index-20261030-v2"


@dataclass
class TokenizerModel:
    tokenizer: object
    model: str
    _evalchemy_chat_template_kwargs: dict

    def tok_encode(self, text):
        return self.tokenizer.encode(text)


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(plan["model"], revision=plan["revision"], trust_remote_code=True)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    summary = []
    for benchmark in plan["plans"]:
        rows = []
        for command in benchmark["commands"]:
            args = setup_custom_parser().parse_args(command)
            limits = resolve_evaluation_limits(args)
            tasks = args.tasks.split(",")
            custom = TaskManager(
                annotator_model=args.annotator_model,
                max_length=limits.max_length,
                max_tokens=limits.max_tokens,
                limit=args.limit,
                num_fewshot=args.num_fewshot,
                debug=args.debug,
                seed=args.seed,
                task_list=tasks,
                system_instruction=args.system_instruction,
                num_samples=args.num_samples,
                pass_at_k=args.pass_at_k,
                dataset_cache=args.dataset_cache,
            )
            native = PretrainTaskManager(args.verbosity, include_path=[DEFAULT_LM_EVAL_INCLUDE_DIR])
            routes = resolve_task_routes(tasks, custom, native)
            prepare_requested_tasks(tasks, routes, custom, native)
            # Source discovery never grades; avoid resolving judge credentials.
            run = replace(RunSpec.from_args(args), predict_only=True)
            random.seed(run.seeds[0])
            np.random.seed(run.seeds[1])  # noqa: NPY002 -- Match Evalchemy's global source RNG.
            utils.set_torch_seed(run.seeds[2])
            # Uses the same source-side tokenizer interface as execution. This object has no HTTP transport.
            model = TokenizerModel(tokenizer, plan["model"], benchmark["chat_template_kwargs"])
            for source_plan in source_plans(tasks, routes, custom, native, run, model):
                for source in source_plan.sources:
                    for spec in source:
                        identity = document_identity(benchmark["benchmark"], spec.doc, spec.source_id)
                        payload = json.dumps(spec.requests, default=lambda value: value.__dict__, sort_keys=True)
                        rows.append(
                            {
                                "task_id": identity,
                                "source_key": source_key(spec),
                                "namespace": spec.namespace,
                                "source_id": spec.source_id,
                                "trial": spec.trial,
                                "ordinal": spec.ordinal,
                                "request_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                                "request_count": len(spec.requests),
                            }
                        )
        document = {
            "benchmark": benchmark["benchmark"],
            "parent": benchmark,
            "rows": rows,
            "tokenizer_model": plan["model"],
            "tokenizer_revision": plan["revision"],
            "discovery_only": True,
        }
        content = json.dumps(document, indent=2, default=str).encode()
        name = benchmark["benchmark"] + ".json"
        (output / name).write_bytes(content)
        fs, path = filesystem_for(DEST + "/" + name)
        fs.pipe_file(path, content)
        summary.append(
            {"benchmark": benchmark["benchmark"], "specs": len(rows), "tasks": len({row["task_id"] for row in rows})}
        )
        print(json.dumps(summary[-1]), flush=True)
    content = json.dumps(summary, indent=2).encode()
    (output / "summary.json").write_bytes(content)
    fs, path = filesystem_for(DEST + "/summary.json")
    fs.pipe_file(path, content)


if __name__ == "__main__":
    main()
