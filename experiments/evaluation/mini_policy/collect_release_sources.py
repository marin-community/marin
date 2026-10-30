# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture selected source specifications for a portable dataset release."""

import json
import os
import random
from dataclasses import asdict, replace
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

from experiments.evaluation.mini_policy.experiment import read_bytes
from experiments.evaluation.mini_policy.identities import document_identity
from experiments.evaluation.mini_policy.source_index import TokenizerModel

STUDY = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/study-20261030-v6-21x21"
DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/dataset-release-20261030"


def publish(name, value):
    content = json.dumps(value, indent=2, ensure_ascii=False).encode()
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / name).write_bytes(content)
    fs, path = filesystem_for(DEST + "/" + name)
    fs.pipe_file(path, content)


def main():
    settings = json.loads(read_bytes(STUDY + "/settings.json"))
    model_index = settings["models"].index("Qwen/Qwen3.6-35B-A3B")
    fold = next(i for i, split in enumerate(settings["splits"]) if model_index in split["test"])
    matrix = json.loads(read_bytes(STUDY + "/inputs/tasks.json"))
    selection = {}
    for budget in (210, 840):
        record = json.loads(read_bytes(STUDY + f"/fold-{fold}-random-0-{budget}.json"))
        selected = next(
            row["selected"] for row in record["probe_manifests"] if row["model"] == settings["models"][model_index]
        )
        selection[str(budget)] = [
            dict(matrix["tasks"][column], matrix_column=column, selection_order=i) for i, column in enumerate(selected)
        ]
    assert selection["210"] == selection["840"][:210]
    publish(
        "selection.json",
        {
            "study": STUDY,
            "seed": 0,
            "source_seed": settings["seed"],
            "fold": fold,
            "selection": selection,
            "settings": settings,
        },
    )
    wanted = {}
    for row in selection["840"]:
        wanted.setdefault(row["benchmark"], set()).add(row["task_id"])
    plan = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "experiments/evaluation/mini_policy/runtime-configs/release-source-plan.json"
        ).read_text()
    )
    publish("canonical-source-plan.json", plan)
    tokenizer = AutoTokenizer.from_pretrained(plan["model"], revision=plan["revision"], trust_remote_code=True)
    for benchmark in plan["plans"]:
        name = benchmark["benchmark"]
        if name not in wanted:
            continue
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
            run = replace(RunSpec.from_args(args), predict_only=True)
            random.seed(run.seeds[0])
            np.random.seed(run.seeds[1])  # noqa: NPY002
            utils.set_torch_seed(run.seeds[2])
            model = TokenizerModel(tokenizer, plan["model"], benchmark["chat_template_kwargs"])
            for source_plan in source_plans(tasks, routes, custom, native, run, model):
                for source in source_plan.sources:
                    for spec in source:
                        identity = document_identity(name, spec.doc, spec.source_id)
                        if identity not in wanted[name] or (name == "cruxeval" and spec.namespace != "output"):
                            continue
                        rows.append({"task_id": identity, "source_key": source_key(spec), "spec": asdict(spec)})
        found = {row["task_id"] for row in rows}
        if found != wanted[name]:
            raise ValueError(f"Missing selected identities for {name}: {wanted[name] - found}")
        publish("evalchemy-" + name + ".json", {"benchmark": name, "parent": benchmark, "rows": rows})
        print(json.dumps({"benchmark": name, "unique_tasks": len(found), "specs": len(rows)}), flush=True)
    publish("evalchemy-complete.json", {"complete": True})


if __name__ == "__main__":
    main()
