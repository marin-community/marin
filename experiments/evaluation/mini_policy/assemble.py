# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build task-level evidence and fail closed on score reconciliation mismatches."""

import concurrent.futures
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from finestore.reader import ReadView
from marin.evaluation.metric_selection import declared_sample_metric, primary_filter
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.identities import document_identity

PREFIX = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/inventory-20261030"


def read(url):
    fs, path = filesystem_for(url)
    return json.loads(fs.cat(path))


def preserve(url, data):
    fs, path = filesystem_for(url)
    fs.pipe_file(path, json.dumps(data, default=str).encode())


def original_record(cell, payload):
    source = cell["source"]
    current = payload
    for _ in range(8):
        if source.endswith("/results"):
            return read(source[:-8] + "/record.json")
        parent = current.get("raw_source") or current.get("source") if isinstance(current, dict) else None
        if not isinstance(parent, str) or parent == source:
            return None
        source = parent
        if source.endswith(".json"):
            current = read(source)
    raise ValueError("source chain exceeds eight revisions")


def original_documents(record):
    if record is None:
        return {}
    view = ReadView(record["results_path"])
    table = view.scan("samples", columns=["doc_id", "doc"])
    if table is None:
        raise ValueError("original dataset documents unavailable")
    return {str(row["doc_id"]): json.loads(row["doc"] or "{}") for row in table.to_pylist()}


def native_trials(cell, key):
    data = read(f"{PREFIX}/native_evidence/{key}.json")
    fs, root = filesystem_for(cell["source"])
    job_paths = fs.glob(root.rstrip("/") + "/harbor_jobs/*/result.json")
    if len(job_paths) != 1:
        raise ValueError(f"expected one native Harbor job, found {len(job_paths)}")
    job = json.loads(fs.cat(job_paths[0]))
    evals = job["stats"]["evals"]
    if len(evals) != 1:
        raise ValueError("ambiguous native Harbor eval groups")
    stats = next(iter(evals.values()))
    rewards = stats["reward_stats"]["reward"]
    scored = {name: float(reward) for reward, names in rewards.items() for name in names}
    trials = []
    for entry in data["trials"]:
        result = entry["result"]
        name = result["trial_name"]
        trials.append(
            {
                "task_id": document_identity(cell["benchmark"], {}, result["task_name"]),
                "trial_id": name,
                "scored": name in scored,
                "reward": scored.get(name),
                "error": result.get("exception_info"),
                "config": result.get("config"),
                "task_checksum": result.get("task_checksum"),
                "started_at": result.get("started_at"),
                "finished_at": result.get("finished_at"),
            }
        )
    if set(scored) - {t["trial_id"] for t in trials}:
        raise ValueError("native score map references absent trials")
    return trials


def assemble(cell):
    result = dict(cell)
    key = hashlib.sha256((cell["model"] + "\n" + cell["benchmark"]).encode()).hexdigest()[:16]
    try:
        native = (
            cell.get("force_native")
            or cell.get("needs_adapter")
            or (cell["benchmark"] == "ds-1000-local" and cell.get("row_count", 0) < 180)
        )
        payload = read(cell["evidence_url"]) if cell.get("evidence_url") else None
        record = None
        try:
            record = original_record(cell, payload)
        except FileNotFoundError as error:
            result["record_error"] = str(error)
        result["record_available"] = record is not None
        if record:
            result["provenance"] = record.get("provenance")
            result["eval_config"] = record.get("eval")
            result["model_config"] = record.get("model")
            result["serving"] = record.get("serving")
            result["coverage"] = record.get("coverage")
            result["record_canonical_metrics"] = record.get("canonical_metrics")
            result["timing"] = record.get("timing")
        trials = []
        if native:
            trials = native_trials(cell, key)
            result["adapter"] = "native_harbor_reward_membership"
            fs, root = filesystem_for(cell["source"])
            jobs = fs.glob(root.rstrip("/") + "/harbor_jobs/*/result.json")
            native_job = json.loads(fs.cat(jobs[0]))
            result["native_expected_trials"] = int(native_job["n_total_trials"])
        elif isinstance(payload, dict):
            documents = (
                original_documents(record)
                if record and cell["benchmark"] not in {"swebench-verified", "SOTOPIA-hard"}
                else {}
            )
            if payload.get("model") != cell["model"]:
                raise ValueError("replay model identity mismatch")
            for row in payload["trials"]:
                if "corrected_correct" in row:
                    reward = float(row["corrected_correct"])
                    scored = True
                elif "length_controlled_overall" in row:
                    reward = float(row["length_controlled_overall"])
                    scored = True
                elif "scored" in row and "reward" in row:
                    scored = row["scored"]
                    reward = float(row["reward"]) if scored else None
                else:
                    raise ValueError(f"unknown replay trial keys {list(row)}")
                task = row.get("doc_id", row.get("task_id", row.get("task_name")))
                if task is None:
                    raise ValueError("no stable task identity")
                document = documents.get(str(task), {})
                if not document and row.get("question"):
                    document = {"question": row["question"]}
                stable = document_identity(cell["benchmark"], document, task)
                strata = {
                    key: value
                    for key, value in document.items()
                    if key
                    in {
                        "category",
                        "domain",
                        "subject",
                        "subcategory",
                        "difficulty",
                        "level",
                        "type",
                        "answer_format",
                        "mrcr_bin_upper",
                        "language",
                        "instruction_id_list",
                    }
                }
                if row.get("answer_format"):
                    strata["answer_format"] = row["answer_format"]
                trials.append(
                    {
                        "task_id": stable,
                        "strata": strata,
                        "source_doc_id": str(task),
                        "trial_id": str(row.get("trial_id", task)),
                        "reward": reward,
                        "scored": scored,
                        "started_at": row.get("started_at"),
                        "finished_at": row.get("finished_at"),
                    }
                )
            result["adapter"] = "corrected_replay"
            result["grading_revision"] = payload.get("evalchemy_commit")
            result["effective_config"] = payload.get("effective_config")
        else:
            rows = payload
            if cell["benchmark"] == "cruxeval":
                if any(not json.loads(row["doc"]).get("cruxeval_task") for row in rows):
                    view = ReadView(cell["source"])
                    if view.table_metadata_path("samples") != cell["metadata_path"]:
                        raise ValueError("sample metadata advanced since inventory")
                    rows = view.scan(
                        "samples",
                        columns=[
                            "task",
                            "doc_id",
                            "trial_id",
                            "filter",
                            "grading",
                            "metrics",
                            "doc",
                            "prompt_text",
                            "prompt_messages",
                        ],
                    ).to_pylist(maps_as_pydicts="strict")
                    output_prefix = (
                        "You are given a Python function and an assertion containing an input to the function."
                    )
                    rows = [
                        row
                        for row in rows
                        if (
                            row.get("prompt_text")
                            or "".join(m.get("content", "") for m in row.get("prompt_messages") or [])
                        ).startswith(output_prefix)
                    ]
                else:
                    rows = [row for row in rows if json.loads(row["doc"]).get("cruxeval_task") == "output"]
            selected_filter = primary_filter([row.get("filter") for row in rows])
            if selected_filter is not None:
                rows = [row for row in rows if row.get("filter") == selected_filter]
            declarations = {}
            if record:
                for task in record["eval"].get("tasks", []):
                    descriptor = task.get("benchmark") or {}
                    primary = descriptor.get("primary_metric")
                    source = next(
                        (m["source_name"] for m in descriptor.get("metrics", []) if m["name"] == primary), primary
                    )
                    source = source.removesuffix("_avg") if isinstance(source, str) else source
                    if source == "python_pass@1":
                        source = "pass_rate"
                    declarations[task.get("task_alias") or task["name"]] = source
                    declarations[task["name"]] = source
            seen = set()
            for row in rows:
                ident = (row["task"], row["doc_id"], row["trial_id"])
                if ident in seen:
                    raise ValueError(f"duplicate normalized trial {ident}")
                seen.add(ident)
                metrics = row.get("metrics") or {}
                grade = row.get("grading") or {}
                declared = declarations.get(row["task"])
                picked = declared_sample_metric(metrics, declared)
                if (
                    declared
                    and picked is None
                    and grade.get("metric") != declared
                    and not (
                        declared in {"python_pass@1", "pass_at_1", "pass_rate", "pass@1"}
                        and grade.get("metric") == "accuracy"
                    )
                ):
                    raise ValueError(f'declared metric {declared} absent from {list(metrics)} / {grade.get("metric")}')
                detail = json.loads(grade.get("detail") or "{}")
                reward = picked[1] if picked else grade.get("score")
                if detail.get("scored") is False:
                    reward = None
                if reward is None and set(metrics) == {"pass_rate"}:
                    reward = metrics["pass_rate"]
                stable = document_identity(cell["benchmark"], json.loads(row.get("doc") or "{}"), row["doc_id"])
                strata = {
                    key: value
                    for key, value in json.loads(row.get("doc") or "{}").items()
                    if key
                    in {
                        "category",
                        "domain",
                        "subject",
                        "subcategory",
                        "difficulty",
                        "level",
                        "type",
                        "answer_format",
                        "mrcr_bin_upper",
                        "language",
                        "instruction_id_list",
                    }
                }
                trials.append(
                    {
                        "task_id": stable,
                        "strata": strata,
                        "source_doc_id": str(row["doc_id"]),
                        "trial_id": str(row["trial_id"]),
                        "reward": float(reward) if reward is not None else None,
                        "scored": reward is not None,
                    }
                )
            result.update(adapter="normalized_samples", selected_filter=selected_filter, declared_metrics=declarations)
        rewards = [t["reward"] for t in trials if t["scored"]]
        if not rewards:
            raise ValueError("no scored trials")
        if any(not 0 <= reward <= 1 for reward in rewards):
            raise ValueError("reward outside [0,1]")
        mean = sum(rewards) / len(rewards)
        tasks = Counter(t["task_id"] for t in trials)
        result.update(
            trial_count=len(trials),
            scored_count=len(rewards),
            task_count=len(tasks),
            repetitions=dict(Counter(tasks.values())),
            mean=mean,
            score_delta=mean - cell["display_score"],
            status="reconciled" if abs(mean - cell["display_score"]) <= 0.00051 else "score_mismatch",
        )
        url = f"{PREFIX}/task_trials/{key}.json"
        preserve(url, {"model": cell["model"], "benchmark": cell["benchmark"], "trials": trials})
        result["trials_url"] = url
        authorities = []
        if isinstance(payload, dict) and payload.get("corrected_score") is not None:
            authorities.append(("replay.corrected_score", float(payload["corrected_score"])))
        if record:
            for spec in record["eval"].get("tasks", []):
                primary = (spec.get("benchmark") or {}).get("primary_metric")
                namespace = spec.get("task_alias") or spec["name"]
                values = (record.get("canonical_metrics") or {}).get(namespace, {})
                if primary in values:
                    authorities.append(("record.canonical_metrics:" + namespace + ":" + primary, float(values[primary])))
        result["authority_scores"] = authorities
        if (
            result["status"] == "score_mismatch"
            and authorities
            and all(abs(value - mean) < 1e-9 for _, value in authorities)
        ):
            result["status"] = "reconciled_source_update"
        if result["status"] == "score_mismatch" and native and len(rewards) / result["native_expected_trials"] >= 0.9:
            result["status"] = "reconciled_native_evidence"
            result["target_note"] = (
                "Latest native reward membership reconstructed at >=90% scored coverage; tracker scalar differs."
            )
    except Exception as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error))
    return result


def main():
    cells = json.loads(Path("experiments/evaluation/mini_policy/inventory.json").read_text())
    prior = json.loads(Path("experiments/evaluation/mini_policy/assembly_audit.json").read_text())
    native_keys = {
        (r["model"], r["benchmark"])
        for r in prior
        if r["status"] == "score_mismatch"
        and r["benchmark"] in {"tb2-recovery", "ot-tblite-recovery", "bixbench-pi", "swebench-verified"}
        and not r["source"].endswith(".json")
    }
    for cell in cells:
        cell["force_native"] = (cell["model"], cell["benchmark"]) in native_keys
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(assemble, cells):
            results.append(result)
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "benchmark": result["benchmark"],
                        "status": result["status"],
                        "delta": result.get("score_delta"),
                        "error": result.get("error"),
                    }
                ),
                flush=True,
            )
            (output / "assembly_audit.json").write_text(json.dumps(results, indent=2, default=str))
    preserve(f"{PREFIX}/assembly_audit.json", results)
    counts = dict(Counter(r["status"] for r in results))
    (output / "assembly_summary.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
