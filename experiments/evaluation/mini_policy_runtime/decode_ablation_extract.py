# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn one Evalchemy IFBench output store into a cell summary and a trace file.

The trace file follows trace-sets/ifbench-11-authors-100-prompts.jsonl: the same prompt keys,
responses cut at 4,000 characters, think blocks stripped. ``passed`` is strict prompt-level
accuracy for the sample.
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

MAX_CHARS = 4000


def table_rows(store: Path, name: str) -> list[dict]:
    files = sorted(glob.glob(str(store / "data" / name / "*" / "*" / "*.parquet")))
    rows = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def answer_text(output: str) -> str | None:
    if "<think>" in output and "</think>" not in output:
        return None
    if "</think>" in output:
        return output.rsplit("</think>", 1)[1].lstrip()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--author-model", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--gen-kwargs", required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--out-traces", required=True)
    parser.add_argument("--out-summary", required=True)
    args = parser.parse_args()

    store = Path(args.store)
    keys = {int(line) for line in Path(args.keys).read_text().split()}
    # FineStore keeps compacted copies next to writer files; keep one sample per prompt key.
    samples = list({json.loads(r["doc"])["key"]: r for r in table_rows(store, "samples")}.values())
    operations = table_rows(store, "evalchemy_operations")
    results = [json.loads(r["document"]) for r in table_rows(store, "evalchemy_results")]
    assert len({json.dumps(r["results"], sort_keys=True) for r in results}) == 1, len(results)

    finish_by_text: dict[str, tuple[str, int]] = {}
    for op in operations:
        if json.loads(op["operation"]).get("phase") != "generate":
            continue
        response = json.loads(op["payload"])["output"]["response"]
        usage = response.get("usage") or {}
        finish_by_text[response.get("content") or ""] = (response.get("finish_reason"), usage.get("completion_tokens"))

    finish = Counter()
    completion_tokens = []
    traces = []
    strict = []
    unclosed_think: list[int] = []
    empty_keys: list[int] = []
    for sample in samples:
        doc = json.loads(sample["doc"])
        metrics = dict(sample["metrics"])
        passed = metrics["strict_prompt_accuracy"] == 1.0
        strict.append(passed)
        output = sample["output"] or ""
        reason, tokens = finish_by_text.get(output, (None, None))
        finish[reason] += 1
        if tokens is not None:
            completion_tokens.append(tokens)
        if doc["key"] not in keys:
            continue
        text = answer_text(output)
        if text is None:
            # An unclosed literal "<think>" in a non-reasoning model's text: keep the raw output.
            unclosed_think.append(doc["key"])
            text = output
        if not text.strip():
            empty_keys.append(doc["key"])
            continue
        traces.append(
            {
                "trace_id": f"{doc['key']}-{args.author}-{args.cell}",
                "prompt_key": doc["key"],
                "author": f"{args.author}@{args.cell}",
                "author_model": args.author_model,
                "cell": args.cell,
                "passed": passed,
                "truncated": len(text) > MAX_CHARS,
                "full_chars": len(text),
                "finish_reason": reason,
                "completion_tokens": tokens,
                "prompt": doc["prompt"],
                "response": text[:MAX_CHARS],
                "full_response": text,
            }
        )
    traces.sort(key=lambda t: t["prompt_key"])
    with open(args.out_traces, "w") as out:
        for trace in traces:
            out.write(json.dumps(trace) + "\n")
    completion_tokens.sort()
    summary = {
        "author": args.author,
        "author_model": args.author_model,
        "cell": args.cell,
        "gen_kwargs": args.gen_kwargs,
        "n_samples": len(samples),
        "strict_prompt_accuracy_store": results[0]["results"]["IFBench"]["strict_prompt_accuracy"],
        "strict_prompt_accuracy_samples": sum(strict) / len(strict),
        "task_outcome": {k: v for k, v in results[0]["task_outcomes"]["IFBench"].items() if k != "metrics"},
        "finish_reasons": dict(finish),
        "completion_tokens_median": completion_tokens[len(completion_tokens) // 2] if completion_tokens else None,
        "completion_tokens_total": sum(completion_tokens),
        "n_traces": len(traces),
        "trace_keys_empty_response": sorted(empty_keys),
        "trace_keys_unclosed_think": sorted(unclosed_think),
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
