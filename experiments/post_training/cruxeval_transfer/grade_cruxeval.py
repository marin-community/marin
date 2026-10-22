# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade saved CruxEval responses inside a credential-free, network-disabled container."""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def test_code(problem: dict, direction: str, content: str) -> str:
    """Apply the pinned Evalchemy answer extraction and assertion construction."""
    if "[ANSWER]" in content and "[/ANSWER]" in content:
        content = content[content.index("[ANSWER]") + len("[ANSWER]") : content.index("[/ANSWER]")].strip()
    function = re.search(r"f\(.*?\)", content)
    if function is not None:
        if direction == "input":
            content = function.group(0)
        else:
            content = content.replace(function.group(0), "").replace("=", "").replace("assert", "").strip()
    return f"{problem['code']}\nassert {problem['output']} == {content}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--execution-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("execution", args.execution_file)
    assert spec is not None and spec.loader is not None
    execution = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = execution
    spec.loader.exec_module(execution)
    requests = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    raw = [json.loads(line) for line in args.responses.read_text().splitlines() if line.strip()]
    responses = {row["custom_id"]: row for row in raw}
    if len(responses) != len(raw) or set(responses) != {row["custom_id"] for row in requests}:
        raise ValueError("Response IDs do not match the request set exactly")
    audit = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = []
        for request in requests:
            response = responses[request["custom_id"]]
            row = {"custom_id": request["custom_id"], "task": request["task"], "source_id": request["source_id"]}
            if response.get("error") or response.get("response", {}).get("status_code") != 200:
                audit.append(row | {"correct": False, "result": "request_failed"})
                continue
            choice = response["response"]["body"]["choices"][0]
            message = choice["message"]
            row["finish_reason"] = choice.get("finish_reason")
            row["has_reasoning"] = bool((message.get("reasoning") or message.get("reasoning_content") or "").strip())
            code = test_code(request["source"], request["task"], message.get("content") or "")
            future = pool.submit(execution.check_correctness, row["custom_id"], {"test_code": code}, "python", 10.0)
            pending.append((row, future))
        for row, future in pending:
            result = future.result()
            audit.append(row | {"correct": result["passed"], "result": result["result"]})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audit))
    totals = Counter(row["task"] for row in audit)
    correct = Counter(row["task"] for row in audit if row["correct"])
    summary = {
        task: {"correct": correct[task], "total": total, "pass_at_1": correct[task] / total}
        for task, total in sorted(totals.items())
    }
    (args.output / "scores.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
