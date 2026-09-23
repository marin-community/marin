# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade native outputs against private corpus references and report all 50 repos."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from experiments.post_training.bio_tasks.contract import grade_files


def report(bundle: Path, results: Path) -> dict:
    plan = json.loads((bundle / "plan.json").read_text())
    executions = {row["repository_index"]: row for row in json.loads((results / "results.json").read_text())}
    rows = []
    for repository in plan["repositories"]:
        index = repository["repository_index"]
        execution = executions.get(index, {})
        row = {"repository_index": index, "repository": repository["repository"], "cases": [], "status": "pending"}
        rows.append(row)
        attempts = {case["task_id"]: case for case in execution.get("cases", [])}
        hashes = set()
        for case in repository["cases"]:
            attempt = attempts.get(case["task_id"])
            if attempt is None or attempt["exit_code"] != 0:
                row["cases"].append({"task_id": case["task_id"], "status": "not_completed"})
                continue
            directory = results / attempt["output"]
            evidence = json.loads((directory / "execution.json").read_text())
            inputs = bundle / case["inputs"]
            expected_hashes = {
                str(p.relative_to(inputs)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(inputs.rglob("*"))
                if p.is_file()
            }
            if evidence["input_sha256"] != expected_hashes or evidence["execution"] != "completed":
                raise ValueError(f"Native execution/input mismatch for {index}/{case['task_id']}")
            commands = [json.loads(line) for line in (directory / "commands.jsonl").read_text().splitlines()]
            if not commands or any(command.get("error") or command["exit_code"] != 0 for command in commands):
                raise ValueError(f"Missing successful native commands for {index}/{case['task_id']}")
            verdict = grade_files(bundle / case["reference"], directory / "answer.json")
            row["cases"].append({"task_id": case["task_id"], "verdict": asdict(verdict)})
            hashes.add(json.dumps(expected_hashes, sort_keys=True))
        if len(row["cases"]) == len(hashes) == 3:
            if all(case.get("verdict", {}).get("reward") == 1 for case in row["cases"]):
                row["status"] = "passed"
            else:
                row["status"] = "scientific_mismatch"
        elif attempts:
            row["status"] = "incomplete"
    return {
        "expected_repositories": 50,
        "passed_repositories": sum(row["status"] == "passed" for row in rows),
        "repositories": rows,
        "corpus_manifest_sha256": plan["corpus_manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = report(args.bundle, args.results)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"passed_repositories": data["passed_repositories"], "expected_repositories": 50}))


if __name__ == "__main__":
    main()
