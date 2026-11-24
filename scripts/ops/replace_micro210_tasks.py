# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build deterministic same-family replacements for gate-broken Micro210 tasks.

Three rows fail their oracle/NoP build gate in the verified in-region environment and
their task environments are the defect (the task's own oracle or its image-baked
dependencies, not the grading code):

- ``tb2-recovery/terminal-bench/mcmc-sampling-stan`` (its solve builds R packages from
  source and fails in the verified environment),
- ``ot-tblite-recovery/pdf-table-parsing`` (its verifier imports ``camelot`` before the
  solve's install lands),
- ``humanevalplus/HumanEval/130`` (oracle and NoP both zero in-region).

Each gets one same-family replacement chosen deterministically: seed 20261124, shuffled
family pool, first candidate not already in the policy whose gate grades oracle 1.0 and
NoP 0.0 on every repetition. Pools are pinned upstream sources; candidates marked
AI-judged are skipped. The swap and each removed task's reason land in the policy
provenance and the merged gate receipt.
"""

import argparse
import base64
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.evaluation.micro210_replacements import (
    BROKEN_TASKS,
    REPLACEMENT_SEED,
    CandidateRow,
    humanevalplus_candidate,
    ot_tblite_candidate,
    tb2_candidate,
)
from experiments.evaluation.micro210_gate import HARBOR_BENCHMARKS, row_probes

REPLACEMENT_ATTEMPTS_PER_FAMILY = 4
CANDIDATE_RETRIES = 2


def table_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rollout_template(candidate: CandidateRow, rows: list[dict]) -> str:
    benchmark = candidate.benchmark
    return next(row["task_rollout_spec"] for row in rows if row["benchmark"] == benchmark)


def row_files(spec_json: str) -> dict[str, bytes]:
    spec = json.loads(spec_json)
    return {
        resource["path"]: base64.b64decode(resource["source"]["content_base64"])
        for resource in spec["resources"]["verifier"]
    }


def grade_candidate(candidate: CandidateRow, table_path: Path, output: Path) -> dict:
    """Grade one candidate row with the spec gate's own grading path (all repetitions)."""
    table = pq.read_table(table_path)
    template = next(row for row in table.to_pylist() if row["benchmark"] == candidate.benchmark)
    records = []
    for repetition in range(template["repetitions"]):
        records.append(
            {
                **template,
                "task_id": candidate.task_id,
                "source_task_id": candidate.source_task_id,
                "task_spec": candidate.task_spec,
                "task_rollout_spec": candidate.task_rollout_spec,
                "selection_order": -1,
                "repetition_index": repetition,
            }
        )
    for record in records:
        rollout = json.loads(record["task_rollout_spec"])
        rollout["task"] = json.loads(record["task_spec"])
        record["task_rollout_spec"] = json.dumps(rollout, sort_keys=True)
    candidates_path = output / f"candidates-{candidate.benchmark}.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), candidates_path)
    from scripts.ops.verify_micro210_spec_gate import grade_family_a_row, grade_family_b_row

    graded = []
    for record in pq.read_table(candidates_path).to_pylist():
        if candidate.benchmark in HARBOR_BENCHMARKS:
            graded.append(grade_family_b_row(record, judge_environment=None, source_row=None))
        else:
            graded.append(grade_family_a_row(record, judge_environment=None, source_row=None))
    return {"task_id": candidate.task_id, "repetitions": graded}


def candidate_is_eligible(outcome: dict) -> bool:
    """Eligible when every repetition grades oracle 1.0 and NoP 0.0."""
    repetitions = outcome.get("repetitions") or []
    if not repetitions or any(repetition.get("failed") for repetition in repetitions):
        return False
    return all(repetition.get("oracle_reward") == 1.0 and repetition.get("nop_reward") == 0.0 for repetition in repetitions)


def family_candidates(benchmark: str, template_row: dict, in_policy: set[str], limit: int) -> list[CandidateRow]:
    """The deterministic candidate order for one family, first eligible wins."""
    ordered = _ordered_family_candidates(benchmark, in_policy)
    template = template_row
    candidates = []
    for source_task_id in ordered[:limit]:
        try:
            if benchmark == "humanevalplus":
                candidates.append(humanevalplus_candidate(source_task_id, template))
            elif benchmark == "ot-tblite-recovery":
                candidates.append(ot_tblite_candidate(source_task_id, template, _ot_tblite_package(source_task_id)))
            elif benchmark == "tb2-recovery":
                candidates.append(tb2_candidate(source_task_id, template, _tb2_package(source_task_id)))
            else:
                raise ValueError(f"No replacement pool for benchmark {benchmark!r}")
        except Exception as error:  # noqa: BLE001 - a candidate that cannot be built is skipped
            print(f"candidate {source_task_id} skipped: {type(error).__name__}: {error}", flush=True)
    return candidates


def _ordered_family_candidates(benchmark: str, in_policy: set[str]) -> list[str]:
    names = _family_pool(benchmark)
    return _deterministic_order(names, in_policy)


def _deterministic_order(names: list[str], in_policy: set[str]) -> list[str]:
    ordered = sorted(name for name in names if name not in in_policy)
    random.Random(REPLACEMENT_SEED).shuffle(ordered)
    return ordered


def _family_pool(benchmark: str) -> list[str]:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files

    if benchmark == "humanevalplus":
        dataset, revision = "evalplus/humanevalplus", "v0.1.10"
        path = hf_hub_download(dataset, repo_type="dataset", revision=revision, filename="data/test-00000-of-00001-5973903632b82d40.parquet")
        return sorted(row["task_id"] for row in pq.read_table(path).to_pylist())
    if benchmark == "ot-tblite-recovery":
        dataset, revision = "DCAgent/dev_set_v2", "377118ff3031c934"
        files = list_repo_files(dataset, repo_type="dataset", revision=revision)
        return sorted({name.split("/")[0] for name in files if "/" in name and not name.startswith(".")})
    if benchmark == "tb2-recovery":
        api = HfApi()
        listing = api.list_repo_files("harbor-framework/terminal-bench-1", repo_type="dataset") if False else None
        import urllib.request

        request = urllib.request.Request(
            "https://api.github.com/repos/harbor-framework/terminal-bench-1/contents/original-tasks",
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            entries = json.loads(response.read())
        return sorted(entry["name"] for entry in entries)
    raise ValueError(f"No replacement pool for benchmark {benchmark!r}")


def _ot_tblite_package(source_task_id: str) -> dict[str, bytes]:
    from huggingface_hub import hf_hub_download

    dataset, revision = "DCAgent/dev_set_v2", "377118ff3031c934"
    files = list_repo_files_local(dataset, revision)
    package = {}
    for name in files:
        if not name.startswith(f"{source_task_id}/"):
            continue
        relative = name[len(source_task_id) + 1 :]
        if relative.startswith("."):
            continue
        package[relative] = Path(hf_hub_download(dataset, repo_type="dataset", revision=revision, filename=name)).read_bytes()
    required = ("task.toml", "instruction.md", "solution/solve.sh", "tests/test.sh", "environment/Dockerfile")
    missing = [name for name in required if name not in package]
    if missing:
        raise ValueError(f"tblite package {source_task_id} is missing {missing}")
    return package


def list_repo_files_local(dataset: str, revision: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return list_repo_files(dataset, repo_type="dataset", revision=revision)


def _tb2_package(source_task_id: str) -> dict[str, bytes]:
    """Fetch one terminal-bench task's package from the pinned registry commit."""
    import urllib.request

    base = f"https://raw.githubusercontent.com/harbor-framework/terminal-bench-1/{_TB2_PIN}/{source_task_id}"
    package = {}
    listing = _github_json(f"https://api.github.com/repos/harbor-framework/terminal-bench-1/git/trees/{_TB2_PIN}?recursive=1")
    prefix = f"original-tasks/{source_task_id}/"
    paths = [
        entry["path"]
        for entry in listing.get("tree", [])
        if entry["path"].startswith(prefix) and entry["type"] == "blob"
    ]
    if not paths:
        raise ValueError(f"terminal-bench task {source_task_id} has no files at {_TB2_PIN}")
    for path in paths:
        relative = path[len(prefix) :]
        if relative.startswith("run-tests") or relative == "docker-compose.yaml":
            continue
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/harbor-framework/terminal-bench-1/{_TB2_PIN}/{path}", timeout=60) as response:
            package[relative] = response.read()
    for required in ("instruction.md", "tests", "solution.sh"):
        if not any(name.startswith(required) for name in package):
            raise ValueError(f"terminal-bench task {source_task_id} lacks {required} at {_TB2_PIN}")
    # Micro210 rows use task.toml; the upstream task.yaml converts mechanically.
    yaml_text = package.pop("task.yaml", b"").decode()
    package["instruction.md"] = package.get("instruction.md", _yaml_instruction(yaml_text).encode())
    package.setdefault("solution/solve.sh", package.pop("solution.sh", b""))
    package.setdefault("tests/test.sh", _TB_TEST_SH.encode())
    package["task.toml"] = _tb_task_toml(yaml_text, source_task_id).encode()
    return package


def _github_json(url: str) -> dict:
    import urllib.request

    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def _yaml_instruction(yaml_text: str) -> str:
    import yaml

    parsed = yaml.safe_load(yaml_text)
    return parsed["instruction"]


def _tb_task_toml(yaml_text: str, source_task_id: str) -> str:
    import yaml

    parsed = yaml.safe_load(yaml_text)
    metadata = parsed.get("metadata") or {}
    return (
        f'schema_version = "1.0"\n\n[task]\nname = "terminal-bench/{source_task_id}"\n\n'
        f'[metadata]\ndifficulty = "{metadata.get("difficulty", "hard")}"\n'
        f'category = "{metadata.get("category", "general")}"\n\n'
        f"[verifier]\ntimeout_sec = 1800.0\n\n[agent]\ntimeout_sec = 3600.0\n\n"
        f"[environment]\nbuild_timeout_sec = 1800.0\ncpus = 1\nmemory_mb = 2048\n"
        f"storage_mb = 10240\ngpus = 0\nallow_internet = true\n"
    )


_TB_TEST_SH = """#!/bin/bash
apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env
if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set."
    exit 1
fi
uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest /tests/test_outputs.py -rA
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""

_TB2_PIN = "2af0e447794d89f71f7b982d5a5fdb684679d421"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    table = pq.read_table(arguments.table)
    rows = table.to_pylist()
    in_policy_by_benchmark = {
        benchmark: {row["source_task_id"] for row in rows if row["benchmark"] == benchmark}
        for benchmark in BROKEN_TASKS
    }

    replacements = {}
    for broken_task_id, reason in BROKEN_TASKS.items():
        broken_row = next(row for row in rows if row["task_id"] == broken_task_id)
        benchmark = broken_row["benchmark"]
        candidates = family_candidates(benchmark, broken_row, in_policy_by_benchmark[benchmark], REPLACEMENT_ATTEMPTS_PER_FAMILY)
        chosen = None
        for candidate in candidates:
            print(f"gating candidate {candidate.task_id}", flush=True)
            outcome = None
            for attempt in range(CANDIDATE_RETRIES + 1):
                try:
                    outcome = grade_candidate(candidate, arguments.table, arguments.output)
                    break
                except Exception as error:  # noqa: BLE001 - infra failures retry
                    print(f"attempt {attempt} for {candidate.task_id} failed: {type(error).__name__}: {error}", flush=True)
            if outcome is not None and candidate_is_eligible(outcome):
                chosen = candidate
                break
        if chosen is None:
            raise SystemExit(f"No eligible {benchmark} replacement after {len(candidates)} candidates")
        replacements[broken_task_id] = {"reason": reason, "replacement": chosen}

    final_rows = []
    for row in rows:
        if row["task_id"] in BROKEN_TASKS:
            candidate = replacements[row["task_id"]]["replacement"]
            # The rollout spec embeds the task; rewrite it so runtime validation passes.
            rollout = json.loads(candidate.task_rollout_spec)
            rollout["task"] = json.loads(candidate.task_spec)
            final_rows.append(
                {
                    **row,
                    "task_id": candidate.task_id,
                    "source_task_id": candidate.source_task_id,
                    "task_spec": candidate.task_spec,
                    "task_rollout_spec": json.dumps(rollout, sort_keys=True),
                }
            )
        else:
            final_rows.append(row)
    output_table = arguments.output / "rollouts.parquet"
    pq.write_table(pa.Table.from_pylist(final_rows, schema=table.schema), output_table)
    swaps = {
        broken: {
            "reason": entry["reason"],
            "replacement_task_id": entry["replacement"].task_id,
            "replacement_source_task_id": entry["replacement"].source_task_id,
        }
        for broken, entry in replacements.items()
    }
    (arguments.output / "task-swaps.json").write_text(json.dumps(swaps, indent=2))
    print(f"final table: {output_table} digest {table_digest(output_table)}")
    for broken, entry in swaps.items():
        print(f"swap: {broken} -> {entry['replacement_task_id']} ({entry['reason']})")


if __name__ == "__main__":
    main()
