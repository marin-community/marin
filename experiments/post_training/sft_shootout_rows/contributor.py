# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the LR-sweep pilot's contributor traces into trace rows and a pool table the arm builder reads.

The contributor-data experiment trains the shootout arms on the task #202 pilot set instead of GLM generations:
allenai/tmax-sft (Qwen3.6-27B, success split) for TMax and open-athena/Kimi-2.5-swesmith (Kimi-2.5) for SWE-smith.
The task set is the pilot's, as is: sweep_data.select runs unchanged on the pinned sources, so the leak filters
(reads_git_history for Kimi, references_hidden_asset for TMax), the SWE-bench Verified repo exclusion, the 32k
context rule, the 2,100-task Kimi sample and the 5% held-out split are the pilot's own. The run checks that the
selected trajectories and held-out ids equal the pilot artifact's.

Each kept trajectory becomes one trace row in build.py's schema: every assistant turn is the teacher's (author glm),
one conversation, passed, no problems, and a `harness` field, because these sources ran other harnesses than the v7
generation: Kimi SWE-smith ran Terminus-2 (JSON keystrokes, task_complete), and Qwen TMax ran mini-swe-agent with a
`bash` tool call per turn, which keeps a rejected reply in history and answers "Format error: ..." (HARNESS_NOTES).
A TMax trajectory's sample is its solution index (task__solN); a Kimi task has one trajectory (sample 0).

Outputs:
  <traces>/<family>/shard-00/rebuilt.jsonl and _SUCCESS: training trajectories plus each held-out task's pilot
      validation trajectory.
  <pool>/tasks.jsonl and manifest.json: one row per task (family, held-out flag, repo, decontamination record) and the
      source pins, selection counts and the decontamination gap. Pool v5 (the v7 pool) has a decon record for some of
      these tasks; the rest have only the pilot's checks, which the manifest states.

Usage:
    python -m experiments.post_training.sft_shootout_rows.contributor --traces <prefix> --pool <prefix>
"""

import argparse
import collections
import hashlib
import json
import re
import tempfile
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.recovery import MINI_SWE_FORMAT_ERROR, TERMINUS_2
from experiments.post_training.sft_shootout_rows.traces import TEACHER
from experiments.post_training.sweep_data.prepare_agentic import (
    ALLKIMI,
    KIMI,
    TMAX,
    TMAX_ENVS,
    audit_hashes,
    load_sources,
    load_tokenizer,
)
from experiments.post_training.sweep_data.render import kimi_messages, tmax_messages
from experiments.post_training.sweep_data.select import select

PILOT = "s3://marin-us-east-02a/marin/teacher-traces/mopd2/task202-sweep-prepared-20261016d"
POOL_V5 = "s3://marin-us-east-02a/marin/experiments/sft-shootout/pool/tasks.jsonl"
HARNESS = {"swesmith": TERMINUS_2, "tmax": MINI_SWE_FORMAT_ERROR}
HARNESS_NOTES = {
    TERMINUS_2: (
        "Kimi-2.5 SWE-smith: Terminus-2 JSON responses (analysis, plan, commands with keystrokes, task_complete); "
        "the trial ends on task_complete"
    ),
    MINI_SWE_FORMAT_ERROR: (
        "Qwen3.6-27B TMax: mini-swe-agent with one `bash` tool call per turn; a reply without one stays in history and "
        "the harness answers with a fixed 'Format error' user message (409 such replies in the source split, one "
        "text); the trial ends on `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`"
    ),
}
SOLUTION = re.compile(r"__sol([0-9]+)$")


def trace_row(family: str, row: dict, record: dict) -> dict:
    """One build.py trace row for a selected pilot trajectory; every assistant turn is the teacher's."""
    if family == "tmax":
        messages, tools = tmax_messages(row)
        match = SOLUTION.search(record["trajectory"])
        if match is None:
            raise ValueError(f"TMax trajectory without a solution index: {record['trajectory']}")
        sample = int(match[1])
    else:
        messages, tools, sample = kimi_messages(row), None, 0
    for message in messages:
        if message["role"] == "assistant":
            message["author"] = TEACHER
    return {
        "trial_id": record["trajectory"],
        "trial_name": record["trajectory"],
        "task_id": record["task"],
        "family": family,
        "harness": HARNESS[family],
        "relay": False,
        "passed": True,
        "problems": [],
        "exception": None,
        "sample": sample,
        "retry": 0,
        "conversation": 0,
        "n_conversations": 1,
        "messages": messages,
        "tools": tools,
    }


def check_against_pilot(selection: dict) -> dict:
    """The selected trajectories and held-out ids must be the pilot artifact's; return what was compared."""
    pilot = StoragePath(PILOT)
    index = [json.loads(line) for line in (pilot / "train-index.jsonl").read_text().splitlines()]
    expected = {(row["family"], row["trajectory"]) for row in index if row["family"] in ("tmax", "swesmith")}
    found = {(family, r["trajectory"]) for family, records in selection["train"].items() for r in records}
    if found != expected:
        raise ValueError(f"Selection differs from the pilot's training trajectories: {len(found ^ expected)} differ")
    heldout = json.loads((pilot / "heldout-ids.json").read_text())
    if {family: sorted(ids) for family, ids in heldout.items()} != selection["heldout"]:
        raise ValueError("Held-out ids differ from the pilot's")
    return {"pilot": PILOT, "training_trajectories": len(expected), "heldout_tasks": sum(map(len, heldout.values()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, help="output trace prefix (<family>/shard-00/rebuilt.jsonl)")
    parser.add_argument("--pool", required=True, help="output pool prefix (tasks.jsonl, manifest.json)")
    args = parser.parse_args()
    for prefix in (args.traces, args.pool):
        if StoragePath(prefix).exists():
            raise ValueError(f"Use a fresh output prefix: {prefix}")
    with tempfile.TemporaryDirectory() as temporary:
        (Path(temporary) / "tokenizer").mkdir()
        tokenizer = load_tokenizer(Path(temporary) / "tokenizer", audit_hashes())
        sources = load_sources()
        selection = select(tokenizer, sources.tmax_rows, sources.kimi_rows, sources.allkimi, sources.tmax_hidden)
    pilot_check = check_against_pilot(selection)

    source_rows = {"tmax": {}, "swesmith": {}}
    for row in sources.tmax_rows:
        source_rows["tmax"][row["metadata"]["trial_name"]] = row
    for row in sources.kimi_rows:
        source_rows["swesmith"][row["trial_name"]] = row
    pool_v5 = {row["task_id"]: row for row in map(json.loads, StoragePath(POOL_V5).read_text().splitlines())}
    allkimi_tasks = {trial.rsplit("__", 1)[0] for trial in sources.allkimi}

    tasks, counts = {}, collections.Counter()
    for family in ("tmax", "swesmith"):
        records = selection["train"][family] + selection["validation"][family]
        lines = []
        for record in sorted(records, key=lambda r: r["trajectory"]):
            lines.append(json.dumps(trace_row(family, source_rows[family][record["trajectory"]], record)) + "\n")
            task = record["task"]
            if task in tasks:
                continue
            v5 = pool_v5.get(task)
            tasks[task] = {
                "task_id": task,
                "family": family,
                "heldout": task in selection["heldout"][family],
                "repo": record.get("repo"),
                "in_allkimi": family == "swesmith" and task in allkimi_tasks,
                "decon": (
                    {"source": "pool_v5", "decon": v5.get("decon"), "leak_risk": v5.get("leak_risk")}
                    if v5 is not None and v5["family"] == family
                    else {"source": "pilot_only"}
                ),
            }
            counts[(family, tasks[task]["decon"]["source"], tasks[task]["heldout"])] += 1
        shard = StoragePath(args.traces) / family / "shard-00"
        data = "".join(lines).encode()
        (shard / "rebuilt.jsonl").write_bytes(data)
        if (shard / "rebuilt.jsonl").read_bytes() != data:
            raise ValueError(f"{shard}: re-read differs")
        (shard / "_SUCCESS").write_text(json.dumps({"rows": len(lines), "sha256": hashlib.sha256(data).hexdigest()}))
        print(f"TRACES {family} rows={len(lines)} sha256={hashlib.sha256(data).hexdigest()}", flush=True)

    tasks_bytes = "".join(json.dumps(row) + "\n" for row in sorted(tasks.values(), key=lambda r: r["task_id"])).encode()
    decon = {
        family: {
            "tasks": sum(n for (f, _, _), n in counts.items() if f == family),
            "pool_v5_record": sum(n for (f, s, _), n in counts.items() if f == family and s == "pool_v5"),
            "pilot_only": sum(n for (f, s, _), n in counts.items() if f == family and s == "pilot_only"),
        }
        for family in ("tmax", "swesmith")
    }
    manifest = {
        "name": "t202-contributor-pool",
        "file_sha256": {"tasks.jsonl": hashlib.sha256(tasks_bytes).hexdigest()},
        "sources": {
            "tmax": {"repo": TMAX[0], "revision": TMAX[1], "config": TMAX[2]},
            "tmax_environments": {"repo": TMAX_ENVS[0], "revision": TMAX_ENVS[1], "file": TMAX_ENVS[2]},
            "swesmith": {"repo": KIMI[0], "revision": KIMI[1]},
            "allkimi": {"repo": ALLKIMI[0], "revision": ALLKIMI[1], "file": ALLKIMI[2]},
        },
        "harness": {family: {"name": HARNESS[family], "note": HARNESS_NOTES[HARNESS[family]]} for family in HARNESS},
        "selection": {
            "rule": "sweep_data.select on the pinned sources, unchanged (the task #202 pilot set as is)",
            "pilot_check": pilot_check,
            "dropped": selection["dropped"],
            "eligible_tasks": selection["eligible_tasks"],
            "swesmith_allocation": selection["swesmith_allocation"],
            "allkimi_overlap": selection["allkimi_overlap"],
        },
        "notes": {
            "decontamination": {
                "by_family": decon,
                "gap": (
                    "tasks marked decon.source == 'pilot_only' have no pool v5 decontamination record; their only "
                    "checks are the pilot's: SWE-bench Verified repos excluded (0 of the Kimi repos are Verified "
                    "repos), Kimi passes reading code from git history dropped, TMax trajectories naming a hidden "
                    "verifier asset dropped"
                ),
            },
            "allkimi": (
                "Kimi training tasks with a trial in allkimi's SFT list (in_allkimi) are kept, as in the pilot; v0 "
                "carries a graft of that SFT, and no held-out task is an allkimi task"
            ),
        },
    }
    pool = StoragePath(args.pool)
    (pool / "tasks.jsonl").write_bytes(tasks_bytes)
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode()
    (pool / "manifest.json").write_bytes(manifest_bytes)
    print("POOL_MANIFEST_SHA256 " + hashlib.sha256(manifest_bytes).hexdigest(), flush=True)
    print("CONTRIBUTOR_CONVERTED " + json.dumps({"decon": decon, "pilot_check": pilot_check}), flush=True)


if __name__ == "__main__":
    main()
