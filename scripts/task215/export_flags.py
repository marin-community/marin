# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task #471: the SWE-smith git-history flag list (issue 10212) from audit.py's <dataset>-traces.jsonl files.

Usage: export_flags.py <out dir> <publish dir>

Writes <publish dir>/data/<dataset>.parquet: one row per trace whose agent commands match GIT_HISTORY_READ, with ids
and flags only (no trace text), and prints the issue's count table.
"""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SOURCES = {
    "kimi": ("open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
             "1b87a9cf78b897a2355105d47afd44d818cad25d", "trial_name"),
    "calibforge": ("laion/calibforge-relay-traces", "66feab4d24bfc37af4c38d95a6a26187b5648308",
                   "trial (subset terminus2_relay_sft, source kimi_swesmith)"),
    "glm47": ("DCAgent2/GLM-4.7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k",
              "e209a88db18950c3ce4e72a45a6088561d99d1bf", "trial_name"),
    "agenttrove": ("open-thoughts/AgentTrove", "b395a4307a2bc9950a90dc899438f149e115fc60",
                   "trial_name (rows with original_source == swesmith)"),
}
SCHEMA = pa.schema([
    ("dataset", pa.string()), ("revision", pa.string()), ("trace_id", pa.string()), ("task_id", pa.string()),
    ("stream", pa.string()), ("outcome", pa.string()), ("repo", pa.string()), ("clone_task", pa.bool_()),
    ("history_commands", pa.list_(pa.string())), ("fix_testable", pa.bool_()), ("saw_fix_line", pa.bool_()),
    ("printed_source_diff", pa.bool_()),
])


def main():
    out, publish = Path(sys.argv[1]), Path(sys.argv[2])
    (publish / "data").mkdir(parents=True, exist_ok=True)
    for name, (repo_id, revision, _) in SOURCES.items():
        rows = [json.loads(line) for line in open(out / f"{name}-traces.jsonl")]
        flagged = []
        for r in rows:
            hits = r["hits"]
            if "git_history" not in hits:
                continue
            testable = "git_history_fix_untestable" not in hits
            flagged.append({
                "dataset": repo_id, "revision": revision, "trace_id": r["trial"], "task_id": r["task"],
                "stream": r["stream"], "outcome": r["outcome"], "repo": r["repo"], "clone_task": r["clone_task"],
                "history_commands": r["git_history_commands"], "fix_testable": testable,
                "saw_fix_line": bool(hits.get("git_history_saw_fix")) if testable else None,
                "printed_source_diff": bool(hits.get("git_history_printed_source_diff")),
            })
        pq.write_table(pa.Table.from_pylist(flagged, schema=SCHEMA), publish / "data" / f"{name}.parquet")
        passes = [r for r in rows if r["outcome"] == "pass"]
        fp = [f for f in flagged if f["outcome"] == "pass"]
        pool = fp if passes else flagged
        testable = [f for f in pool if f["fix_testable"]]
        completed = [r for r in rows if not r["outcome"].startswith("error")]
        print(f"| {name} | traces {len(rows)} | passes {len(passes)} | flagged {len(flagged)} | flagged passes {len(fp)}"
              f" | completed {len(completed)} | flagged completed {sum(1 for f in flagged if not f['outcome'].startswith('error'))}"
              f" | saw fix {sum(1 for f in testable if f['saw_fix_line'])} / {len(testable)} (of {'passes' if passes else 'all flagged'})"
              f" | clone-task rows {sum(r['clone_task'] for r in rows)}, flagged {sum(f['clone_task'] for f in flagged)} |")


if __name__ == "__main__":
    main()
