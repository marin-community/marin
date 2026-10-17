"""Task #215: audit public SWE-smith trace sets for the #10212 / #10223 leak paths. Read-only, CPU only.

Usage: audit.py <data dir> <out dir> [dataset ...]
Datasets: kimi, calibforge, glm47, agenttrove. <data dir> holds the local snapshots (see README in trace-audit notes).
Writes <out dir>/<dataset>-traces.jsonl (one line per trace: ids, outcome, repo, which rules fired, first matching
command per rule) and prints the tables.

Rules (definitions shared with the SFT shootout):
- git_history: vizcacha's GIT_HISTORY_READ (task/211, sweep_data/select.py), applied to every Terminus-2 keystroke
  string in assistant turns.
- library_copy: vizcacha's LIBRARY_LOCATION + the task repo's import names (task/211, sft_shootout_rows/build.py).
  Counted twice: in agent commands (the leak path this audit asks about) and anywhere in the trace (the row builder's
  drop rule, which also sees terminal output).
- tool_env: AGENT_TOOL_ENV from build.py (chukar's #207 pattern; uv tools dir, mini-swe-agent bin/lib), anywhere.
- swesmith_mirror: chukar's SWESMITH_REPOSITORY host prefix (task/212, tasktrove/repository_pytest.py),
  'github.com/swesmith/', in agent commands.
"""

import collections
import glob
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

# task/211 experiments/post_training/sweep_data/select.py (vizcacha), verbatim.
SWESMITH_INSTANCE = re.compile(r"\b([A-Za-z0-9_.-]+?)__([A-Za-z0-9_.-]+?)\.([0-9a-f]{8})\b")
KEYSTROKES = re.compile(r'"keystrokes":\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
GIT_HISTORY_READ = re.compile(
    r"git\s+(diff|show)\s+[^\n|;&]*(HEAD~|HEAD\^|\bmain\b|\bmaster\b|origin/|\b[0-9a-f]{7,40}\b)"
    r"|git\s+(checkout|restore)\s+[^\n|;&]*(\bmain\b|\bmaster\b|HEAD~|HEAD\^|--source)"
    r"|git\s+log\s+[^\n|;&]*(-p\b|--patch|--stat)"
    r"|git\s+(stash\s+show|cherry-pick|revert)"
)
# task/211 experiments/post_training/sft_shootout_rows/build.py (vizcacha; tool-env pattern from chukar's #207), verbatim.
LIBRARY_LOCATION = r"(?:site-packages|dist-packages|_vendor|/opt/miniconda3/pkgs|/go/pkg/mod)/(?:[^\s\"'<>]*/)?"
AGENT_TOOL_ENV = re.compile(r"(?:\.local/share/uv/tools|uv\s+tool\s+(?:dir|list|run)|mini-swe-agent/(?:bin|lib)/)")
# task/212 experiments/post_training/task_curation/datasets/tasktrove/repository_pytest.py (chukar): host prefix only.
SWESMITH_MIRROR = re.compile(r"github\.com/swesmith/")

POOL_TASKS = Path.home() / "Dropbox/marin-notes/mopd2/pool/tasks.jsonl"
IMPORT_NAMES = Path(__file__).with_name("import_names.json")


def library_copy_pattern(import_names):
    names = "|".join(re.escape(name) for name in import_names)
    return re.compile(LIBRARY_LOCATION + rf"(?:{names})(?:/|\.py\b|-[0-9]|@)")


def commands(messages):
    """Every keystroke string the agent sent, JSON-unescaped where possible."""
    out = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        for raw in KEYSTROKES.findall(message["content"] or ""):
            try:
                out.append(json.loads(f'"{raw}"'))
            except ValueError:
                out.append(raw)
    return out


def repo_of_text(text):
    repos = {f"{owner}/{name}" for owner, name, _ in SWESMITH_INSTANCE.findall(text or "")}
    return repos.pop() if len(repos) == 1 else None


def kimi_rows(data):
    for f in sorted(glob.glob(f"{data}/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k/data/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            yield {"stream": "Kimi-2.5", "trial": r["trial_name"], "task": r["task"], "outcome": outcome(r["result"]),
                   "repo_text": r["verifier_output"], "messages": r["conversations"]}


def calibforge_rows(data):
    for f in sorted(glob.glob(f"{data}/calibforge-relay-traces/data/terminus2_relay_sft/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            if r["source"] != "kimi_swesmith":
                continue
            yield {"stream": "kimi_swesmith", "trial": r["trial"], "task": r["task"],
                   "outcome": "pass" if str(r["passed"]) == "True" else "fail", "repo_text": None,
                   "messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]]}


def glm47_rows(data):
    for f in sorted(glob.glob(f"{data}/GLM-4.7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k/data/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            yield {"stream": "GLM-4.7", "trial": r["trial_name"], "task": r["task"], "outcome": outcome(r["result"]),
                   "repo_text": r["verifier_output"], "messages": r["conversations"]}


def agenttrove_rows(data):
    for f in sorted(glob.glob(f"{data}/agenttrove-swesmith/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            yield {"stream": f"{r['trace_source']} ({r['original_teacher']})", "trial": r["trial_name"], "task": r["task"],
                   "outcome": outcome(r["result"]), "repo_text": r["verifier_output"], "messages": r["conversations"]}


def outcome(result):
    if result in ("1.0", "1", 1.0):
        return "pass"
    if result in ("0.0", "0", 0.0):
        return "fail"
    return "unlabelled" if result in (None, "None", "") else f"error:{result}"


LOADERS = {"kimi": kimi_rows, "calibforge": calibforge_rows, "glm47": glm47_rows, "agenttrove": agenttrove_rows}


def main():
    data, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    names = json.loads(IMPORT_NAMES.read_text())
    task_repo = {}
    for line in POOL_TASKS.read_text().splitlines():
        row = json.loads(line)
        if row.get("family") == "swesmith" and row.get("repo"):
            task_repo[row["task_id"]] = row["repo"]
    # The Kimi set's verifier output names each task's instance; the other sets reuse its task ids (swesmith-NNNNN).
    for r in kimi_rows(data):
        if (repo := repo_of_text(r["repo_text"])) and r["task"] not in task_repo:
            task_repo[r["task"]] = repo
    patterns = {repo: library_copy_pattern(n) for repo, n in names.items() if n}
    for dataset in sys.argv[3:] or LOADERS:
        n, missing = 0, collections.Counter()
        with open(out / f"{dataset}-traces.jsonl", "w") as w:
            for r in LOADERS[dataset](data):
                n += 1
                cmds = commands(r["messages"])
                whole = json.dumps(r["messages"])
                repo = repo_of_text(r["repo_text"]) or task_repo.get(r["task"]) or repo_of_text(whole)
                pattern = patterns.get(repo)
                if repo and repo not in names:
                    missing[repo] += 1
                hits = {}
                for rule, test in (
                    ("git_history", lambda c: GIT_HISTORY_READ.search(c)),
                    ("library_copy_cmd", lambda c: pattern and pattern.search(c)),
                    ("swesmith_mirror", lambda c: SWESMITH_MIRROR.search(c)),
                ):
                    first = next((c for c in cmds if test(c)), None)
                    if first is not None:
                        hits[rule] = first[:300]
                if pattern and (m := pattern.search(whole)):
                    hits["library_copy_any"] = whole[max(0, m.start() - 120):m.end() + 60]
                if m := AGENT_TOOL_ENV.search(whole):
                    hits["tool_env_any"] = whole[max(0, m.start() - 120):m.end() + 60]
                w.write(json.dumps({"dataset": dataset, "stream": r["stream"], "trial": r["trial"], "task": r["task"],
                                    "outcome": r["outcome"], "repo": repo, "n_commands": len(cmds), "hits": hits}) + "\n")
        print(f"WROTE {dataset} {n} traces; repos without import names: {dict(missing)}", flush=True)


if __name__ == "__main__":
    main()
