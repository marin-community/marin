# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task #215: audit public SWE-smith trace sets for the #10212 / #10223 leak paths. Read-only, CPU only.

Usage: audit.py <data dir> <out dir> <task_repo.json> <fix_lines.json> [dataset ...]
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
- tool_env_cmd: AGENT_TOOL_ENV in agent commands (the leak path; tool_env_any also counts terminal output).
Supplementary, not in the shootout's rules:
- go_module_cache: LIBRARY_LOCATION + a Go repo's module path (the shootout lists no import names for Go repos).
- git_history_saw_fix: the terminal output after a git_history command holds a line of the fix (a '-' line of the
  task's bug patch in a non-test file, 15+ characters, unlike every '+' line) that no earlier agent turn wrote.
  Needs the task id, so AgentTrove rows are not tested. Tasks whose fix has no such line are marked untestable.
- git_remote_fetch: 'git fetch|pull|clone|ls-remote' in agent commands; in the upstream images the remote is the
  task's github.com/swesmith mirror, so these reach it without naming the host.
Import names: the shootout's IMPORT_NAMES (sft_shootout_pool/repo_names.py) for its 38 repos, extended to other
repos in import_names.json; entries with a '/' are Go module paths and feed go_module_cache only.
Repo of a trace: DCAgent's package config (tests/config.json instance_id) for the task id, else the one SWE-smith
instance id named in the verifier output or the trace.
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
# task/211 experiments/post_training/sft_shootout_rows/build.py (vizcacha; tool-env pattern from chukar's #207),
# verbatim.
LIBRARY_LOCATION = r"(?:site-packages|dist-packages|_vendor|/opt/miniconda3/pkgs|/go/pkg/mod)/(?:[^\s\"'<>]*/)?"
AGENT_TOOL_ENV = re.compile(r"(?:\.local/share/uv/tools|uv\s+tool\s+(?:dir|list|run)|mini-swe-agent/(?:bin|lib)/)")
# task/212 experiments/post_training/task_curation/datasets/tasktrove/repository_pytest.py (chukar): host prefix only.
SWESMITH_MIRROR = re.compile(r"github\.com/swesmith/")
# Task format: some AgentTrove streams ship no repo; the instruction says "clone the repo `swesmith/<mirror>` and run
# git fetch && git checkout <instance branch>", so the mirror (every branch, the clean main included) is in the sandbox
# by design.
CLONE_TASK = re.compile(r"clone the repo `?swesmith/")
GIT_REMOTE_FETCH = re.compile(r"\bgit\s+(?:fetch|pull|clone|ls-remote)\b")

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


DIFF_CONTENT = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@|^diff --git ", re.MULTILINE)
SOURCE_HUNK = re.compile(r"^diff --git a/(\S+)", re.MULTILINE)


def history_output(messages):
    """Per agent turn whose keystrokes match GIT_HISTORY_READ: (next terminal output, all agent text up to that turn)."""
    out = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        keys = []
        for raw in KEYSTROKES.findall(message["content"] or ""):
            try:
                keys.append(json.loads(f'"{raw}"'))
            except ValueError:
                keys.append(raw)
        if any(GIT_HISTORY_READ.search(k) for k in keys):
            following = next((m["content"] or "" for m in messages[index + 1 :] if m["role"] == "user"), "")
            earlier = "\n".join(m["content"] or "" for m in messages[: index + 1] if m["role"] == "assistant")
            out.append((following, earlier))
    return out


def repo_of_text(text):
    repos = {f"{owner}/{name}" for owner, name, _ in SWESMITH_INSTANCE.findall(text or "")}
    return repos.pop() if len(repos) == 1 else None


def kimi_rows(data):
    for f in sorted(
        glob.glob(f"{data}/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k/data/*.parquet")
    ):
        for r in pq.read_table(f).to_pylist():
            yield {
                "stream": "Kimi-2.5",
                "trial": r["trial_name"],
                "task": r["task"],
                "outcome": outcome(r["result"]),
                "repo_text": r["verifier_output"],
                "messages": r["conversations"],
            }


def calibforge_rows(data):
    for f in sorted(glob.glob(f"{data}/calibforge-relay-traces/data/terminus2_relay_sft/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            if r["source"] != "kimi_swesmith":
                continue
            yield {
                "stream": "kimi_swesmith",
                "trial": r["trial"],
                "task": r["task"],
                "outcome": "pass" if str(r["passed"]) == "True" else "fail",
                "repo_text": None,
                "source_flags": {"leak": r["leak"], "hunt": r["hunt"]},
                "messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]],
            }


def glm47_rows(data):
    for f in sorted(
        glob.glob(f"{data}/GLM-4.7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k/data/*.parquet")
    ):
        for r in pq.read_table(f).to_pylist():
            yield {
                "stream": "GLM-4.7",
                "trial": r["trial_name"],
                "task": r["task"],
                "outcome": outcome(r["result"]),
                "repo_text": r["verifier_output"],
                "messages": r["conversations"],
            }


def agenttrove_rows(data):
    for f in sorted(glob.glob(f"{data}/agenttrove-swesmith/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            # Only the Kimi-2.5 rows (the Kimi set's 5,237 passes, result '1.0') carry a verifier verdict. GLM-5.0's
            # 'success' is the agent's exit status; the other streams have no result. Error names are kept.
            source = r["trace_source"] or "none"
            kind = "summarization" if source.startswith("summarization") else "job path" if "/" in source else source
            # AgentTrove's streams number tasks differently (9,050 of 14,943 task-id lookups disagreed with the trace),
            # so the task id is not passed on: the repo comes from the verifier output or the trace text only.
            yield {
                "stream": f"{r['original_teacher']} / trace_source {kind}",
                "trial": r["trial_name"],
                "task": None,
                "outcome": "unlabelled" if r["result"] == "success" else outcome(r["result"]),
                "repo_text": r["verifier_output"],
                "messages": r["conversations"],
            }


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
    task_repo = {task: entry["repo"] for task, entry in json.loads(Path(sys.argv[3]).read_text()).items()}
    fix_lines = json.loads(Path(sys.argv[4]).read_text())
    python_names = {repo: [n for n in found if "/" not in n] for repo, found in names.items()}
    go_names = {repo: [n for n in found if "/" in n] for repo, found in names.items()}
    patterns = {repo: library_copy_pattern(n) for repo, n in python_names.items() if n}
    go_patterns = {repo: library_copy_pattern(n) for repo, n in go_names.items() if n}
    for dataset in sys.argv[5:] or LOADERS:
        n, missing = 0, collections.Counter()
        with open(out / f"{dataset}-traces.jsonl", "w") as w:
            for r in LOADERS[dataset](data):
                n += 1
                cmds = commands(r["messages"])
                whole = json.dumps(r["messages"])
                by_label, by_task, by_trace = repo_of_text(r["repo_text"]), task_repo.get(r["task"]), repo_of_text(whole)
                repo = by_task or by_label or by_trace
                pattern = patterns.get(repo)
                go_pattern = go_patterns.get(repo)
                if repo and repo not in names:
                    missing[repo] += 1
                hits = {}
                for rule, regex in (
                    ("git_history", GIT_HISTORY_READ),
                    ("library_copy_cmd", pattern),
                    ("swesmith_mirror", SWESMITH_MIRROR),
                    ("tool_env_cmd", AGENT_TOOL_ENV),
                    ("go_module_cache", go_pattern),
                    ("git_remote_fetch", GIT_REMOTE_FETCH),
                ):
                    if regex is None:
                        continue
                    first = next((c for c in cmds if regex.search(c)), None)
                    if first is not None:
                        hits[rule] = first[:300]
                if "git_history" in hits:
                    outputs = history_output(r["messages"])
                    printed = [o for o, _ in outputs if DIFF_CONTENT.search(o)]
                    lines = fix_lines.get(r["task"]) if r["task"] else None
                    if lines:
                        if any(line in o and line not in earlier for o, earlier in outputs for line in lines):
                            hits["git_history_saw_fix"] = True
                    else:
                        hits["git_history_fix_untestable"] = True
                    if printed:
                        hits["git_history_printed_diff"] = True
                        files = {f for o in printed for f in SOURCE_HUNK.findall(o)}
                        if any(
                            not re.search(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.(py|go)$", f)
                            for f in files
                        ):
                            hits["git_history_printed_source_diff"] = True
                if pattern and (m := pattern.search(whole)):
                    hits["library_copy_any"] = whole[max(0, m.start() - 120) : m.end() + 60]
                if m := AGENT_TOOL_ENV.search(whole):
                    hits["tool_env_any"] = whole[max(0, m.start() - 120) : m.end() + 60]
                first_user = next((m["content"] or "" for m in r["messages"] if m["role"] == "user"), "")
                clone_task = bool(CLONE_TASK.search(first_user))
                w.write(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "clone_task": clone_task,
                            "stream": r["stream"],
                            "trial": r["trial"],
                            "task": r["task"],
                            "outcome": r["outcome"],
                            "repo": repo,
                            "repo_by_task": by_task,
                            "repo_by_label": by_label,
                            "repo_by_trace": by_trace,
                            "n_commands": len(cmds),
                            "hits": hits,
                            "source_flags": r.get("source_flags"),
                        }
                    )
                    + "\n"
                )
        print(f"WROTE {dataset} {n} traces; repos without import names: {dict(missing)}", flush=True)


if __name__ == "__main__":
    main()
