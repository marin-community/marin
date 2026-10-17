"""Task #215: markdown tables from audit.py's <dataset>-traces.jsonl files.

Usage: report.py <out dir> [dataset ...]. Prints one section per dataset: counts by outcome, by stream (when a set has
several), and passing traces by repo.
"""

import collections
import json
import sys
from pathlib import Path

RULES = ("git_history", "library_copy_cmd", "swesmith_mirror", "tool_env_any")
LEAK = set(RULES)  # 'any leak path': library copies count only when the agent's own command reads them.
COLUMNS = ("traces", "git_history", "library_copy_cmd", "library_copy_any", "tool_env_any", "swesmith_mirror", "any")


def bucket(outcome):
    return outcome if outcome in ("pass", "fail", "unlabelled") else "error"


def counts(rows):
    c = collections.Counter()
    for r in rows:
        c["traces"] += 1
        for rule in r["hits"]:
            c[rule] += 1
        c["any"] += bool(LEAK & set(r["hits"]))
    return c


def share(c):
    return f"{100 * c['any'] / c['traces']:.1f}%" if c["traces"] else "-"


def table(groups, label):
    lines = [f"| {label} | " + " | ".join(COLUMNS) + " | any, share |", "|---|" + "---:|" * (len(COLUMNS) + 1)]
    for key, rows in groups:
        c = counts(rows)
        lines.append(f"| {key} | " + " | ".join(f"{c[col]:,}" for col in COLUMNS) + f" | {share(c)} |")
    return lines


def section(dataset, rows):
    out = [f"### {dataset}", ""]
    by_outcome = collections.defaultdict(list)
    for r in rows:
        by_outcome[bucket(r["outcome"])].append(r)
    order = [k for k in ("pass", "fail", "unlabelled", "error") if k in by_outcome]
    out += table([(k, by_outcome[k]) for k in order] + [("all", rows)], "outcome") + [""]
    streams = sorted({r["stream"] for r in rows})
    if len(streams) > 1:
        out += ["By stream (all outcomes; pass share in the last column):", ""]
        lines = table([(s, [r for r in rows if r["stream"] == s]) for s in streams], "stream")
        lines[0] += " passes | passes with any leak |"
        lines[1] += "---:|---:|"
        for i, s in enumerate(streams):
            passes = [r for r in rows if r["stream"] == s and r["outcome"] == "pass"]
            c = counts(passes)
            lines[i + 2] += f" {c['traces']:,} | {c['any']:,} ({share(c)}) |"
        out += lines + [""]
    key = "pass" if by_outcome.get("pass") else "unlabelled"
    by_repo = collections.defaultdict(list)
    for r in by_outcome.get(key, []):
        by_repo[r["repo"] or "(unknown)"].append(r)
    out += [f"By repo ({key} traces only):", ""]
    out += table(sorted(by_repo.items(), key=lambda kv: (-len(kv[1]), kv[0])), "repo") + [""]
    return out


def main():
    out = Path(sys.argv[1])
    datasets = sys.argv[2:] or [p.name.removesuffix("-traces.jsonl") for p in sorted(out.glob("*-traces.jsonl"))]
    for dataset in datasets:
        rows = [json.loads(line) for line in open(out / f"{dataset}-traces.jsonl")]
        print("\n".join(section(dataset, rows)))


if __name__ == "__main__":
    main()
