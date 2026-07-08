#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "numpy"]
# ///
"""Redundancy pools: what a shared wiki/memory or RAG index could have served.

Two levers, both priced with each session's *observed* amplifier (not the
overcounting full-retention model), and split by content stability so we never
count re-reading a genuinely-changed file as a free win.

- **Read redundancy (wiki/cache lever).** A file read is *cross-session redundant*
  if an earlier session already read that path — the knowledge predated this
  session, so a shared memory could have served a compact entry instead. Split
  byte-identical (pure cache win) vs changed (needs a structural summary).
  Within-session re-reads are tracked separately (context hygiene, priced at the
  0.1x cache-read rate only — not bankable as shared memory).

- **Exploration redundancy (RAG/memory lever).** Bash/Grep/Glob discovery whose
  *output is stable across sessions* (low distinct-output-hash ratio) is
  memory-addressable; volatile self-inspection (git diff/status) is not, judged
  by output-hash churn rather than command name.

Prices are base-price input-equivalents: saving C content tokens in a session
with observed amplifier A saves `C * (1.25 + 0.1*min(remaining, A))`.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

P_WRITE, P_READ = 1.25, 0.10
DOC_RE = ("AGENTS.md", "OPS.md", "CLAUDE.md", "README", "TESTING.md", ".md")

# Shapes whose answer is a *stable fact about the repo/world* — memory/RAG can
# serve them. Excludes working-tree self-inspection (git diff/status) and any
# build/mutation. `git log`/`git show` (a fixed ref) and `gh * view` are stable.
ADDRESSABLE_SHAPES = {
    "rg",
    "grep",
    "glob",
    "find",
    "cat",
    "head",
    "tail",
    "ls",
    "tree",
    "wc",
    "sed",
    "awk",
    "git log",
    "git show",
    "git blame",
    "gh pr",
    "gh issue",
    "column",
    "jq",
}


def load(data_dir):
    b = pd.read_parquet(os.path.join(data_dir, "blocks.parquet"))
    amp = pd.read_parquet(os.path.join(data_dir, "session_amplifier.parquet"))
    return b, amp


def billed_saved(C, remaining, A):
    """Base-price input-equivalents saved by not loading C tokens."""
    return C * (P_WRITE + P_READ * np.minimum(remaining, A))


def path_bucket(p: str) -> str:
    if p is None:
        return "other"
    if any(p.endswith(s) or s in p for s in DOC_RE):
        return "docs"
    if p.endswith((".py", ".rs", ".ts", ".tsx", ".js", ".go", ".java", ".c", ".cpp", ".h")):
        return "source"
    if p.endswith((".yaml", ".yml", ".toml", ".json", ".cfg", ".ini")):
        return "config"
    return "other"


def read_redundancy(blocks, amp):
    reads = blocks[(blocks.tool_name == "Read") & (blocks.block_type == "tool_result")].copy()
    reads = reads.dropna(subset=["target"])
    reads = reads.merge(amp[["session_id", "n_turns", "observed_amplifier"]], on="session_id", how="left")
    reads["n_turns"] = reads["n_turns"].fillna(1)
    reads["observed_amplifier"] = reads["observed_amplifier"].fillna(1.0)
    reads["remaining"] = (reads["n_turns"] - reads["first_paid_turn"]).clip(lower=0)
    reads = reads.sort_values("ts")

    # earliest session per path, and that path's first-ever content hash
    first = (
        reads.groupby("target")
        .agg(first_session=("session_id", "first"), first_sha=("content_sha", "first"))
        .reset_index()
    )
    reads = reads.merge(first, on="target")
    reads["cross_session"] = reads.session_id != reads.first_session
    # within a session: occurrence index per (session, path)
    reads["occ"] = reads.groupby(["session_id", "target"]).cumcount()
    reads["within_session_redundant"] = (~reads.cross_session) & (reads.occ > 0)
    reads["byte_identical"] = reads.content_sha == reads.first_sha
    reads["bucket"] = reads.target.map(path_bucket)
    reads["saved"] = billed_saved(reads.est_tokens, reads.remaining, reads.observed_amplifier)

    xs = reads[reads.cross_session]
    ws = reads[reads.within_session_redundant]
    out = {
        "n_read_results": len(reads),
        "n_distinct_paths": int(reads.target.nunique()),
        "cross_session": {
            "n_reads": len(xs),
            "n_paths": int(xs.target.nunique()),
            "gross_saved_input_equiv": int(xs.saved.sum()),
            "byte_identical_saved": int(xs[xs.byte_identical].saved.sum()),
            "changed_saved": int(xs[~xs.byte_identical].saved.sum()),
            "byte_identical_frac": round(float(xs[xs.byte_identical].saved.sum() / max(xs.saved.sum(), 1)), 3),
            "saved_by_bucket": {
                k: int(v) for k, v in xs.groupby("bucket").saved.sum().sort_values(ascending=False).items()
            },
        },
        "within_session_hygiene": {
            "n_reads": len(ws),
            # within-session re-reads are already cache-read; value them at 0.1x only
            "saved_input_equiv_cacheread_only": int(
                (ws.est_tokens * P_READ * ws.remaining.clip(upper=ws.observed_amplifier)).sum()
            ),
        },
    }
    top = (
        xs.groupby("target")
        .agg(
            saved=("saved", "sum"),
            n=("saved", "size"),
            n_sess=("session_id", "nunique"),
            bucket=("bucket", "first"),
            bi=("byte_identical", "mean"),
        )
        .sort_values("saved", ascending=False)
        .head(25)
    )
    out["top_paths"] = [
        {
            "path": t,
            "saved": int(r.saved),
            "reads": int(r.n),
            "sessions": int(r.n_sess),
            "bucket": r.bucket,
            "byte_identical_frac": round(float(r.bi), 2),
        }
        for t, r in top.iterrows()
    ]
    return out, reads


def exploration_redundancy(blocks, amp):
    disc = blocks[(blocks.tool_name.isin(["Bash", "Grep", "Glob"])) & (blocks.block_type == "tool_result")].copy()
    disc = disc.dropna(subset=["target"])
    disc = disc.merge(amp[["session_id", "n_turns", "observed_amplifier"]], on="session_id", how="left")
    disc["n_turns"] = disc["n_turns"].fillna(1)
    disc["observed_amplifier"] = disc["observed_amplifier"].fillna(1.0)
    disc["remaining"] = (disc["n_turns"] - disc["first_paid_turn"]).clip(lower=0)
    disc["saved"] = billed_saved(disc.est_tokens, disc.remaining, disc.observed_amplifier)
    disc["verb"] = disc.target.str.split("\t").str[0]

    # per (target = shape+arg): output-hash churn = distinct output shas / calls
    g = (
        disc.groupby("target")
        .agg(
            calls=("content_sha", "size"),
            sessions=("session_id", "nunique"),
            distinct_out=("content_sha", "nunique"),
            tokens=("est_tokens", "sum"),
            saved=("saved", "sum"),
            verb=("verb", "first"),
        )
        .reset_index()
    )
    g["churn"] = g.distinct_out / g.calls  # ~0 = stable output, ~1 = volatile
    g["repeated"] = g.calls > 1

    # memory-addressable pool: an allowlisted stable-answer shape, repeated across
    # >1 session, with low output churn (identical output seen before).
    g["addressable_shape"] = g["verb"].isin(ADDRESSABLE_SHAPES)
    stable = g[g.addressable_shape & (g.sessions > 1) & (g.churn <= 0.5)]
    volatile = g[~(g.addressable_shape & (g.churn <= 0.5))]
    # gross saving = the repeat calls (beyond first) of stable targets
    stable_repeat_saved = int(
        disc.merge(stable[["target"]], on="target")
        .assign(occ=lambda d: d.groupby("target").cumcount())
        .query("occ > 0")
        .saved.sum()
    )
    out = {
        "n_discovery_results": len(disc),
        "stable_addressable": {
            "n_targets": len(stable),
            "gross_repeat_saved_input_equiv": stable_repeat_saved,
            "example_targets": [
                {
                    "target": r.target[:70],
                    "calls": int(r.calls),
                    "sessions": int(r.sessions),
                    "churn": round(float(r.churn), 2),
                    "saved": int(r.saved),
                }
                for _, r in stable.sort_values("saved", ascending=False).head(20).iterrows()
            ],
        },
        "volatile_not_addressable": {
            "n_targets": len(volatile),
            "tokens_input_equiv": int(volatile.saved.sum()),
            "top": [
                {"shape": r.verb, "calls": int(r.calls), "churn": round(float(r.churn), 2), "saved": int(r.saved)}
                for _, r in volatile.sort_values("saved", ascending=False).head(10).iterrows()
            ],
        },
    }
    by_shape = (
        g.groupby("verb")
        .agg(saved=("saved", "sum"), calls=("calls", "sum"), churn=("churn", "mean"))
        .sort_values("saved", ascending=False)
        .head(15)
    )
    out["by_shape"] = {
        str(shape): {"saved": int(r.saved), "calls": int(r.calls), "mean_churn": round(float(r.churn), 2)}
        for shape, r in by_shape.iterrows()
    }
    return out, disc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "_data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "_data", "redundancy.json"))
    args = ap.parse_args()
    blocks, amp = load(args.data)

    rr, reads = read_redundancy(blocks, amp)
    er, _disc = exploration_redundancy(blocks, amp)
    result = {"read_redundancy": rr, "exploration_redundancy": er}
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    # export redundant-event records for the grounded audit (uplift.py)
    xs = reads[reads.cross_session][
        ["session_id", "target", "est_tokens", "saved", "byte_identical", "bucket", "content_sha"]
    ].copy()
    xs["kind"] = "read"
    xs.sort_values("saved", ascending=False).head(4000).to_parquet(
        os.path.join(args.data, "redundant_reads.parquet"), index=False
    )

    print(json.dumps(result, indent=2)[:3000])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
