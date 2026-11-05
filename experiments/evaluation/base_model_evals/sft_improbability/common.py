# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared loading for the Snowball-SFT improbability analysis.

Loads the IFBench trace set and the five scorers' per-token NLLs, spreads each token's NLL
evenly over its characters (as trace-likelihood/analyze_traces.py does), restricts to
characters every scorer scored, and cuts each response at its first leaked turn.
"""

import gzip
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TL = HERE.parent / "trace-likelihood"
TRACES = TL / "ifbench-11-authors-100-prompts.jsonl"
CACHE = HERE / "cache.pkl"

SCORERS = [
    "qwen3.5-35b-a3b-base",
    "qwen3.5-35b-a3b",
    "qwen3.6-35b-a3b",
    "snowball-67b-a2b-base-skew8",
    "grug-67b-a2b-datakit-sft",
]
SHORT = {
    "qwen3.5-35b-a3b-base": "q35base",
    "qwen3.5-35b-a3b": "q35post",
    "qwen3.6-35b-a3b": "q36post",
    "snowball-67b-a2b-base-skew8": "sbbase",
    "grug-67b-a2b-datakit-sft": "sft",
}
SFT_AUTHOR = "grugsft471"

# A leaked turn: a bare role word ending a line (often fused onto the previous sentence, e.g.
# "...built.user\n"), or a "User:"/"Assistant:" line.
LEAK = re.compile(r"(?<![A-Za-z])(?:user|assistant)[ \t]*\n|(?:^|\n)\[?(?:User|Assistant)\]?:")


@dataclass
class Trace:
    trace_id: str
    prompt_key: str
    author: str
    passed: bool
    prompt: str
    text: str  # leak-cut response
    full_len: int
    leaked: bool
    nll: dict[str, np.ndarray]  # scorer -> per-char nll over text (NaN where not scored by all)

    @property
    def mask(self) -> np.ndarray:
        return ~np.isnan(self.nll[SCORERS[0]])


def char_nll(n: int, offsets, nlls) -> np.ndarray:
    values = np.full(n, np.nan)
    for (start, end), nll in zip(offsets, nlls, strict=True):
        if end > start:
            values[start : min(end, n)] = nll / (end - start)
    return values


def leak_cut(text: str) -> int:
    m = LEAK.search(text)
    return m.start() if m else len(text)


def load() -> list[Trace]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    raw = {t["trace_id"]: t for t in map(json.loads, TRACES.read_text().splitlines())}
    per = {}
    for s in SCORERS:
        for line in gzip.open(TL / "data" / f"{s}.jsonl.gz", "rt"):
            r = json.loads(line)
            per[(s, r["trace_id"])] = char_nll(len(raw[r["trace_id"]]["response"]), r["offsets"], r["nll"])
    out = []
    for tid, t in raw.items():
        full = t["response"]
        cut = leak_cut(full)
        arrs = {s: per[(s, tid)][:cut].copy() for s in SCORERS}
        common = np.all([~np.isnan(a) for a in arrs.values()], axis=0)
        for a in arrs.values():
            a[~common] = np.nan
        out.append(
            Trace(
                tid,
                str(t["prompt_key"]),
                t["author"],
                t["passed"] in (True, "True"),
                t["prompt"],
                full[:cut],
                len(full),
                cut < len(full),
                arrs,
            )
        )
    CACHE.write_bytes(pickle.dumps(out))
    return out


def char_class(text: str, i: int) -> str:
    """Same classes as analyze_traces.py."""
    ch = text[i]
    line_start = i == 0 or text[i - 1] == "\n"
    if ch == "\n":
        return "newline"
    if ch in "#*|`>_~" or (line_start and ch in "-+"):
        return "markdown syntax"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    if not ch.isalnum():
        return "punctuation"
    word_start = i == 0 or not text[i - 1].isalnum()
    return "word start" if word_start else "word interior"


def gap_arrays(t: Trace, scorer: str, base: str = SCORERS[0]) -> np.ndarray:
    return t.nll[scorer] - t.nll[base]


def bootstrap(per_prompt_num: np.ndarray, per_prompt_den: np.ndarray, n: int = 4000, seed: int = 0):
    """Ratio-of-sums estimate with a 95% percentile CI, resampling prompts."""
    rng = np.random.default_rng(seed)
    k = len(per_prompt_num)
    idx = rng.integers(0, k, size=(n, k))
    stats = per_prompt_num[idx].sum(1) / per_prompt_den[idx].sum(1)
    return per_prompt_num.sum() / per_prompt_den.sum(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)
