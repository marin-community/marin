# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-response loader: Qwen-tokenizer token table plus per-char NLL for all five scorers.

Keeps whole responses and records two leak cuts so either can be applied later:
  cut_line  - first line-start role line (`\\nuser\\n`, `\\nAssistant:`), the cut behind the
              +0.035 (Qwen3.6) figure quoted for the "leak-cut" SFT text.
  cut_fused - common.LEAK, which also catches a role word fused onto the end of the answer
              (`...be built.user\\n`), i.e. a Llama-3 turn header whose special tokens were
              stripped on decode.
"""

import gzip
import json
import pickle
import re
from dataclasses import dataclass

import numpy as np
from common import HERE, SCORERS, TL, TRACES, char_nll, leak_cut

CACHE = HERE / "full.pkl"
QS = SCORERS[:3]  # the three Qwen scorers share one tokenizer
LINE_ROLE = re.compile(r"(^|\n)\s*(user|assistant|User|Assistant)\s*(\n|:)")


def line_cut(text: str) -> int:
    m = LINE_ROLE.search(text)
    return m.start() if m else len(text)


@dataclass
class Full:
    trace_id: str
    prompt_key: int
    author: str
    passed: bool
    prompt: str
    text: str
    cut_line: int
    cut_fused: int
    offsets: np.ndarray  # Qwen token char offsets, shape (n, 2)
    tok: dict[str, np.ndarray]  # Qwen scorer -> token NLL
    char: dict[str, np.ndarray]  # every scorer -> per-char NLL over full text (NaN = unscored)


def load_full() -> list[Full]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    raw = {t["trace_id"]: t for t in map(json.loads, TRACES.read_text().splitlines())}
    recs = {s: {} for s in SCORERS}
    for s in SCORERS:
        for line in gzip.open(TL / "data" / f"{s}.jsonl.gz", "rt"):
            r = json.loads(line)
            recs[s][r["trace_id"]] = r
    out = []
    for tid, t in raw.items():
        text = t["response"]
        offs = np.array(recs[QS[0]][tid]["offsets"]).reshape(-1, 2)
        for s in QS[1:]:
            assert np.array_equal(np.array(recs[s][tid]["offsets"]).reshape(-1, 2), offs), (s, tid)
        keep = offs[:, 1] > offs[:, 0]
        out.append(
            Full(
                tid,
                int(t["prompt_key"]),
                t["author"],
                t["passed"] in (True, "True"),
                t["prompt"],
                text,
                line_cut(text),
                leak_cut(text),
                offs[keep],
                {s: np.array(recs[s][tid]["nll"])[keep] for s in QS},
                {s: char_nll(len(text), recs[s][tid]["offsets"], recs[s][tid]["nll"]) for s in SCORERS},
            )
        )
    CACHE.write_bytes(pickle.dumps(out))
    return out


if __name__ == "__main__":
    F = load_full()
    print(len(F), sum(len(f.offsets) for f in F))
