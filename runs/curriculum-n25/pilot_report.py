# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize a proposal run's completions: finish reasons, token percentiles, blank-line runaways, verdicts.

uv run --frozen runs/curriculum-n25/pilot_report.py s3://.../pilot
"""

import json
import re
import sys
from collections import Counter

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

BLANK_TAIL = re.compile(r"(?:[ \t]*\n){20,}[ \t]*\Z")

configure_coreweave_s3()
fs, root = fsspec.core.url_to_fs(sys.argv[1].rstrip("/"))
finish, decisions, reasons = Counter(), Counter(), Counter()
out_tokens, reasoning_tokens, runaways, separate = [], [], 0, 0
for path in sorted(fs.glob(f"{root}/*/*.completions.json")):
    for c in json.loads(fs.cat(path)):
        finish[c["finish_reason"]] += 1
        out_tokens.append(c["usage"]["completion_tokens"])
        reasoning_tokens.append(c["usage"]["reasoning_tokens"])
        separate += bool(c["reasoning"]) and not c["content"].lstrip().startswith("<think>")
        if c["finish_reason"] == "length" and BLANK_TAIL.search(c["content"]):
            runaways += 1
for path in fs.glob(f"{root}/*/*.verdict.json"):
    v = json.loads(fs.cat(path))
    decisions[v["decision"]] += 1
    for r in v["reasons"]:
        reasons[r.split(":")[0]] += 1


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


print("completions", len(out_tokens), "finish", dict(finish), "blank-line runaways among length", runaways)
print("reasoning separate from content", separate, "of", len(out_tokens))
for name, xs in (("completion_tokens", out_tokens), ("reasoning_tokens", reasoning_tokens)):
    print(name, "p50", pct(xs, 0.5), "p95", pct(xs, 0.95), "max", max(xs, default=0))
print("verdicts", dict(decisions), "reject reasons", dict(reasons))
