# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare v0-template trace length with GLM-side token counts, to size a generation budget (task #211 follow-up).

For every single-conversation trace, the GLM-side length is the last assistant turn's reported usage (prompt +
completion tokens: the full context GLM saw plus its final reply); the v0 length is the trace rendered with v0's chat
template. Prints per-family ratio percentiles and, for a target v0 length (32,768), the GLM-side budget at which a
given share of traces would fit.

Usage:
    python -m experiments.post_training.sft_shootout_rows.length_ratio <rebuilt.jsonl> ...
"""

import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.traces import render_trace
from experiments.post_training.sweep_data.prepare_agentic import STUDENT_MODEL, STUDENT_REVISION, load_tokenizer
from experiments.post_training.sweep_data.select import CONTEXT


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        audit = hf_hub_download(STUDENT_MODEL, "provenance/campaign3/audit.json", revision=STUDENT_REVISION)
        expected = {name: entry["sha256"] for name, entry in json.loads(Path(audit).read_text())["files"].items()}
        tokenizer = load_tokenizer(Path(temporary), expected)
        samples: dict[str, list[tuple[int, int, bool]]] = {}
        for uri in sys.argv[1:]:
            for line in StoragePath(uri).read_text().splitlines():
                trace = json.loads(line)
                if trace.get("n_conversations", 1) > 1 or trace["problems"]:
                    continue
                usage = next(
                    (m.get("usage") for m in reversed(trace["messages"]) if m["role"] == "assistant" and m.get("usage")),
                    None,
                )
                if not usage:
                    continue
                glm = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
                try:
                    v0 = len(render_trace(tokenizer, trace["messages"], trace.get("tools")).ids)
                except ValueError:
                    continue
                if glm:
                    samples.setdefault(trace["family"], []).append((glm, v0, bool(trace.get("passed"))))
    for family, rows in sorted(samples.items()):
        ratios = [v0 / glm for glm, v0, _ in rows]
        report = {
            "family": family,
            "traces": len(rows),
            "passed": sum(passed for _, _, passed in rows),
            "ratio_v0_over_glm": {
                f"p{int(f * 100)}": round(percentile(ratios, f), 3) for f in (0.05, 0.5, 0.9, 0.95, 0.99)
            },
            "over_context_share": round(sum(v0 > CONTEXT for _, v0, _ in rows) / len(rows), 3),
            "glm_budget_for_fit": {f"p{int(f * 100)}": int(CONTEXT / percentile(ratios, f)) for f in (0.9, 0.95, 0.99)},
        }
        print("LENGTH_RATIO " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
