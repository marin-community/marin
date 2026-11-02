# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Choose the files the pilot scan reads (Gate 3).

Takes a deterministic sample of each hero source's normalized files, and every
file of the sources most likely to hold eval text.

Usage: pilot_plan.py SOURCES_JSON OUT_PLAN_JSON
"""

import json
import math
import random
import sys
from pathlib import Path

SAMPLE_FRACTION = 0.01
SEED = 0
# Sources scanned in full: Q&A, exam, math, SFT and recent-crawl data.
HIGH_RISK_SOURCES = (
    "common-crawl-focus-2026-22",
    "cp/stackexchange",
    "cp/wikiteam",
    "nemotron_cc_math_v1/3",
    "nemotron_cc_math_v1/4plus_mind",
    "nemotron_sft/sft_code",
    "nemotron_sft/sft_general",
    "nemotron_sft/sft_math",
    "nemotron_specialized/infinibyte_reasoning",
    "nemotron_specialized/math_textbooks",
    "nemotron_specialized/stem_sft",
    "nemotron_specialized_v1_1/multiple_choice",
    "nemotron_specialized_v1_2/fact_seeking",
    "nemotron_specialized_v1_2/multiple_choice",
    "numinamath-1.5",
    "numinamath-tir",
    "synthetic-1",
)


def main() -> None:
    sources = json.loads(Path(sys.argv[1]).read_text())
    missing = set(HIGH_RISK_SOURCES) - set(sources)
    if missing:
        raise ValueError(f"unknown sources: {sorted(missing)}")
    rng = random.Random(SEED)
    plan = {}
    for name, info in sorted(sources.items()):
        paths = [p for p, _ in info["files"]]
        if name in HIGH_RISK_SOURCES:
            plan[name] = paths
        else:
            plan[name] = sorted(rng.sample(paths, max(1, math.ceil(SAMPLE_FRACTION * len(paths)))))
    sizes = {p: s for info in sources.values() for p, s in info["files"]}
    chosen = [p for paths in plan.values() for p in paths]
    print(f"{len(chosen)} files, {sum(sizes[p] for p in chosen) / 1e12:.2f} TB")
    Path(sys.argv[2]).write_text(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
