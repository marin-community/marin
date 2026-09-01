# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate retained-trajectory statistics for a snowball arm.

Reports, per 20-step bucket: truncation rate (stop_reason=length), canonical
think-token usage, graded answer-line compliance, and \\boxed{} usage. Run
in-region on cw-us-east-02a so the S3 reads stay local:

    python -m experiments.post_training.curriculum_rl.trajectory_stats \\
        snowball-grade-prior-dapo-snowball-full-r4 2026.09.01.1
"""

import gzip
import io as std_io
import json
import re
import sys
import zipfile
from collections import defaultdict

from marin.training.training import temporary_storage_base_path
from rigging.filesystem.storage_path import StoragePath, prefix_join

MAX_ARCHIVES = 80
STEP_BUCKET = 20
THINK_TOKENS = ("<|start_think|>", "<|end_think|>", "<think>", "</think>")
ANSWER_LINE = re.compile(r"(####\s*\S+|Answer:\s*\S+)")
BIN_TOKENS = ("gsm8k", "svamp", "asdiv", "math", "numina", "omni", "aime", "theoremqa", "hardmath", "rg-sum", "putnam")


def record_stats(rec: dict, reasons: tuple[str, ...]) -> dict:
    resp = rec.get("response", {})
    text = resp.get("text") or ""
    extras = json.dumps(rec.get("trajectory", {}).get("environment_extras"))
    bin_token = next((t for t in BIN_TOKENS if t in extras), "other")
    return {
        "step": rec.get("global_step") or 0,
        "bin": bin_token,
        # Failures and truncations are retained mandatorily, so the
        # non-mandatory remainder is a hash sample of successful terminating
        # rollouts only. None when the record id is missing from the ledger.
        "sampled": None if not reasons else "mandatory" not in reasons,
        "truncated": resp.get("stop_reason") == "length",
        "think": any(t in text for t in THINK_TOKENS),
        "answer_line": bool(ANSWER_LINE.search(text[-400:])),
        "boxed": "\\boxed{" in text[-400:],
        "chars": len(text),
    }


def main() -> None:
    arm = sys.argv[1]
    version = sys.argv[2]
    out = f"s3://marin-us-east-02a/marin/users/power/checkpoints/curriculum-rl/{arm}/{version}"
    root = temporary_storage_base_path(out, ttl_days=14, category="skyrl")
    traj_root = prefix_join(prefix_join(root, "attempts"), "trajectories")
    ledger = json.loads(StoragePath(prefix_join(traj_root, "_retention_ledger.json")).read_text())
    reasons_by_id = {rid: tuple(entry.get("reasons", ())) for rid, entry in ledger.get("records", {}).items()}
    archives = sorted(ledger.get("archives", {}))
    if len(archives) > MAX_ARCHIVES:
        stride = len(archives) / MAX_ARCHIVES
        sampled = [archives[int(i * stride)] for i in range(MAX_ARCHIVES)]
        print(f"sampling {MAX_ARCHIVES} of {len(archives)} archives (even stride)")
    else:
        sampled = archives
        print(f"reading all {len(archives)} archives")

    rows = []
    for apath in sampled:
        payload = StoragePath(prefix_join(traj_root, apath)).read_bytes()
        try:
            zf = zipfile.ZipFile(std_io.BytesIO(payload))
        except zipfile.BadZipFile as exc:
            print("archive", apath, "unreadable:", exc)
            continue
        for name in zf.namelist():
            if not name.endswith(".json.gz"):
                continue
            record_id = name.rsplit("/", 1)[-1].removesuffix(".json.gz")
            reasons = reasons_by_id.get(record_id, ())
            rows.append(record_stats(json.loads(gzip.decompress(zf.read(name))), reasons))
    matched = sum(1 for r in rows if r["sampled"] is not None)
    print(f"records: {len(rows)} (ledger reason matched for {matched})")

    def bucket(rows_subset: list[dict], label: str) -> None:
        n = len(rows_subset)
        if not n:
            return
        trunc = sum(r["truncated"] for r in rows_subset) / n
        think = sum(r["think"] for r in rows_subset) / n
        ans = sum(r["answer_line"] for r in rows_subset) / n
        boxed = sum(r["boxed"] for r in rows_subset) / n
        chars = sum(r["chars"] for r in rows_subset) / n
        print(
            f"{label:>16} n={n:<6} trunc={trunc:.3f} think={think:.3f} "
            f"answer_line={ans:.3f} boxed={boxed:.3f} avg_chars={chars:,.0f}"
        )

    for label, subset in (
        ("success stream (hash-sampled)", [r for r in rows if r["sampled"] is True]),
        ("failure/trunc stream (mandatory)", [r for r in rows if r["sampled"] is False]),
    ):
        print(f"\n==== {label} ====")
        by_step = defaultdict(list)
        for r in subset:
            by_step[r["step"] // STEP_BUCKET * STEP_BUCKET].append(r)
        for k in sorted(by_step):
            bucket(by_step[k], f"steps {k}-{k + STEP_BUCKET - 1}")
        by_bin = defaultdict(list)
        for r in subset:
            by_bin[r["bin"]].append(r)
        for k in sorted(by_bin):
            bucket(by_bin[k], k)


if __name__ == "__main__":
    main()
