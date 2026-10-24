# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive fixed-token serial/repeat/packed comparisons before evaluation."""

import hashlib
import json
import time

import numpy as np
from rigging.filesystem.storage_path import StoragePath


def _artifact(tokens, scores, identity, elapsed):
    rows = []
    for pos, (token, lp, tops) in enumerate(zip(tokens, scores.token_logprobs, scores.top_token_logprobs, strict=True)):
        top_id, top_lp = max(tops.items(), key=lambda pair: pair[1])
        if not np.isfinite(lp) or not all(np.isfinite(value) for value in tops.values()):
            raise ValueError("Nonfinite teacher-forced score; numerical gate cannot pass")
        rows.append(
            dict(
                position=pos,
                token_id=token,
                logprob=None if pos == 0 else lp,
                top1_token_id=None if pos == 0 else top_id,
                top1_logprob=None if pos == 0 else top_lp,
            )
        )
    return dict(
        id=identity,
        token_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        tokens=rows,
        elapsed_seconds=elapsed,
    )


def compare(candidate, reference):
    assert candidate["token_ids_sha256"] == reference["token_ids_sha256"]
    pairs = list(zip(candidate["tokens"][1:], reference["tokens"][1:], strict=True))

    def summarize(rows):
        if not rows:
            return None
        delta = np.asarray([abs(a["logprob"] - b["logprob"]) for a, b in rows])
        flips = sum(a["top1_token_id"] != b["top1_token_id"] for a, b in rows)
        return dict(
            positions=len(rows),
            mean_absolute_delta=float(delta.mean()),
            max_absolute_delta=float(delta.max()),
            argmax_disagreements=flips,
            argmax_disagreement_fraction=flips / len(rows),
            positions_above_one_nat=int(np.sum(delta > 1)),
        )

    ranges = [(1, 1024), (1024, 1800), (1800, 2046), (2046, 2054), (2054, 2304), (2304, 4000), (4000, 6000)]
    return dict(
        overall=summarize(pairs),
        position_bins={f"{a}-{b}": summarize([(x, y) for x, y in pairs if a <= x["position"] < b]) for a, b in ranges},
    )


def capture(engine, tokens, original, output):
    # Different contents and lengths, each with independent positions from zero.
    sequences = [list(tokens), list(tokens[97:4196])]
    root = StoragePath(output)
    artifacts = {}
    for label in ("serial", "repeat"):
        for idx, sequence in enumerate(sequences):
            start = time.monotonic()
            result = engine.score_packed_token_logprobs([sequence], 1)[0]
            artifact = _artifact(sequence, result, f"303-sequence-{idx}", time.monotonic() - start)
            artifact["query_layout"] = engine.last_packed_score_layout
            artifacts[f"{label}-{idx}"] = artifact
            (root / f"packed-{label}-{idx}.json").write_text(json.dumps(artifact, indent=2))
    start = time.monotonic()
    packed = engine.score_packed_token_logprobs(sequences, 1)
    elapsed = time.monotonic() - start
    for idx, (sequence, result) in enumerate(zip(sequences, packed, strict=True)):
        artifact = _artifact(sequence, result, f"303-sequence-{idx}", elapsed)
        artifact["query_layout"] = engine.last_packed_score_layout
        artifacts[f"batch-{idx}"] = artifact
        (root / f"packed-batch-{idx}.json").write_text(json.dumps(artifact, indent=2))
    report = {}
    for idx in range(2):
        report[f"serial_repeat_{idx}"] = compare(artifacts[f"repeat-{idx}"], artifacts[f"serial-{idx}"])
        report[f"packed_vs_serial_{idx}"] = compare(artifacts[f"batch-{idx}"], artifacts[f"serial-{idx}"])
    report["serial_vs_original_0"] = compare(
        artifacts["serial-0"], _artifact(sequences[0], original, "303-sequence-0", 0)
    )
    report["second_sequence_note"] = "Offset excerpt of303, with independent cache ownership; not independent text."
    report["note"] = (
        "Numerical acceptance requires external pooled/binned review against the oracle band before evaluation."
    )
    (root / "packed-gate-comparisons.json").write_text(json.dumps(report, indent=2))
    print("PACKED_TEACHER_FORCED_COMPLETE", json.dumps(report), flush=True)
