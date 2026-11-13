# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute a bounded panel using a caller-authenticated scoring client."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from experiments.evals.vep.paced import COMPLETION_GAP, DATA_PARALLEL_RANK, SENTINEL_TOLERANCE, PacedClient
from experiments.evals.vep.prepare import rank
from experiments.evals.vep.protocol import AUTOSOMES
from experiments.evals.vep.targets import targets


def run_panel(
    client: PacedClient,
    panels_path: Path,
    reference_summary_path: Path,
    panel: str,
    variant_count: int,
    output: TextIO,
) -> None:
    """Write each successful score immediately; an incomplete run stays partial."""
    panels_bytes = panels_path.read_bytes()
    panels_hash = hashlib.sha256(panels_bytes).hexdigest()
    panels = json.loads(panels_bytes)
    verified = json.loads(reference_summary_path.read_text())
    assert verified["panels_sha256"] == panels_hash
    assert verified["matching_windows"] == verified["unique_windows"]
    assert panel in {"pilot", "validation"}
    rows = panels["panels"][panel]
    assert 0 < variant_count <= len(rows)
    rows = rows[:variant_count]
    assert all(row["chrom"] in AUTOSOMES - {"1"} for row in rows)
    demonstrations = panels["panels"]["demonstrations"]
    assert len(demonstrations) == 2 and sum(row["label"] for row in demonstrations) == 1
    assert all(row["chrom"] == "1" for row in demonstrations)
    # The pilot's representation check uses three positives and three controls,
    # chosen before scoring. Validation only runs the frozen primary methods.
    format_ids = set()
    if panel == "pilot":
        for label in (True, False):
            candidates = sorted(
                (row for row in rows if row["label"] == label),
                key=lambda row: rank(panels["seed"], "format-check", row["id"]),
            )
            format_ids.update(row["id"] for row in candidates[:3])
    client.session.headers["X-data-parallel-rank"] = str(DATA_PARALLEL_RANK)
    client.verify_model()
    client.sentinels = [
        target
        for target in targets(panels["panels"]["pilot"][:1], demonstrations, set())
        if target.strand == "+"
        and ((target.mapping == "none" and target.candidate == "reference") or target.mapping == "primary")
    ]
    client.check_sentinels()
    header = {
        "type": "start",
        "panels_sha256": panels_hash,
        "panel": panel,
        "variant_count": len(rows),
        "max_requests": client.max_requests,
        "max_seconds": client.max_seconds,
        "concurrency": 1,
        "data_parallel_rank": DATA_PARALLEL_RANK,
        "completion_gap_seconds": COMPLETION_GAP,
        "sentinel_tolerance": SENTINEL_TOLERANCE,
        "format_check_ids": sorted(format_ids),
    }
    output.write(json.dumps(header) + "\n")
    output.flush()
    counts = Counter()
    for row in rows:
        for target in targets([row], demonstrations, format_ids):
            result = client.score(target.prefix, target.text, target.single_token)
            item = asdict(target)
            item["input_sha256"] = hashlib.sha256((target.prefix + target.text).encode()).hexdigest()
            del item["prefix"], item["text"]
            output.write(json.dumps({"type": "score", **item, **asdict(result)}) + "\n")
            output.flush()
            counts[target.method] += 1
    client.check_sentinels()
    output.write(json.dumps({"type": "complete", "counts": counts, "requests_sent": client.requests_sent}) + "\n")
    output.flush()
