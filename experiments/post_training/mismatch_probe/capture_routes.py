# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture expert selections on frozen probe sequences without changing training."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import ray

from experiments.post_training.mismatch_probe.long_context import scoring_worker
from experiments.post_training.mismatch_probe.routing_worker import RoutingWorker
from experiments.post_training.mismatch_probe.run_score_pass import run_score
from experiments.post_training.mismatch_probe.score_probe_sets import DP_SIZE, forward_groups, parse_sets, unique_rows
from experiments.post_training.mismatch_probe.scoring_pass import build_score_batch, load_probe_sequences

logger = logging.getLogger(__name__)


def capture_routes(
    policy,
    model_id: str,
    sequences_path: str,
    out: Path,
    limit: int | None,
    rank_order: Literal["forward", "reverse"],
) -> None:
    sequences = load_probe_sequences(sequences_path)
    if limit is not None:
        sequences = sequences[:limit]
    groups = forward_groups([len(sequence.token_ids) for sequence in sequences])
    out.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    for group in groups:
        if rank_order == "reverse":
            real = group[: unique_rows(group)][::-1]
            group = real + [real[-1]] * (len(group) - len(real))
        batch_sequences = [sequences[index] for index in group]
        batch = build_score_batch(
            batch_sequences,
            pad_token_id=0,
            gather_ids=None,
            exact_positions=False,
            exact_rate=0.0,
            master_seed=0,
        )
        results, seconds = run_score(policy, batch, top_k=1, micro_batch_size=1)
        for row, index in enumerate(group[: unique_rows(group)]):
            sequence = sequences[index]
            length = len(sequence.token_ids)
            arrays = {
                "selected": results["router_selected"][row, -length:],
                "combine": results["router_combine"][row, -length:],
                "margin": results["router_margin"][row, -length:],
                "layers": results["router_layers"][row],
            }
            if (arrays["selected"] < 0).any() or not all(
                np.isfinite(arrays[key]).all() for key in ("combine", "margin")
            ):
                raise ValueError(f"Invalid or padded route rows for {sequence.sample_id}")
            filename = f"sequence-{index:05d}.npz"
            np.savez_compressed(out / filename, **arrays)
            records.append(
                {
                    "file": filename,
                    "sample_id": sequence.sample_id,
                    "probe_hash": sequence.probe_hash,
                    "tokens": length,
                    "token_type_tags": sequence.token_type_tags,
                    "position_tags": sequence.position_tags,
                }
            )
        logger.info("Captured %d/%d sequences; forward %.1f seconds", len(records), len(sequences), seconds)
    report = {
        "model_id": model_id,
        "rank_order": rank_order,
        "sequences_path": sequences_path,
        "complete": limit is None,
        "seconds": time.monotonic() - started,
        "position_convention": "input token index; all real tokens including context and position zero",
        "selection_margin": "kth minus (k+1)th biased selection logit",
        "records": sorted(records, key=lambda record: record["file"]),
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True, help="Canonical source checkpoint URI, before local staging")
    parser.add_argument("--set", dest="sets", action="append", required=True, help="SET=frozen sequence JSONL")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rank-order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--limit", type=int, help="Pilot only; marks the capture incomplete")
    args = parser.parse_args()
    try:
        with scoring_worker(args.model_path, num_gpus=DP_SIZE, worker_type=RoutingWorker) as policy:
            for name, path in parse_sets(args.sets).items():
                capture_routes(policy, args.model_id, path, args.out / name, args.limit, args.rank_order)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
