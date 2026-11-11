# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fault-isolating routing-capture probes on named probe-set groups.

The held OPD seed-2 step-10 routing pass failed twice with the same cuBLAS
GEMM tail at a DP-shard boundary on two different nodes, so the unchanged
compute is deterministic at that checkpoint. This driver reproduces the
capture's exact per-group batch construction for a chosen small set of
groups, dispatching each with :func:`capture_routes`'s call parameters and
recording one bounded report row per group before the next. A kill inside a
dispatch leaves the previous rows in the report, so the surviving report
names the last group that completed and the group that died.

No canonical artifacts are touched: results go under ``--out`` only.
"""

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import ray

from experiments.post_training.mismatch_probe.long_context import scoring_worker
from experiments.post_training.mismatch_probe.routing_worker import RoutingWorker
from experiments.post_training.mismatch_probe.score_probe_sets import forward_groups, parse_sets
from experiments.post_training.mismatch_probe.scoring_pass import build_score_batch, load_probe_sequences, run_score

logger = logging.getLogger(__name__)

ERROR_BOUND = 3000


def probe_rows(pol, set_name: str, group_index: int, groups, sequences, out: Path) -> bool:
    """Dispatch one group exactly as :func:`capture_routes` would; return survival."""
    group = groups[group_index]
    batch = build_score_batch(
        [sequences[index] for index in group],
        pad_token_id=0,
        gather_ids=None,
        exact_positions=False,
        exact_rate=0.0,
        master_seed=0,
    )
    lengths = [len(sequences[index].token_ids) for index in group]
    row = {
        "set": set_name,
        "group": group_index,
        "rows": len(group),
        "width": int(batch["sequences"].shape[1]),
        "real_lengths": lengths[: len(group)],
        "sample_ids": [sequences[index].sample_id for index in group],
    }
    started = time.monotonic()
    try:
        results, seconds = run_score(pol, batch, top_k=1, micro_batch_size=1)
        row["status"] = "ok"
        row["seconds"] = round(seconds, 1)
        row["finite_router"] = bool(
            np.isfinite(results["router_selected"]).all() and np.isfinite(results["router_margin"]).all()
        )
        row["shape"] = list(results["router_selected"].shape)
    except BaseException as exc:
        row["status"] = "crash"
        row["seconds"] = round(time.monotonic() - started, 1)
        row["error_type"] = type(exc).__name__
        row["error_text"] = "\n".join(traceback.format_exception(exc))[-ERROR_BOUND:]
    _write_report(out, row)
    logger.info("group result %s", json.dumps(row)[:600])
    return row["status"] == "ok"


def _write_report(out: Path, row: dict) -> None:
    report_path = out / "routing-diag.json"
    rows = []
    if report_path.exists():
        rows = json.loads(report_path.read_text())
    rows.append(row)
    report_path.write_text(json.dumps(rows, indent=1) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-groups", type=int, default=1, help="Leading groups probed per set; 0 means all")
    parser.add_argument("--set", dest="sets", action="append", required=True, help="SET=frozen sequence JSONL")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    probes = parse_sets(args.sets)
    try:
        with scoring_worker(args.model_path, num_gpus=8, worker_type=RoutingWorker) as policy:
            for name, path in probes.items():
                sequences = load_probe_sequences(path)
                groups = forward_groups([len(sequence.token_ids) for sequence in sequences])
                wanted = len(groups) if args.max_groups == 0 else min(len(groups), max(args.max_groups, 1))
                for group_index in range(wanted):
                    alive = probe_rows(policy, name, group_index, groups, sequences, args.out)
                    logger.info("probed group %s/%s/%s", name, group_index, groups[group_index][0])
                    if not alive:
                        return
    except BaseException as exc:
        _write_report(
            args.out,
            {
                "set": "driver",
                "status": "crash",
                "error_type": type(exc).__name__,
                "error_text": ("\n".join(traceback.format_exception(exc)))[-ERROR_BOUND:],
            },
        )
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
