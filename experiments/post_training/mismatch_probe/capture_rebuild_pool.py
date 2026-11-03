# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The canonical routing capture with a fresh worker pool from a chosen group on.

Diagnostics (2026-11-03) localized the held OPD seed-2 step-10 pass's repeated
CUDA/cuBLAS failure (``CUBLAS_STATUS_EXECUTION_FAILED`` in the gated-RMS-norm's
plain torch bf16 linear, ``grug_megatron.py:90``) to ``P_R`` group 40; an
instrumented sweep with identical batches otherwise completed every other
group in one process, and the layout reproduces on both nodes. This entry
point runs the capture the way ``capture_routes`` does with one
coordinator-approved deviation: at ``--rebuild-at SET:GROUP`` it tears the
scoring worker pool down and rebuilds it against the same staged model, so
that dispatch runs in a fresh CUDA/cuBLAS context. Batches, sequence order,
npz names, and the per-set manifest are the canonical capture's own.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np

from experiments.post_training.mismatch_probe.score_probe_sets import (
    forward_groups,
    parse_sets,
    unique_rows,
)
from experiments.post_training.mismatch_probe.scoring_pass import (
    build_score_batch,
    load_probe_sequences,
    run_score,
    sequence_probe_hash,
)

logger = logging.getLogger(__name__)

ERROR_BOUND = 3000


def capture_span(
    policy,
    set_name: str,
    set_path: str,
    out: Path,
    rank_order: Literal["forward", "reverse"],
    start_group: int,
    end_group: int,
    records: list[dict],
    report_path: Path,
) -> None:
    """Capture one set's groups ``[start_group, end_group)`` with this pool.

    Per-group batches, npz names, row validation, and the manifest row layout
    match ``capture_routes``; the driver writes the set's manifest after the
    set's last span because groups may run on two pools. One heartbeat JSON row
    per group is appended to ``report_path`` before the next dispatch, so a
    kill leaves the surviving record in the task's outputs.
    """
    sequences = load_probe_sequences(set_path)
    groups = forward_groups([len(sequence.token_ids) for sequence in sequences])
    (out / set_name).mkdir(parents=True, exist_ok=True)
    span_started = time.monotonic()
    for group_index in range(start_group, end_group):
        group = groups[group_index]
        if rank_order == "reverse":
            real = group[: unique_rows(group)][::-1]
            group = real + [real[-1]] * (len(group) - len(real))
        batch = build_score_batch(
            [sequences[index] for index in group],
            pad_token_id=0,
            gather_ids=None,
            exact_positions=False,
            exact_rate=0.0,
            master_seed=0,
        )
        heartbeat = {
            "set": set_name,
            "group": group_index,
            "rows": len(group),
            "width": int(batch["sequences"].shape[1]),
            "first_sample_id": sequences[group[0]].sample_id,
        }
        dispatch_started = time.monotonic()
        try:
            results, seconds = run_score(policy, batch, top_k=1, micro_batch_size=1)
            heartbeat.update(status="ok", seconds=round(seconds, 1))
        except BaseException as exc:
            heartbeat.update(status="crash", seconds=round(time.monotonic() - dispatch_started, 1))
            heartbeat["error_text"] = str(exc)[-ERROR_BOUND:]
            _append_heartbeat(report_path, heartbeat)
            raise
        _append_heartbeat(report_path, heartbeat)
        for row_index, sequence_index in enumerate(group[: unique_rows(group)]):
            sequence = sequences[sequence_index]
            length = len(sequence.token_ids)
            arrays = {
                "selected": results["router_selected"][row_index, -length:],
                "combine": results["router_combine"][row_index, -length:],
                "margin": results["router_margin"][row_index, -length:],
                "layers": results["router_layers"][row_index],
            }
            if (arrays["selected"] < 0).any() or not all(
                np.isfinite(arrays[key]).all() for key in ("combine", "margin")
            ):
                raise ValueError(f"Invalid or padded route rows for {sequence.sample_id}")
            filename = f"sequence-{sequence_index:05d}.npz"
            np.savez_compressed(out / set_name / filename, **arrays)
            records.append(
                {
                    "file": filename,
                    "sample_id": sequence.sample_id,
                    "probe_hash": sequence_probe_hash(sequence.token_ids),
                    "tokens": length,
                    "token_type_tags": sequence.token_type_tags,
                    "position_tags": sequence.position_tags,
                }
            )
        saved = len({record["file"] for record in records})
        logger.info(
            "Set %s: %d/%d sequences captured overall; group %s (%d rows, width %d) in %.1f s",
            set_name,
            saved,
            len(sequences),
            group_index,
            len(group),
            int(batch["sequences"].shape[1]),
            seconds,
        )
    logger.info("Set %s span [%d,%d) done in %.1f s", set_name, start_group, end_group, time.monotonic() - span_started)


def _append_heartbeat(report_path: Path, row: dict) -> None:
    rows = json.loads(report_path.read_text()) if report_path.exists() else []
    rows.append(row)
    report_path.write_text(json.dumps(rows, indent=1) + "\n")


def _write_manifest(
    model_id: str, rank_order: str, set_name: str, set_path: str, out: Path, records: list[dict]
) -> None:
    """The set's manifest in ``capture_routes``'s exact layout."""
    report = {
        "model_id": model_id,
        "rank_order": rank_order,
        "sequences_path": set_path,
        "complete": True,
        "seconds": 0.0,
        "position_convention": "input token index; all real tokens including context and position zero",
        "selection_margin": "kth minus (k+1)th biased selection logit",
        "records": sorted(records, key=lambda record: record["file"]),
    }
    (out / set_name / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def slabs(sets: list[tuple[str, int]], boundary: tuple[str, int]) -> list[tuple[int, str, int, int]]:
    """Pool-containing dispatch plan over ``[(set_name, group_count)]``.

    Returns ``[(pool, set_name, start_group, end_group)]`` in capture order. The
    first pool covers every set before the boundary set plus the boundary set's
    leading groups; the fresh pool covers the boundary set's remaining groups
    and every later set."""
    boundary_set, boundary_group = boundary
    counts = dict(sets)
    if boundary_set not in counts or not 0 < boundary_group < counts[boundary_set]:
        raise ValueError(f"rebuild boundary {boundary_set}:{boundary_group} does not name an interior group")
    plan: list[tuple[int, str, int, int]] = []
    boundary_reached = False
    for name, count in sets:
        if not boundary_reached:
            if name != boundary_set:
                plan.append((0, name, 0, count))
            else:
                plan.append((0, name, 0, boundary_group))
                plan.append((1, name, boundary_group, count))
                boundary_reached = True
        else:
            plan.append((1, name, 0, count))
    return [slab for slab in plan if slab[3] > slab[2]]


def main() -> None:
    # SkyRL runtime imports only inside the driver: the plan above is pure and
    # the capture machinery runs inside the scoring image.
    import ray  # noqa: PLC0415  # ray is only available in the scoring image

    from experiments.post_training.mismatch_probe.long_context import scoring_worker  # noqa: PLC0415
    from experiments.post_training.mismatch_probe.routing_worker import RoutingWorker  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True, help="Canonical source checkpoint URI, before local staging")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rank-order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--rebuild-at", required=True, help="SET:GROUP — first dispatch of the fresh pool")
    parser.add_argument("--report", required=True, type=Path, help="Heartbeat JSON written per group")
    parser.add_argument("--set", dest="sets", action="append", required=True, help="SET=frozen sequence JSONL")
    args = parser.parse_args()
    boundary_set, boundary_group = (part.strip() for part in args.rebuild_at.split(":", 1))
    boundary_group = int(boundary_group)
    probes = parse_sets(args.sets)
    report_path = args.report
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    counts = {
        name: len(forward_groups([len(sequence.token_ids) for sequence in load_probe_sequences(path)]))
        for name, path in probes.items()
    }
    plan = slabs([(name, counts[name]) for name, _ in probes.items()], (boundary_set, boundary_group))
    logger.info("Slab plan: %s", [(slab[0], slab[1], slab[2], slab[3]) for slab in plan])
    records_by_set = {name: [] for name, _ in probes.items()}
    try:
        for pool, name, start, end in plan:
            with scoring_worker(args.model_path, num_gpus=8, worker_type=RoutingWorker) as policy:
                logger.info("pool %s starts on set %s span [%d,%d)", pool, name, start, end)
                capture_span(
                    policy,
                    name,
                    probes[name],
                    out,
                    args.rank_order,
                    start,
                    end,
                    records_by_set[name],
                    report_path,
                )
                last_pool = max((slab[0] for slab in plan if slab[1] == name), default=None)
                if pool == last_pool:
                    _write_manifest(args.model_id, args.rank_order, name, probes[name], out, records_by_set[name])
                    logger.info("Wrote %s manifest with %d records", name, len(records_by_set[name]))
    except BaseException as exc:
        _append_heartbeat(
            report_path,
            {"set": "driver", "status": "crash", "error_text": str(exc)[-ERROR_BOUND:]},
        )
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
