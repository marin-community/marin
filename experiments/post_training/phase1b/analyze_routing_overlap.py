#!/usr/bin/env python3
"""Response-token expert-load overlap between B's math and terminal probes."""

import argparse
import json
from pathlib import Path

import numpy as np


def rows_by_id(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return by_id


def load_counts(capture: Path, source: Path, model_id: str, experts: int) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = json.loads((capture / "manifest.json").read_text())
    if manifest["model_id"] != model_id or not manifest["complete"]:
        raise ValueError(f"Unexpected model or incomplete capture: {capture}")
    source_rows = rows_by_id(source)
    records = manifest["records"]
    if set(source_rows) != {record["sample_id"] for record in records}:
        raise ValueError(f"Source and capture sample IDs differ: {capture}")
    counts = None
    layers = None
    response_tokens = 0
    top_k = None
    for record in records:
        row = source_rows[record["sample_id"]]
        if row["prompt_tokens"] + row["completion_tokens"] != record["tokens"]:
            raise ValueError(f"Token length mismatch for {record['sample_id']}")
        with np.load(capture / record["file"]) as stored:
            selected = stored["selected"]
            observed_layers = stored["layers"]
            if layers is None:
                layers = observed_layers
                top_k = selected.shape[-1]
                counts = np.zeros((len(layers), experts), dtype=np.int64)
            if not np.array_equal(layers, observed_layers):
                raise ValueError("Routing layer order differs across sequences")
            if selected.shape != (record["tokens"], len(layers), top_k):
                raise ValueError(f"Selected shape mismatch for {record['sample_id']}")
            response = selected[row["prompt_tokens"] :]
            if response.shape[0] != row["completion_tokens"]:
                raise ValueError(f"Response length mismatch for {record['sample_id']}")
            if (response < 0).any() or (response >= experts).any():
                raise ValueError(f"Invalid routed expert in {record['sample_id']}")
            for column in range(len(layers)):
                counts[column] += np.bincount(response[:, column].reshape(-1), minlength=experts)
            response_tokens += response.shape[0]
    if counts is None or layers is None or top_k is None:
        raise ValueError(f"Empty capture: {capture}")
    if not np.all(counts.sum(axis=1) == response_tokens * top_k):
        raise ValueError("Routed-token counts do not cover all response positions")
    return counts, layers, {"sequences": len(records), "response_tokens": response_tokens, "top_k": top_k}


def overlap(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    intersection = float(np.minimum(p, q).sum())
    mean = (p + q) / 2
    p_nonzero = p > 0
    q_nonzero = q > 0
    jsd_bits = float(
        0.5 * np.sum(p[p_nonzero] * np.log2(p[p_nonzero] / mean[p_nonzero]))
        + 0.5 * np.sum(q[q_nonzero] * np.log2(q[q_nonzero] / mean[q_nonzero]))
    )
    return intersection, 1 - jsd_bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--math-source", type=Path, required=True)
    parser.add_argument("--terminal-source", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    r_counts, r_layers, r_meta = load_counts(
        args.capture_root / "P_R", args.math_source, args.model_id, args.experts
    )
    j_counts, j_layers, j_meta = load_counts(
        args.capture_root / "P_J", args.terminal_source, args.model_id, args.experts
    )
    if not np.array_equal(r_layers, j_layers) or r_meta["top_k"] != j_meta["top_k"]:
        raise ValueError("Task captures have different layer order or top-k")
    per_layer = []
    for column, layer in enumerate(r_layers.tolist()):
        p = r_counts[column] / r_counts[column].sum()
        q = j_counts[column] / j_counts[column].sum()
        intersection, one_minus_jsd = overlap(p, q)
        per_layer.append(
            {"layer": int(layer), "histogram_intersection": intersection, "one_minus_jsd_bits": one_minus_jsd}
        )
    result = {
        "definition": "For each MoE layer, normalize selected-expert counts on B response tokens; O_l=sum_e min(p_l^R(e),p_l^J(e)); report the unweighted layer mean.",
        "jsd_definition": "Jensen-Shannon divergence in base-2 bits, so 1-JSD is in [0,1].",
        "model_id": args.model_id,
        "capture_root": str(args.capture_root),
        "sources": {"P_R": str(args.math_source), "P_J": str(args.terminal_source)},
        "experts": args.experts,
        "P_R": r_meta,
        "P_J": j_meta,
        "mean_histogram_intersection": float(np.mean([row["histogram_intersection"] for row in per_layer])),
        "mean_one_minus_jsd_bits": float(np.mean([row["one_minus_jsd_bits"] for row in per_layer])),
        "per_layer": per_layer,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
