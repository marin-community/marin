# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare selected expert sets on identical frozen token sequences."""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FlipCount:
    pairs: int = 0
    flips: int = 0

    def add(self, values: np.ndarray) -> None:
        self.pairs += values.size
        self.flips += int(values.sum())

    def summary(self) -> dict:
        return {"pairs": self.pairs, "flips": self.flips, "flip_rate": self.flips / self.pairs}


def expert_set_flips(base: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return one disagreement bit per token and layer, ignoring top-k order."""
    if base.shape != candidate.shape or base.ndim != 3:
        raise ValueError("Routing captures must share [tokens, layers, top-k] shape")
    if (base < 0).any() or (candidate < 0).any():
        raise ValueError("Routing comparison contains padding sentinels")
    return (np.sort(base, axis=-1) != np.sort(candidate, axis=-1)).any(axis=-1)


def compare_routes(base_path: Path, candidate_path: Path) -> dict:
    """Aggregate routing flips by layer, base-margin quartile, and token bucket."""
    base = json.loads((base_path / "manifest.json").read_text())
    candidate = json.loads((candidate_path / "manifest.json").read_text())
    base_rows = {row["sample_id"]: row for row in base["records"]}
    candidate_rows = {row["sample_id"]: row for row in candidate["records"]}
    if len(base_rows) != len(base["records"]) or len(candidate_rows) != len(candidate["records"]):
        raise ValueError("Duplicate sequence identities in routing manifest")
    if not base_rows or base_rows.keys() != candidate_rows.keys():
        raise ValueError("Routing captures cover different sequence identities")
    margins = []
    layers = None
    for row in base_rows.values():
        with np.load(base_path / row["file"]) as stored:
            if layers is None:
                layers = stored["layers"]
            if not np.array_equal(layers, stored["layers"]):
                raise ValueError("Base routing layer order changed")
            margins.append(stored["margin"])
    cuts = np.quantile(np.concatenate(margins), [0.25, 0.5, 0.75], axis=0)
    del margins
    totals = FlipCount()
    by_layer = defaultdict(FlipCount)
    by_margin = defaultdict(FlipCount)
    by_bucket = defaultdict(FlipCount)
    by_position = defaultdict(FlipCount)
    tokens = 0
    for sample_id, row in base_rows.items():
        other = candidate_rows[sample_id]
        for field in ("probe_hash", "tokens", "token_type_tags", "position_tags"):
            if row[field] != other[field]:
                raise ValueError(f"{sample_id}: routing {field} differs between captures")
        with np.load(base_path / row["file"]) as reference, np.load(candidate_path / other["file"]) as observed:
            if not np.array_equal(layers, observed["layers"]):
                raise ValueError("Candidate routing layer order differs from base")
            flipped = expert_set_flips(reference["selected"], observed["selected"])
            if flipped.shape != (row["tokens"], len(layers)):
                raise ValueError(f"{sample_id}: route shape disagrees with manifest")
            margin = reference["margin"]
            if margin.shape != flipped.shape or not np.isfinite(margin).all():
                raise ValueError(f"{sample_id}: invalid selection margins")
            # Boundaries come only from B. Tied margins stay in the same bucket.
            quartiles = (margin[None] > cuts[:, None]).sum(axis=0)
        totals.add(flipped)
        tokens += row["tokens"]
        for column, layer in enumerate(layers.tolist()):
            by_layer[str(layer)].add(flipped[:, column])
        for quartile in range(4):
            values = flipped[quartiles == quartile]
            if values.size:
                by_margin[str(quartile)].add(values)
        for tags, destination in ((row["token_type_tags"], by_bucket), (row["position_tags"], by_position)):
            tags = np.asarray(tags)
            for tag in np.unique(tags):
                destination[str(tag)].add(flipped[tags == tag])
    return {
        "definition": "Selected expert set differs from B at the same input token and layer",
        "base_model": base["model_id"],
        "probe_identity": (
            hashlib.sha256(
                json.dumps(sorted((row["sample_id"], row["probe_hash"]) for row in base_rows.values())).encode()
            ).hexdigest()
        ),
        "candidate_model": candidate["model_id"],
        "complete": base["complete"] and candidate["complete"],
        "sequences": len(base_rows),
        "tokens": tokens,
        "layers": layers.tolist(),
        **totals.summary(),
        "by_layer": {key: value.summary() for key, value in by_layer.items()},
        "base_margin_quartiles": {key: value.summary() for key, value in by_margin.items()},
        "base_margin_cutpoints_by_layer": cuts.tolist(),
        "buckets": {key: value.summary() for key, value in by_bucket.items()},
        "position_buckets": {key: value.summary() for key, value in by_position.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare_routes(args.base, args.candidate)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
