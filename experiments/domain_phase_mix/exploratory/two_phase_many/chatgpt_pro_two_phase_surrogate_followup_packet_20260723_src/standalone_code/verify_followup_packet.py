# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
# ]
# ///
"""Verify the follow-up packet's integrity and canonical data invariants."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt"}
FORBIDDEN = (
    "/" + "Users/",
    "pinlin_" + "calvin_xu",
    "plambda" + "four",
    "ANTHROPIC_" + "API_KEY",
    "OPENAI_" + "API_KEY",
    "WANDB_" + "API_KEY",
    "HF_" + "TOKEN",
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weights(frame: pd.DataFrame, phase: int, domains: list[str]) -> np.ndarray:
    columns = [f"phase_{phase}_weight::{domain}" for domain in domains]
    return frame[columns].to_numpy(float)


def verify_checksums() -> None:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Size mismatch: {record['path']}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {record['path']}")


def verify_data() -> None:
    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    datasets = catalog["datasets"]
    domains = list(datasets["delphi_3e18_two_phase_fit"]["domains"])
    if len(domains) != 39:
        raise ValueError(f"Expected 39 buckets, found {len(domains)}")

    frames: dict[str, pd.DataFrame] = {}
    for dataset, spec in datasets.items():
        frame = pd.read_csv(ROOT / spec["path"], low_memory=False)
        frames[dataset] = frame
        if len(frame) != spec["row_count"]:
            raise ValueError(f"Row-count mismatch for {dataset}")
        for phase in (0, 1):
            phase_weights = weights(frame, phase, domains)
            if np.any(phase_weights < -1e-12):
                raise ValueError(f"Negative weight in {dataset}, phase {phase}")
            if not np.allclose(phase_weights.sum(axis=1), 1.0, atol=1e-8):
                raise ValueError(f"Unnormalized weights in {dataset}, phase {phase}")
        for target in ("uncheatable_bpb", "table9_macro_bpb"):
            if not frame[target].notna().all():
                raise ValueError(f"Incomplete {target} in {dataset}")

    one = frames["delphi_3e18_one_phase_fit"]
    two = frames["delphi_3e18_two_phase_fit"]
    if len(one) != 280 or len(two) != 280:
        raise ValueError("Expected 280 rows in each matched fit table")
    if not np.array_equal(one["group_id"].astype(str), two["group_id"].astype(str)):
        raise ValueError("Fit pair IDs are not aligned")

    spec = datasets["delphi_3e18_two_phase_fit"]
    c0 = np.asarray(spec["c0"], dtype=float)
    c1 = np.asarray(spec["c1"], dtype=float)
    alpha0_by_bucket = c0 / (c0 + c1)
    alpha0 = float(np.median(alpha0_by_bucket))
    if not np.allclose(alpha0_by_bucket, alpha0, atol=1e-10):
        raise ValueError("Phase fractions differ across buckets")
    alpha1 = 1.0 - alpha0

    one0 = weights(one, 0, domains)
    one1 = weights(one, 1, domains)
    two0 = weights(two, 0, domains)
    two1 = weights(two, 1, domains)
    if not np.allclose(one0, one1, atol=5e-10):
        raise ValueError("One-phase fit table is not phase tied")
    if not np.allclose(alpha0 * two0 + alpha1 * two1, one0, atol=5e-10):
        raise ValueError("Fit pairs are not aggregate matched")


def verify_privacy() -> None:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="ignore")
        if any(value in text for value in FORBIDDEN) or EMAIL.search(text):
            failures.append(str(path.relative_to(ROOT)))
    if failures:
        raise ValueError(f"Privacy audit failed: {failures[:20]}")


def main() -> None:
    verify_checksums()
    verify_data()
    verify_privacy()
    metadata = json.loads((ROOT / "PACKET_METADATA.json").read_text())
    print(
        "PASS: "
        f"{metadata['fit_pair_count']} exact fit pairs, "
        f"{metadata['heldout_rows']} heldout observations, "
        f"{metadata['heldout_unique_coordinates']} heldout coordinates"
    )


if __name__ == "__main__":
    main()
