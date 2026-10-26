# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate complete shipped-stack arms against reviewed references and distinct hardware peers."""

import argparse
import dataclasses
import hashlib
import json
from itertools import combinations
from pathlib import Path

from marin.testing.inference.sequence_parity import (
    AccuracyObservation,
    ScoredSequence,
    ScoredToken,
    SequenceThresholds,
    compare_accuracy,
    compare_sequences,
)
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.serving_parity.settings import TASKS, HardwareKind, frozen_items

BANDS = (0, 1024, 2048, 2304, 2560, 4096, 5279)
THRESHOLDS = SequenceThresholds(pooled_mean=0.025, band_mean=0.04, top1_fraction=0.03)


class IneligibleStack(ValueError):
    """Diagnostic kernels cannot certify the shipped serving stack."""


def provenance(prefix: str) -> dict:
    root = StoragePath(prefix)
    json.loads((root / "complete.json").read_text())
    result = json.loads((root / "provenance.json").read_text())
    HardwareKind(result["arm"]["hardware"])
    if result.get("shipped_stack_eligible") is not True or any(
        patch.get("arithmetic_changed") for patch in result.get("runtime_patches", [])
    ):
        raise IneligibleStack("Experimental or unclassified runtime cannot produce shipped-stack PASS")
    return result


def validate_arms(arms: dict[str, str]) -> None:
    if len(arms) < 3:
        raise ValueError("Require at least three distinct hardware arms")
    prefixes = [str(StoragePath(path)).rstrip("/") for path in arms.values()]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("Hardware arms must use distinct artifact prefixes")
    recorded = []
    for label, path in arms.items():
        hardware = provenance(path)["arm"]["hardware"]
        if label != hardware:
            raise ValueError("Arm label differs from recorded hardware provenance")
        recorded.append(hardware)
    if HardwareKind.H100 not in recorded or len(set(recorded) - {HardwareKind.H100}) < 2:
        raise ValueError("Require H100 plus at least two distinct TPU geometries")
    if len(set(recorded)) != len(recorded):
        raise ValueError("Arms must represent distinct recorded hardware")


def verify_reference(prefix: str, manifest_path: str, manifest_sha256: str) -> None:
    """Verify the reviewed manifest digest and every referenced artifact before scoring."""
    raw = StoragePath(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("Reference manifest differs from the approved digest")
    manifest = json.loads(raw)
    root = StoragePath(prefix)
    if str(root).rstrip("/") != str(StoragePath(manifest["reference_prefix"])).rstrip("/"):
        raise ValueError("Reference prefix differs from the approved manifest")
    required = {"complete.json", "provenance.json", "teacher-1/0.json", *(f"{task}-items.json" for task in TASKS)}
    if not required <= manifest["files"].keys():
        raise ValueError("Reference manifest omits comparison artifacts")
    for name, identity in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Reference artifact name must remain under its prefix")
        content = (root / name).read_bytes()
        if len(content) != identity["bytes"] or hashlib.sha256(content).hexdigest() != identity["sha256"]:
            raise ValueError(f"Reference artifact changed: {name}")


def read_sequence(path: StoragePath) -> ScoredSequence:
    row = json.loads(path.read_text())
    return ScoredSequence(row["case_id"], row["input_sha256"], tuple(ScoredToken(**x) for x in row["tokens"]))


def compare_arms(left: str, right: str) -> dict:
    candidate, reference = StoragePath(left), StoragePath(right)
    a, b = provenance(left), provenance(right)
    for key in ("model", "revision", "manifest_sha256", "evaluator_source"):
        if not a.get(key) or a[key] != b.get(key):
            raise ValueError(f"Cannot compare different or missing {key}")
    if a["arm"]["hardware"] != HardwareKind.H100 and b["arm"]["hardware"] != HardwareKind.H100:
        if a.get("effective_tpu_pin") != b.get("effective_tpu_pin") or not a.get("effective_tpu_pin"):
            raise ValueError("TPU arms use different or missing shipped pins")
    expected = read_sequence(reference / "teacher-1" / "0.json")
    teachers = {}
    for concurrency in (1, 4):
        for index in range(concurrency):
            actual = read_sequence(candidate / f"teacher-{concurrency}" / f"{index}.json")
            teachers[f"{concurrency}/{index}"] = dataclasses.asdict(
                compare_sequences([actual], [expected], boundaries=BANDS, thresholds=THRESHOLDS)
            )
    accuracy = {}
    for task in TASKS:
        expected_count = len(frozen_items(task))

        def items(root: StoragePath, task=task, count=expected_count):
            rows = json.loads((root / f"{task}-items.json").read_text())
            if len(rows) != count:
                raise ValueError("Expected complete frozen sample coverage")
            return [AccuracyObservation(x["id"], x["correct"], x["finish_reason"]) for x in rows]

        accuracy[task] = dataclasses.asdict(compare_accuracy(items(candidate), items(reference), max_z=2))
    failed = any(row["failures"] for row in [*teachers.values(), *accuracy.values()])
    return dict(status="NUMERICAL_FAIL" if failed else "PASS", teacher=teachers, accuracy=accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="HARDWARE=artifact prefix")
    parser.add_argument("--reference", required=True, help="Reviewed immutable reference artifact prefix")
    parser.add_argument("--reference-manifest", required=True, help="Reviewed artifact hash manifest")
    parser.add_argument("--reference-sha256", required=True, help="Explicitly approved manifest digest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    arms = dict(item.split("=", 1) for item in args.arm)
    if len(arms) != len(args.arm):
        raise ValueError("Duplicate arm label")
    reports = {}
    try:
        validate_arms(arms)
        provenance(args.reference)
        verify_reference(args.reference, args.reference_manifest, args.reference_sha256)
    except (ValueError, FileNotFoundError, KeyError) as error:
        reports["inputs"] = dict(
            status="INELIGIBLE" if isinstance(error, IneligibleStack) else "INCOMPLETE", reason=str(error)
        )
    if not reports:
        pairs = [(name, "reference", path, args.reference) for name, path in arms.items()]
        pairs += [(a, b, arms[a], arms[b]) for a, b in combinations(arms, 2)]
        for a, b, left, right in pairs:
            try:
                reports[f"{a} vs {b}"] = compare_arms(left, right)
            except (ValueError, FileNotFoundError, KeyError) as error:
                reports[f"{a} vs {b}"] = dict(status="INCOMPLETE", reason=str(error))
    args.output.write_text(json.dumps(reports, indent=2))
    if any(row["status"] != "PASS" for row in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
