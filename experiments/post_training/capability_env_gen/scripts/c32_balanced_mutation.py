#!/usr/bin/env python3
"""Create the c32 balanced deletion/duplication audit candidate.

This program is an input to the remote audit.  It must run only inside the
fresh, network-blocked c32 verifier image because it imports GDAL/OGR and
modifies a generated task artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from osgeo import ogr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _features(layer: object) -> dict[str, list[object]]:
    by_id: dict[str, list[object]] = {}
    layer.ResetReading()
    for feature in layer:
        conduit_id = feature.GetField("conduit_id")
        by_id.setdefault(conduit_id, []).append(feature.Clone())
    return by_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ground_truth = json.loads(args.ground_truth.read_text())
    defects = set(ground_truth["defect_membership"])
    retained = set(ground_truth["retained_ids"])
    locked = set(ground_truth["locked_ids"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.reference, args.output)
    dataset = ogr.Open(str(args.output), 1)
    if dataset is None:
        raise RuntimeError("could not open copied reference GeoPackage for update")
    layer = dataset.GetLayerByName("conduits")
    if layer is None:
        raise RuntimeError("reference GeoPackage has no conduits layer")
    by_id = _features(layer)

    eligible: list[tuple[str, float, object]] = []
    for conduit_id, rows in sorted(by_id.items()):
        if (
            conduit_id in defects
            or conduit_id not in retained
            or conduit_id in locked
            or len(rows) != 1
        ):
            continue
        geometry = rows[0].GetGeometryRef()
        if geometry is None:
            continue
        length = float(geometry.Length())
        if abs(length - 100.0) <= 1e-9:
            eligible.append((conduit_id, length, rows[0]))
    if len(eligible) < 2:
        raise RuntimeError(f"need two clean single-part 100 m conduits, found {len(eligible)}")

    delete_id, delete_length, delete_feature = eligible[0]
    duplicate_id, duplicate_length, duplicate_feature = eligible[1]
    if abs(delete_length - duplicate_length) > 1e-9:
        raise RuntimeError("selected clean conduit lengths are not equal")

    delete_fid = delete_feature.GetFID()
    if layer.DeleteFeature(delete_fid) != ogr.OGRERR_NONE:
        raise RuntimeError(f"failed to delete {delete_id} fid {delete_fid}")

    duplicate = ogr.Feature(layer.GetLayerDefn())
    for index in range(layer.GetLayerDefn().GetFieldCount()):
        duplicate.SetField(index, duplicate_feature.GetField(index))
    part_seq_index = duplicate.GetFieldIndex("part_seq")
    old_part_seq = duplicate.GetField(part_seq_index)
    existing_sequences = [
        row.GetField("part_seq")
        for row in by_id[duplicate_id]
        if row.GetField("part_seq") is not None
    ]
    new_part_seq = max(existing_sequences, default=-1) + 1
    duplicate.SetField(part_seq_index, new_part_seq)
    duplicate.SetGeometry(duplicate_feature.GetGeometryRef().Clone())
    if layer.CreateFeature(duplicate) != ogr.OGRERR_NONE:
        raise RuntimeError(f"failed to duplicate {duplicate_id}")
    duplicate_fid = duplicate.GetFID()
    dataset = None

    report = {
        "schema": "c32-balanced-preservation-mutation-v1",
        "reference_sha256": _sha256(args.reference),
        "candidate_sha256": _sha256(args.output),
        "ground_truth_sha256": _sha256(args.ground_truth),
        "selection_predicate": {
            "retained": True,
            "not_in_defect_membership": True,
            "not_survey_locked": True,
            "reference_part_count": 1,
            "geometry_length_m": 100.0,
            "length_tolerance_m": 1e-9,
        },
        "deleted": {
            "conduit_id": delete_id,
            "fid": delete_fid,
            "length_m": delete_length,
        },
        "duplicated": {
            "conduit_id": duplicate_id,
            "source_fid": duplicate_feature.GetFID(),
            "created_fid": duplicate_fid,
            "old_part_seq": old_part_seq,
            "new_part_seq": new_part_seq,
            "length_m": duplicate_length,
            "geometry_and_attributes_cloned": True,
        },
        "provenance_mutation": "none",
        "net_part_count_delta": 0,
        "net_length_delta_m": duplicate_length - delete_length,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
