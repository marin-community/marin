# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight-delta audit: zero change outside an explicit tensor slice.

Two steps, two receipts.

``rehash`` (in-region CPU job) reads an HF export's shards from storage. It re-hashes every tensor payload with the
same streaming SHA-256 as the in-job export audits (raw safetensors payload bytes), and computes delta statistics
for the slice tensors against the source export. It hashes the source slice payloads too, so the statistics are
tied to the pinned source hashes.

``audit`` (local CPU) joins the pinned source receipt (per tensor the source master and native SHA-256), the
``rehash`` receipt, and exactly one producer receipt:

- ``--trained-audit``: a training run's ``trained-precision-audit.json`` (per tensor the trained master and export
  SHA-256). Every tensor outside the slice must keep its source master hash, and every slice master must change.
- ``--merge-receipt``: a merge's ``merge-receipt.json`` (per tensor the export SHA-256). There are no masters, so
  this export-only mode checks exported bytes only.

Checks common to both (all fatal):
1. The source receipt's bytes hash to the pin; the producer receipt covers the same 502 tensors.
2. The re-hashed export SHA-256 equals the producer receipt's ``export_sha256`` for every tensor.
3. Every tensor outside the slice has an export SHA-256 equal to its source native SHA-256.
4. The re-hash read of each source slice payload equals the source native SHA-256.

Slice tensors whose exported bytes did not change are listed by name, not fatal.
"""

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

SOURCE_RECEIPT_SHA256 = "af484f49ac2c1d67989495475bcc79bf7519d09281c8087321d906b67139fa97"
TENSOR_COUNT = 502
CHUNK_BYTES = 64 * 1024**2
INDEX_NAME = "model.safetensors.index.json"
BITS = {"BF16": np.uint16, "F32": np.uint32}
GROUPS = (
    ("router_bias", r"\.mlp\.router\.bias$"),
    ("router_weight", r"\.mlp\.router\.weight$"),
    ("gated_norm", r"gated_norm\."),
    ("norm", r"norm\.weight$"),
    ("embedding", r"^model\.embed_tokens\.weight$"),
    ("lm_head", r"^lm_head\.weight$"),
    ("attn_gate", r"\.self_attn\.attn_gate\.weight$"),
    ("attn_qkvo", r"\.self_attn\.[qkvo]_proj\.weight$"),
    ("routed_experts", r"\.mlp\.experts\."),
    ("shared_expert", r"\.shared_expert\."),
)


def slice_names(layers: str, projections: str) -> list[str]:
    first, last = (int(x) for x in layers.split("-"))
    return [
        f"model.layers.{layer}.self_attn.{p}_proj.weight"
        for layer in range(first, last + 1)
        for p in projections.split(",")
    ]


def group_of(name: str) -> str:
    for group, pattern in GROUPS:
        if re.search(pattern, name):
            return group
    raise ValueError(f"Tensor in no group: {name}")


def shard_records(root: str) -> dict[str, dict]:
    """Tensor name -> dtype, shape and absolute payload byte range, from the index and each shard header."""
    fs, path = filesystem_for(root)
    shards = sorted(set(json.loads(fs.cat_file(prefix_join(path, INDEX_NAME)))["weight_map"].values()))
    records = {}
    for shard in shards:
        shard_path = prefix_join(path, shard)
        with fs.open(shard_path, "rb", cache_type="none") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(length))
        header.pop("__metadata__", None)
        for name, entry in header.items():
            if name in records:
                raise ValueError(f"Tensor in two shards: {name}")
            start, end = entry["data_offsets"]
            records[name] = {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "path": shard_path,
                "start": 8 + length + start,
                "end": 8 + length + end,
            }
    return records


def stream_payload(fs, record: dict, *, digest=None) -> bytes | None:
    """Read one payload in chunks; update ``digest`` if given, otherwise return the bytes."""
    parts = []
    with fs.open(record["path"], "rb", cache_type="none", block_size=CHUNK_BYTES) as f:
        f.seek(record["start"])
        remaining = record["end"] - record["start"]
        while remaining:
            chunk = f.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError(f"Short read: {record['path']}")
            remaining -= len(chunk)
            if digest is None:
                parts.append(chunk)
            else:
                digest.update(chunk)
    return None if digest is not None else b"".join(parts)


def payload_sha256(fs, record: dict) -> str:
    digest = hashlib.sha256()
    stream_payload(fs, record, digest=digest)
    return digest.hexdigest()


def file_sha256(fs, path: str) -> str:
    digest = hashlib.sha256()
    with fs.open(path, "rb", cache_type="none", block_size=CHUNK_BYTES) as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(root: str) -> dict:
    """Index sha256 plus size and sha256 of every shard the index names, read in full."""
    fs, path = filesystem_for(root)
    index = fs.cat_file(prefix_join(path, INDEX_NAME))
    shards = sorted(set(json.loads(index)["weight_map"].values()))
    return {
        "root": root,
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "shards": {
            shard: {"bytes": fs.size(prefix_join(path, shard)), "sha256": file_sha256(fs, prefix_join(path, shard))}
            for shard in shards
        },
    }


def as_float32(payload: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(payload, np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F32":
        return np.frombuffer(payload, np.float32)
    raise ValueError(f"Unsupported dtype {dtype}")


def delta_stats(source: bytes, exported: bytes, dtype: str) -> dict:
    w, t = as_float32(source, dtype).astype(np.float64), as_float32(exported, dtype).astype(np.float64)
    delta = t - w
    norm_w = float(np.linalg.norm(w))
    changed = np.frombuffer(source, BITS[dtype]) != np.frombuffer(exported, BITS[dtype])
    return {
        "max_abs_delta": float(np.abs(delta).max()),
        "mean_abs_delta": float(np.abs(delta).mean()),
        "rms_delta": float(np.sqrt(np.mean(delta**2))),
        "relative_norm": float(np.linalg.norm(delta)) / norm_w if norm_w else None,
        "fraction_changed": float(changed.mean()),
        "elements": int(changed.size),
    }


def rehash(
    export_root: str,
    source_root: str,
    slice_set: set[str],
    *,
    donor_root: str | None = None,
    reference_root: str | None = None,
) -> dict:
    """Re-hash ``export_root``; optionally record the donor's files and the slice hashes of a reference export."""
    exported, source = shard_records(export_root), shard_records(source_root)
    missing = sorted(slice_set - set(exported))
    if missing:
        raise ValueError(f"Slice names not in the export: {missing}")
    export_fs, _ = filesystem_for(export_root)
    source_fs, _ = filesystem_for(source_root)
    tensors = {
        name: {
            "dtype": record["dtype"],
            "shape": record["shape"],
            "export_sha256": payload_sha256(export_fs, record),
        }
        for name, record in sorted(exported.items())
    }
    stats = {}
    for name in sorted(slice_set):
        a, b = source[name], exported[name]
        if (a["dtype"], a["shape"]) != (b["dtype"], b["shape"]):
            raise ValueError(f"Slice tensor dtype or shape differs from source: {name}")
        source_payload, export_payload = stream_payload(source_fs, a), stream_payload(export_fs, b)
        if hashlib.sha256(export_payload).hexdigest() != tensors[name]["export_sha256"]:
            raise ValueError(f"Slice tensor changed between reads: {name}")
        stats[name] = {
            "source_native_sha256": hashlib.sha256(source_payload).hexdigest(),
            **delta_stats(source_payload, export_payload, a["dtype"]),
        }
    result = {
        "export_root": export_root,
        "source_root": source_root,
        "hash": "sha256 of each raw safetensors payload, streamed in 64 MiB chunks",
        "tensor_count": len(tensors),
        "tensors": tensors,
        "slice_stats": stats,
    }
    if donor_root is not None:
        result["donor"] = checkpoint_files(donor_root)
    if reference_root is not None:
        reference = shard_records(reference_root)
        reference_fs, _ = filesystem_for(reference_root)
        result["reference_root"] = reference_root
        result["reference_slice"] = {}
        for name in sorted(slice_set):
            digest = payload_sha256(reference_fs, reference[name])
            result["reference_slice"][name] = {
                "reference_sha256": digest,
                "equal": digest == tensors[name]["export_sha256"],
            }
    return result


def audit(
    source_path: Path,
    rehash_path: Path,
    slice_set: set[str],
    *,
    trained_path: Path | None = None,
    completed_updates: int | None = None,
    merge_path: Path | None = None,
) -> dict:
    if (trained_path is None) == (merge_path is None):
        raise ValueError("Give exactly one of a trained audit and a merge receipt")
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_RECEIPT_SHA256:
        raise SystemExit("Source receipt differs from its pin")
    source = json.loads(source_bytes)["tensors"]
    producer_path = trained_path or merge_path
    producer_bytes = producer_path.read_bytes()
    producer = json.loads(producer_bytes)
    if trained_path is not None:
        if producer.get("source_receipt_sha256") != SOURCE_RECEIPT_SHA256:
            raise SystemExit("Trained audit does not cite the pinned source receipt")
        if producer.get("completed_updates") != completed_updates:
            raise SystemExit(
                f"Trained audit reports {producer.get('completed_updates')} updates, not {completed_updates}"
            )
    elif sorted(producer.get("slice", ())) != sorted(slice_set):
        raise SystemExit("Merge receipt slice differs from the audited slice")
    rehash_bytes = rehash_path.read_bytes()
    recomputed = json.loads(rehash_bytes)
    tensors = producer["tensors"]
    if len(source) != TENSOR_COUNT or set(source) != set(tensors) or set(source) != set(recomputed["tensors"]):
        raise SystemExit(f"Source, producer and rehash receipts do not cover the same {TENSOR_COUNT} tensors")
    missing = sorted(slice_set - set(source))
    if missing:
        raise SystemExit(f"Slice names not in the checkpoint: {missing}")
    if set(recomputed["slice_stats"]) != slice_set:
        raise SystemExit("Rehash receipt statistics do not cover exactly the slice")

    rows, failures = {}, []
    for name in sorted(source):
        in_slice = name in slice_set
        rehashed = recomputed["tensors"][name]["export_sha256"]
        row = {
            "class": "trained-in-slice" if in_slice else "frozen-outside-slice",
            "group": group_of(name),
            "dtype": recomputed["tensors"][name]["dtype"],
            "shape": recomputed["tensors"][name]["shape"],
            "source_native_sha256": source[name]["native_sha256"],
            "export_sha256": tensors[name]["export_sha256"],
            "rehash_export_sha256": rehashed,
            "export_equal": rehashed == source[name]["native_sha256"],
        }
        if trained_path is not None:
            row["source_master_sha256"] = source[name]["master_sha256"]
            row["arm_master_sha256"] = tensors[name]["master_sha256"]
            row["master_equal"] = row["arm_master_sha256"] == row["source_master_sha256"]
        unchanged = row["export_equal"] and row.get("master_equal", True)
        if rehashed != tensors[name]["export_sha256"]:
            failures.append({"tensor": name, "reason": "recomputed export hash differs from the producer receipt"})
        if in_slice:
            stats = dict(recomputed["slice_stats"][name])
            if stats.pop("source_native_sha256") != row["source_native_sha256"]:
                failures.append({"tensor": name, "reason": "source slice read differs from the source receipt"})
            row.update(stats)
            if row.get("master_equal"):
                failures.append({"tensor": name, "reason": "slice master unchanged"})
        elif not unchanged:
            changed = [k.removesuffix("_equal") for k in ("master_equal", "export_equal") if row.get(k) is False]
            failures.append({"tensor": name, "reason": "frozen tensor changed: " + ", ".join(changed)})
        rows[name] = row

    for name, ref in sorted(recomputed.get("reference_slice", {}).items()):
        rows[name]["reference_equal"] = ref["equal"]
        if not ref["equal"]:
            failures.append({"tensor": name, "reason": "slice export differs from the reference export"})

    frozen = [r for r in rows.values() if r["class"] == "frozen-outside-slice"]
    sliced = {n: r for n, r in rows.items() if r["class"] == "trained-in-slice"}
    groups = {}
    for r in frozen:
        g = groups.setdefault(r["group"], {"frozen": 0, "frozen_changed": 0})
        g["frozen"] += 1
        g["frozen_changed"] += not (r["export_equal"] and r.get("master_equal", True))
    counts = {
        "frozen": len(frozen),
        "frozen_changed": sum(g["frozen_changed"] for g in groups.values()),
        "slice": len(sliced),
        "slice_export_unchanged": sum(r["export_equal"] for r in sliced.values()),
        "rehash_mismatch": sum(r["rehash_export_sha256"] != r["export_sha256"] for r in rows.values()),
    }
    if "reference_slice" in recomputed:
        counts["slice_reference_equal"] = sum(r["equal"] for r in recomputed["reference_slice"].values())
    counts["frozen_equal"] = counts["frozen"] - counts["frozen_changed"]
    if trained_path is not None:
        counts["slice_master_changed"] = sum(not r["master_equal"] for r in sliced.values())
    return {
        "verdict": "FAIL" if failures else "PASS",
        "mode": "trained" if trained_path is not None else "export-only",
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "producer_receipt_sha256": hashlib.sha256(producer_bytes).hexdigest(),
        "rehash_receipt_sha256": hashlib.sha256(rehash_bytes).hexdigest(),
        "export_root": recomputed["export_root"],
        "completed_updates": completed_updates,
        "slice": sorted(slice_set),
        "slice_export_unchanged": sorted(n for n, r in sliced.items() if r["export_equal"]),
        "counts": counts,
        "frozen_groups": dict(sorted(groups.items())),
        "failures": failures,
        "tensors": rows,
    }


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rehash_parser, audit_parser = sub.add_parser("rehash"), sub.add_parser("audit")
    for p in (rehash_parser, audit_parser):
        p.add_argument("--layers", required=True, help="inclusive range, e.g. 18-23")
        p.add_argument("--projections", required=True, help="comma list, e.g. q,k,v,o")
    rehash_parser.add_argument("--export-root", required=True)
    rehash_parser.add_argument("--source-root", required=True)
    rehash_parser.add_argument("--out", required=True, help="local path or object-store URL; must not exist")
    rehash_parser.add_argument("--donor-root", help="record this checkpoint's index and per-shard sha256")
    rehash_parser.add_argument("--reference-root", help="compare slice payloads with this export's")
    audit_parser.add_argument("--out", type=Path, required=True)
    audit_parser.add_argument("--source-receipt", type=Path, required=True)
    audit_parser.add_argument("--rehash", type=Path, required=True)
    producer = audit_parser.add_mutually_exclusive_group(required=True)
    producer.add_argument("--trained-audit", type=Path)
    producer.add_argument("--merge-receipt", type=Path)
    audit_parser.add_argument("--completed-updates", type=int)
    args = parser.parse_args()
    slice_set = set(slice_names(args.layers, args.projections))
    if args.command == "rehash":
        fs, out = filesystem_for(args.out)
        if fs.exists(out):
            raise SystemExit(f"{args.out} exists")
        result = rehash(
            args.export_root,
            args.source_root,
            slice_set,
            donor_root=args.donor_root,
            reference_root=args.reference_root,
        )
        fs.pipe_file(out, (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
        print(json.dumps({"tensor_count": result["tensor_count"], "slice_stats": len(result["slice_stats"])}))
        return
    if args.trained_audit is not None and args.completed_updates is None:
        parser.error("--trained-audit needs --completed-updates")
    result = audit(
        args.source_receipt,
        args.rehash,
        slice_set,
        trained_path=args.trained_audit,
        completed_updates=args.completed_updates,
        merge_path=args.merge_receipt,
    )
    write_new(args.out, result)
    print(json.dumps({"verdict": result["verdict"], **result["counts"], "frozen_groups": result["frozen_groups"]}))
    sys.exit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
