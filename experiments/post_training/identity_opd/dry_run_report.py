# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the dry run's gates and step-0 measures offline from its durable outputs.

Three independent parts, each optional:

- ``--archive``: the mismatch-probe FineStore archive. Megatron vs vLLM-generation log-probs on the 64 frozen probe
  samples at update 0 (abs p99, abs max, k3, share beyond 2x; the gate), trainer and generation each vs the vLLM
  re-score (reported, no threshold), and the zero-LR no-op on the trainer's own scores (update 0 vs each later
  update). The archive stores chosen-token log-probs only, so a top-1 agreement cannot be computed from it and is
  reported as unavailable.
- ``--source``/``--export``/``--receipts``: the run's input and exported Hugging Face directories, and a new
  directory for two receipts. Every tensor's raw safetensors payload is hashed (the same SHA-256 as the merge and
  re-hash receipts). ``export-audit.json`` holds ``tensors[name].export_sha256`` for every exported tensor;
  ``readback-receipt.json`` holds the same hashes from a second, independent read of the written shards, in the
  re-hash layout, with the source's hashes beside them. The two reads must agree. The report lists names, dtypes and
  every tensor that differs from the source (for the zero-LR dry run: none).
- ``--trajectories``: the run's retained-trajectory root. Per family at one step: counts by thinking mode, response
  lengths, truncation, end-of-turn stops, identity name and creator mentions, and tool-call rates.

Run where the data lives (the 67B weights are read in full).
"""

import argparse
import gzip
import hashlib
import json
import posixpath
import struct
import zipfile
from collections import Counter
from io import BytesIO

import numpy as np
from rigging.filesystem.buckets import filesystem_for

from experiments.post_training.mismatch_probe.metrics import comparison_metrics
from experiments.post_training.mismatch_probe.report import ArchiveData, load_archive

GENERATION = "vllm.generate@0"
TRAINER = "trainer@{update}:native"
RESCORE = "vllm.rescore@{update}"
LOGPROB_P99_MAX = 0.70
LOGPROB_ABS_MAX = 3.5
END_OF_TURN = 128009
NAME = "marin snowball"
CREATOR = "open athena"
TOOL_CALL = "<tool_call>"
CHUNK_BYTES = 64 << 20
HASH_RULE = "sha256 of each raw safetensors payload, streamed in 64 MiB chunks"
EXPORT_AUDIT = "export-audit.json"
READBACK_RECEIPT = "readback-receipt.json"


def _compare(archive: ArchiveData, target: str, reference: str) -> dict[str, float | int]:
    probes = archive.probes
    return comparison_metrics(
        [archive.scores[target][row.sample_id].logprobs for row in probes],
        [archive.scores[reference][row.sample_id].logprobs for row in probes],
        [row.loss_mask for row in probes],
    )


def logprob_gate(archive: ArchiveData) -> dict:
    """Megatron vs vLLM at update 0, plus the zero-LR no-op on the trainer's own scores."""
    updates = sorted({int(label.split("@", 1)[1].split(":", 1)[0]) for label in archive.scores if "@" in label})
    parity = _compare(archive, TRAINER.format(update=0), GENERATION)
    no_op = {
        f"update {update}": _compare(archive, TRAINER.format(update=update), TRAINER.format(update=0))["abs_max"]
        for update in updates
        if update > 0 and TRAINER.format(update=update) in archive.scores
    }
    rescore = RESCORE.format(update=0)
    has_rescore = rescore in archive.scores
    return {
        "samples": len(archive.probes),
        "trainer_vs_generation": parity,
        "trainer_vs_rescore": _compare(archive, TRAINER.format(update=0), rescore) if has_rescore else None,
        "rescore_vs_generation": _compare(archive, rescore, GENERATION) if has_rescore else None,
        "passed": parity["abs_p99"] <= LOGPROB_P99_MAX and parity["abs_max"] <= LOGPROB_ABS_MAX,
        "thresholds": {"abs_p99": LOGPROB_P99_MAX, "abs_max": LOGPROB_ABS_MAX},
        "trainer_abs_max_vs_update_0": no_op,
        "top1_agreement": "unavailable: the probe archive stores chosen-token log-probs only",
    }


def _read(handle, offset: int, length: int) -> bytes:
    handle.seek(offset)
    data = handle.read(length)
    if len(data) != length:
        raise ValueError(f"short read at {offset}: {len(data)} of {length} bytes")
    return data


def tensor_hashes(hf_dir: str) -> dict[str, dict]:
    """Hash every tensor's stored bytes, shard by shard, from a Hugging Face safetensors directory."""
    fs, root = filesystem_for(hf_dir.rstrip("/"))
    with fs.open(posixpath.join(root, "model.safetensors.index.json"), "rb") as handle:
        shards = sorted(set(json.load(handle)["weight_map"].values()))
    result = {}
    for shard in shards:
        with fs.open(posixpath.join(root, shard), "rb", block_size=CHUNK_BYTES) as handle:
            (header_length,) = struct.unpack("<Q", _read(handle, 0, 8))
            header = json.loads(_read(handle, 8, header_length))
            header.pop("__metadata__", None)
            base = 8 + header_length
            for name, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
                begin, end = entry["data_offsets"]
                digest = hashlib.sha256()
                handle.seek(base + begin)
                remaining = end - begin
                while remaining:
                    chunk = handle.read(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ValueError(f"{shard}: {name} ends early")
                    digest.update(chunk)
                    remaining -= len(chunk)
                if name in result:
                    raise ValueError(f"tensor {name} appears in two shards")
                result[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "export_sha256": digest.hexdigest()}
    return result


def compare_hashes(before: dict[str, dict], after: dict[str, dict]) -> dict:
    shared = sorted(set(before) & set(after))
    return {
        "tensors_before": len(before),
        "tensors_after": len(after),
        "dtypes_before": dict(sorted(Counter(entry["dtype"] for entry in before.values()).items())),
        "dtypes_after": dict(sorted(Counter(entry["dtype"] for entry in after.values()).items())),
        "only_before": sorted(set(before) - set(after)),
        "only_after": sorted(set(after) - set(before)),
        "differing": [name for name in shared if before[name] != after[name]],
        "all_equal": before == after,
    }


def _write_new(url: str, payload: dict) -> str:
    fs, path = filesystem_for(url)
    if fs.exists(path):
        raise FileExistsError(f"{url} already exists")
    data = json.dumps(payload, indent=1, sort_keys=True).encode()
    fs.makedirs(posixpath.dirname(path), exist_ok=True)
    fs.pipe_file(path, data)
    return hashlib.sha256(data).hexdigest()


def export_receipts(source_root: str, export_root: str, receipts: str) -> dict:
    """Hash the source and the export, re-read the export, write both receipts, and compare with the source."""
    source = tensor_hashes(source_root)
    exported = tensor_hashes(export_root)
    readback = tensor_hashes(export_root)
    if readback != exported:
        changed = sorted(name for name in exported if readback.get(name) != exported[name])
        raise ValueError(f"export changed between reads: {changed[:5]}")
    audit_sha = _write_new(
        f"{receipts}/{EXPORT_AUDIT}",
        {"export_root": export_root, "hash": HASH_RULE, "tensor_count": len(exported), "tensors": exported},
    )
    readback_sha = _write_new(
        f"{receipts}/{READBACK_RECEIPT}",
        {
            "export_root": export_root,
            "source_root": source_root,
            "hash": HASH_RULE,
            "tensor_count": len(readback),
            "tensors": readback,
            "slice_stats": {},
            "source_tensors": {
                name: {"dtype": entry["dtype"], "shape": entry["shape"], "source_native_sha256": entry["export_sha256"]}
                for name, entry in source.items()
            },
        },
    )
    return {
        **compare_hashes(source, exported),
        "export_audit": {"path": f"{receipts}/{EXPORT_AUDIT}", "sha256": audit_sha},
        "readback_receipt": {"path": f"{receipts}/{READBACK_RECEIPT}", "sha256": readback_sha},
    }


def trajectory_records(root: str, step: int) -> list[dict]:
    """Every retained training record at one global step."""
    fs, base = filesystem_for(root.rstrip("/"))
    pattern = posixpath.join(base, "schema_v*", "archives", "phase=train", f"step={step:08d}", "*.zip")
    records = []
    for path in sorted(fs.glob(pattern)):
        with fs.open(path, "rb") as handle, zipfile.ZipFile(BytesIO(handle.read())) as archive:
            for entry in sorted(archive.namelist()):
                if entry.startswith("records/"):
                    records.append(json.loads(gzip.decompress(archive.read(entry))))
    return records


def _summary(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "p50": float(np.median(array)), "max": float(array.max())}


def family_measures(records: list[dict]) -> dict:
    by_family: dict[str, list[dict]] = {}
    for record in records:
        route = record["trajectory"]["environment_extras"]["teacher_route"]
        by_family.setdefault(route.rsplit("_", 1)[0], []).append(record)
    result = {}
    for family, rows in sorted(by_family.items()):
        texts = [(row["response"]["text"] or "").lower() for row in rows]
        tokens = [row["response"]["token_ids"] for row in rows]
        measures = {
            "samples": len(rows),
            "by_route": dict(
                sorted(Counter(row["trajectory"]["environment_extras"]["teacher_route"] for row in rows).items())
            ),
            "response_tokens": _summary([len(ids) for ids in tokens]),
            "truncated": sum(row["response"]["stop_reason"] == "length" for row in rows) / len(rows),
            "ends_with_end_of_turn": sum(bool(ids) and ids[-1] == END_OF_TURN for ids in tokens) / len(rows),
        }
        if family == "identity":
            measures["says_name"] = sum(NAME in text for text in texts) / len(rows)
            measures["says_creator"] = sum(CREATOR in text for text in texts) / len(rows)
        if family == "tool":
            measures["no_tool_call"] = sum(TOOL_CALL not in text for text in texts) / len(rows)
        result[family] = measures
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", help="mismatch-probe FineStore archive URI")
    parser.add_argument("--source", help="Hugging Face directory of the run's input weights")
    parser.add_argument("--export", help="Hugging Face directory of the run's export")
    parser.add_argument("--receipts", help="new directory for export-audit.json and readback-receipt.json")
    parser.add_argument("--trajectories", help="retained-trajectory root")
    parser.add_argument("--step", type=int, default=1, help="global step whose rollouts are measured (1 = step 0)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len({args.source is None, args.export is None, args.receipts is None}) != 1:
        parser.error("--source, --export and --receipts go together")
    report: dict = {}
    if args.archive:
        report["logprob_gate"] = logprob_gate(load_archive(args.archive))
    if args.source:
        report["weights"] = export_receipts(args.source, args.export, args.receipts)
    if args.trajectories:
        report["families"] = family_measures(trajectory_records(args.trajectories, args.step))
    fs, path = filesystem_for(args.out)
    with fs.open(path, "w") as handle:
        handle.write(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({key: report[key].get("passed", report[key].get("all_equal")) for key in report}, default=str))


if __name__ == "__main__":
    main()
