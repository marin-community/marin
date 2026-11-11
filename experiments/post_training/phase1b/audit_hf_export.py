# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit an exported HF model against its source model and checkpoint requests."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

from marinskyrl.model_manifest import read_safetensors_header
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    "BOOL": 1,
}
REQUIRED_METADATA = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
)
SOURCE_IDENTITY_FIELDS = ("model_path", "model_source_identity", "model_source_uri", "num_nodes", "gpus_per_node")
ARCHITECTURE_FIELDS = (
    "model_type",
    "architectures",
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_experts",
    "vocab_size",
)


def read_bytes(uri: str) -> bytes:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "rb") as source:
        return source.read()


def model_schema(root: str, *, allow_missing_index: bool = False) -> dict:
    filesystem, path = filesystem_for(root)
    files = {Path(item["name"]).name: item for item in filesystem.ls(path, detail=True)}
    missing_metadata = set(REQUIRED_METADATA) - files.keys()
    if missing_metadata:
        raise ValueError(f"Missing model metadata at {root}: {sorted(missing_metadata)}")
    metadata = {name: read_bytes(f"{root}/{name}") for name in REQUIRED_METADATA}
    for name in REQUIRED_METADATA:
        if not name.endswith(".json"):
            continue
        json.loads(metadata[name])

    index_name = "model.safetensors.index.json"
    index_present = index_name in files
    if not index_present and not allow_missing_index:
        raise ValueError(f"Missing safetensors index at {root}")
    weight_map = json.loads(read_bytes(f"{root}/{index_name}"))["weight_map"] if index_present else None
    if weight_map is not None and (not isinstance(weight_map, dict) or not weight_map):
        raise ValueError(f"Invalid weight map at {root}")
    shard_names = (
        sorted(set(weight_map.values()))
        if weight_map is not None
        else sorted(name for name in files if name.endswith(".safetensors"))
    )
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in shard_names):
        raise ValueError(f"Invalid shard names at {root}")
    tensor_schema = {}
    shard_summary = []
    for name in shard_names:
        if name not in files or files[name]["size"] <= 8:
            raise ValueError(f"Missing or empty shard {name} at {root}")
        with filesystem.open(f"{path}/{name}", "rb") as source:
            header_bytes, keys = read_safetensors_header(source, name)
        header = json.loads(header_bytes[8:])
        payload_size = files[name]["size"] - len(header_bytes)
        offsets = []
        for key in keys:
            value = header[key]
            shape = value["shape"]
            dtype = value["dtype"]
            start, end = value["data_offsets"]
            if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise ValueError(f"Invalid shape for {key} in {name}")
            elements = 1
            for dimension in shape:
                elements *= dimension
            if dtype not in DTYPE_BYTES or end - start != elements * DTYPE_BYTES[dtype]:
                raise ValueError(f"Invalid dtype/byte span for {key} in {name}")
            if start < 0 or end > payload_size or end < start:
                raise ValueError(f"Out-of-range tensor offsets for {key} in {name}")
            offsets.append((start, end))
            if key in tensor_schema:
                raise ValueError(f"Duplicate tensor {key} in {root}")
            tensor_schema[key] = {"shape": shape, "dtype": dtype}
            if weight_map is not None and weight_map.get(key) != name:
                raise ValueError(f"Index/header disagreement for {key} in {root}")
        ordered = sorted(offsets)
        if ordered and (ordered[0][0] != 0 or ordered[-1][1] != payload_size):
            raise ValueError(f"Uncovered payload in {name}")
        if any(left[1] != right[0] for left, right in pairwise(ordered)):
            raise ValueError(f"Gap or overlap in {name}")
        shard_summary.append({"name": name, "bytes": files[name]["size"], "tensors": len(keys)})
    if weight_map is not None and set(tensor_schema) != set(weight_map):
        raise ValueError(f"Index/header tensor set differs at {root}")
    schema_bytes = json.dumps(tensor_schema, sort_keys=True, separators=(",", ":")).encode()
    return {
        "root": root,
        "config": json.loads(metadata["config.json"]),
        "metadata_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in metadata.items()},
        "tensor_schema": tensor_schema,
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "index_present": index_present,
        "manifest_present": ".marinskyrl-model-manifest.json" in files,
        "shards": shard_summary,
        "total_shard_bytes": sum(shard["bytes"] for shard in shard_summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--final-checkpoint-root", help="Checkpoint root holding step 125 after cross-region recovery.")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-index", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    request = json.loads(read_bytes(f"{args.checkpoint_root}/global_step_{args.step}/hf_export_request.json"))
    final_root = args.final_checkpoint_root or args.checkpoint_root
    final_request = json.loads(read_bytes(f"{final_root}/global_step_125/hf_export_request.json"))
    if request["step"] != args.step or final_request["step"] != 125:
        raise ValueError("Unexpected checkpoint request step")
    if any(request[field] != final_request[field] for field in SOURCE_IDENTITY_FIELDS):
        raise ValueError("Step export and final checkpoint source identities differ")
    if request["model_source_uri"] != args.source:
        raise ValueError("Source model URI differs from checkpoint request")

    export = model_schema(args.export.rstrip("/"), allow_missing_index=args.allow_missing_index)
    source = model_schema(args.source.rstrip("/"))
    missing = set(source["tensor_schema"]) - set(export["tensor_schema"])
    extra = set(export["tensor_schema"]) - set(source["tensor_schema"])
    shared = export["tensor_schema"].keys() & source["tensor_schema"].keys()
    shape_mismatches = [
        key for key in shared if export["tensor_schema"][key]["shape"] != source["tensor_schema"][key]["shape"]
    ]
    dtype_mismatches = [
        key for key in shared if export["tensor_schema"][key]["dtype"] != source["tensor_schema"][key]["dtype"]
    ]
    if missing or extra or shape_mismatches:
        raise ValueError(
            f"Tensor schema mismatch: missing={len(missing)} extra={len(extra)} shape={len(shape_mismatches)}"
        )
    if any(export["config"].get(field) != source["config"].get(field) for field in ARCHITECTURE_FIELDS):
        raise ValueError("Export model architecture differs from its source model")
    if export["metadata_sha256"]["tokenizer.json"] != source["metadata_sha256"]["tokenizer.json"]:
        raise ValueError("Export tokenizer differs from its source model")
    export_template = read_bytes(f"{args.export.rstrip('/')}/chat_template.jinja")
    source_template = read_bytes(f"{args.source.rstrip('/')}/chat_template.jinja")
    if export_template.rstrip(b"\n") != source_template.rstrip(b"\n"):
        raise ValueError("Export chat template differs from its source model")

    record = {
        "status": "verified_by_output" if export["index_present"] and export["manifest_present"] else "incomplete",
        "step": args.step,
        "export_uri": args.export,
        "source_model_uri": args.source,
        "source_identity": request["model_source_identity"],
        "final_checkpoint_request": f"{final_root}/global_step_125/hf_export_request.json",
        "request_status": request["status"],
        "shard_count": len(export["shards"]),
        "index_present": export["index_present"],
        "manifest_present": export["manifest_present"],
        "tensor_count": len(export["tensor_schema"]),
        "total_shard_bytes": export["total_shard_bytes"],
        "tensor_schema_sha256": export["schema_sha256"],
        "source_schema_sha256": source["schema_sha256"],
        "architecture": {field: export["config"].get(field) for field in ARCHITECTURE_FIELDS},
        "chat_template_equal_ignoring_final_newline": True,
        "dtype_mismatches": [
            {
                "tensor": key,
                "export": export["tensor_schema"][key]["dtype"],
                "source": source["tensor_schema"][key]["dtype"],
            }
            for key in sorted(dtype_mismatches)
        ],
        "metadata_sha256": export["metadata_sha256"],
        "shards": export["shards"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in record.items() if key not in ("shards", "dtype_mismatches")}
            | {"dtype_mismatch_count": len(dtype_mismatches)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
