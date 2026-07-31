#!/usr/bin/env python3
"""Recover a SkyRL ``TrainingInputBatch`` from a fused Ray spill object.

This is deliberately a one-shot evidence tool. It runs in the immutable image
that wrote the object, restores the Ray payload into a local plasma store, and
exports safetensors shards plus a content-addressed manifest. Mesh dispatch can
leave a rank-local tensor view backed by the full logical batch's storage; the
optional backing-storage mode reconstructs that batch only after checking every
tensor has the same row-aligned backing dimension.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from math import prod
from pathlib import Path
from typing import Any

import boto3
import ray
import torch
from botocore.config import Config
from safetensors.torch import save_file

CHUNK_BYTES = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/grug-replay-recovery"))
    parser.add_argument("--upload-prefix")
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Verify and summarize the restored batch without writing shards or a manifest.",
    )
    parser.add_argument(
        "--use-backing-storage",
        action="store_true",
        help="Recover the full logical batch from checked backing storages before export.",
    )
    parser.add_argument("--keep-payload", action="store_true")
    return parser.parse_args()


def s3_client():
    endpoint = os.environ.get("CW_S3_ENDPOINT")
    access_key = os.environ.get("CW_KEY_ID")
    secret_key = os.environ.get("CW_KEY_SECRET")
    kwargs: dict[str, Any] = {
        "config": Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    }
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def ranged_bytes(client, bucket: str, key: str, start: int, size: int) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{start + size - 1}")
    value = response["Body"].read()
    if len(value) != size:
        raise RuntimeError(f"short S3 range read: wanted {size}, got {len(value)}")
    return value


def download_payload(client, bucket: str, key: str, offset: int, destination: Path) -> dict[str, Any]:
    header = ranged_bytes(client, bucket, key, offset, 24)
    owner_len = int.from_bytes(header[0:8], "little")
    metadata_len = int.from_bytes(header[8:16], "little")
    payload_len = int.from_bytes(header[16:24], "little")
    if owner_len > 1024 or metadata_len > 1024 or payload_len <= 0:
        raise RuntimeError(f"implausible spill header owner={owner_len} metadata={metadata_len} payload={payload_len}")
    owner_start = offset + 24
    owner = ranged_bytes(client, bucket, key, owner_start, owner_len)
    metadata = ranged_bytes(client, bucket, key, owner_start + owner_len, metadata_len)
    if metadata != b"PYTHON":
        raise RuntimeError(f"expected Ray PYTHON metadata, got {metadata!r}")

    payload_start = owner_start + owner_len + metadata_len
    response = client.get_object(
        Bucket=bucket,
        Key=key,
        Range=f"bytes={payload_start}-{payload_start + payload_len - 1}",
    )
    digest = hashlib.sha256()
    written = 0
    with destination.open("wb") as output:
        while True:
            chunk = response["Body"].read(CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            written += len(chunk)
            if written % (512 * 1024 * 1024) < CHUNK_BYTES:
                print(f"downloaded_bytes={written}", flush=True)
    if written != payload_len:
        raise RuntimeError(f"short payload: wanted {payload_len}, got {written}")
    return {
        "owner_length": owner_len,
        "owner_sha256": hashlib.sha256(owner).hexdigest(),
        "metadata": metadata.decode("ascii"),
        "payload_length": payload_len,
        "payload_sha256": digest.hexdigest(),
        "payload_start": payload_start,
    }


def restore_payload(payload: Path, payload_len: int) -> Any:
    plasma = payload.parent / "plasma"
    spill = payload.parent / "ray-spill"
    plasma.mkdir(exist_ok=True)
    spill.mkdir(exist_ok=True)
    object_store_bytes = max(3_000_000_000, payload_len + 256 * 1024 * 1024)
    ray.init(
        num_cpus=1,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=object_store_bytes,
        object_spilling_directory=str(spill),
        _plasma_directory=str(plasma),
    )
    worker = ray._private.worker.global_worker
    seed_ref = ray.put(None)
    local_owner = worker.core_worker.get_owner_address(seed_ref)
    random_ref = ray.ObjectRef.from_random()
    object_ref = ray.ObjectRef(random_ref.binary(), local_owner, b"grug replay recovery")
    with payload.open("rb") as source:
        worker.core_worker.put_file_like_object(b"PYTHON", payload_len, source, object_ref, local_owner)
    objects = worker.core_worker.get_if_local([object_ref])
    if len(objects) != 1 or objects[0][0] is None:
        raise RuntimeError("restored object is absent from the local plasma store")
    context = worker.get_serialization_context()
    result = context.deserialize_objects(objects, [object_ref], {})[0]
    if isinstance(result, ray.exceptions.RaySystemError):
        raise RuntimeError(f"Ray deserialization failed: {result}")
    return result


def find_tensor_batch(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        tensor_values = [item for item in value.values() if isinstance(item, torch.Tensor)]
        if tensor_values:
            return value
        for item in value.values():
            try:
                return find_tensor_batch(item)
            except LookupError:
                pass
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return find_tensor_batch(item)
            except LookupError:
                pass
    raise LookupError(f"no tensor mapping found under {type(value).__module__}.{type(value).__qualname__}")


def tensor_bytes(tensor: torch.Tensor) -> memoryview:
    cpu = tensor.detach().cpu().contiguous()
    byte_view = cpu.view(torch.uint8).numpy()
    return memoryview(byte_view).cast("B")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def safe_metadata(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return {"type": type(value).__name__}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 160 else {"type": "str", "length": len(value)}
    if isinstance(value, Mapping):
        return {str(key): safe_metadata(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) <= 16:
            return [safe_metadata(item, depth + 1) for item in value]
        return {"type": type(value).__name__, "length": len(value)}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def summarize_batch(batch: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    fields: dict[str, Any] = {}
    canonical = hashlib.sha256()
    batch_size: int | None = None
    for name in sorted(batch):
        value = batch[name]
        if value is None:
            fields[name] = {"value": None}
            canonical.update(f"{name}:none\n".encode())
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch field {name!r} is {type(value)}, not a tensor or None")
        tensor = value.detach().cpu().contiguous()
        if batch_size is None:
            batch_size = int(tensor.shape[0])
        elif int(tensor.shape[0]) != batch_size:
            raise RuntimeError(f"field {name!r} has a different leading dimension")
        header = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "numel": tensor.numel(),
            "nbytes": tensor.numel() * tensor.element_size(),
        }
        raw = tensor_bytes(tensor)
        field_digest = hashlib.sha256(raw).hexdigest()
        header["sha256"] = field_digest
        canonical.update(json.dumps({"name": name, **header}, sort_keys=True).encode())
        canonical.update(b"\n")
        canonical.update(raw)
        header["storage_nbytes"] = tensor.untyped_storage().nbytes()
        header["storage_offset"] = tensor.storage_offset()
        fields[name] = header
    if batch_size is None:
        raise RuntimeError("batch contains no tensors")

    counts: dict[str, int] = {}
    sequences = batch.get("sequences")
    if isinstance(sequences, torch.Tensor):
        counts["allocated_positions"] = sequences.numel()
    for name in ("attention_mask", "loss_mask", "response_mask"):
        value = batch.get(name)
        if isinstance(value, torch.Tensor):
            counts[f"{name}_nonzero"] = int(torch.count_nonzero(value).item())
    return {"batch_size": batch_size, "fields": fields, "counts": counts}, canonical.hexdigest()


def recover_backing_batch(batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the full leading dimension held by row-aligned tensor storages."""

    inferred_sizes: set[int] = set()
    field_evidence: dict[str, Any] = {}
    for name, value in batch.items():
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise TypeError(f"batch field {name!r} is not a batched tensor or None")
        tensor = value.detach().cpu()
        row_numel = prod(tensor.shape[1:])
        if row_numel <= 0 or tensor.stride(0) != row_numel:
            raise RuntimeError(
                f"field {name!r} is not row-contiguous: shape={tuple(tensor.shape)} stride={tensor.stride()}"
            )
        storage_nbytes = tensor.untyped_storage().nbytes()
        if storage_nbytes % tensor.element_size():
            raise RuntimeError(f"field {name!r} storage is not dtype aligned")
        storage_numel = storage_nbytes // tensor.element_size()
        if storage_numel % row_numel or tensor.storage_offset() % row_numel:
            raise RuntimeError(f"field {name!r} storage is not row aligned")
        full_size = storage_numel // row_numel
        row_start = tensor.storage_offset() // row_numel
        row_end = row_start + tensor.shape[0]
        if row_end > full_size:
            raise RuntimeError(f"field {name!r} local view exceeds its backing storage")
        inferred_sizes.add(full_size)
        field_evidence[name] = {
            "full_batch_size": full_size,
            "local_row_start": row_start,
            "local_row_end": row_end,
            "storage_nbytes": storage_nbytes,
        }

    if len(inferred_sizes) != 1:
        raise RuntimeError(f"tensor backing dimensions disagree: {sorted(inferred_sizes)}")
    full_batch_size = inferred_sizes.pop()
    local_batch_size = next(value.shape[0] for value in batch.values() if isinstance(value, torch.Tensor))
    if full_batch_size <= local_batch_size:
        raise RuntimeError(f"backing batch {full_batch_size} is not larger than local batch {local_batch_size}")

    full_batch: dict[str, Any] = {}
    for name, value in batch.items():
        if value is None:
            full_batch[name] = None
            continue
        tensor = value.detach().cpu()
        full_shape = (full_batch_size, *tensor.shape[1:])
        rebuilt = torch.empty(0, dtype=tensor.dtype).set_(
            tensor.untyped_storage(),
            0,
            full_shape,
            tensor.stride(),
        )
        row_start = field_evidence[name]["local_row_start"]
        row_end = field_evidence[name]["local_row_end"]
        if not torch.equal(rebuilt[row_start:row_end], tensor):
            raise RuntimeError(f"field {name!r} backing-storage round trip did not preserve the local view")
        full_batch[name] = rebuilt

    evidence = {
        "full_batch_size": full_batch_size,
        "local_batch_size": local_batch_size,
        "fields": field_evidence,
    }
    return full_batch, evidence


def export_shards(
    client,
    batch: Mapping[str, Any],
    work_dir: Path,
    bucket: str,
    prefix: str | None,
    logical_digest: str,
    num_shards: int,
) -> list[dict[str, Any]]:
    batch_size = next(value.shape[0] for value in batch.values() if isinstance(value, torch.Tensor))
    if batch_size % num_shards:
        raise RuntimeError(f"batch size {batch_size} is not divisible by {num_shards} shards")
    rows_per_shard = batch_size // num_shards
    records: list[dict[str, Any]] = []
    for rank in range(num_shards):
        start = rank * rows_per_shard
        end = start + rows_per_shard
        tensors = {
            name: value[start:end].detach().cpu().contiguous()
            for name, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        filename = f"rank-{rank:02d}-of-{num_shards:02d}.safetensors"
        path = work_dir / filename
        save_file(
            tensors,
            path,
            metadata={
                "logical_batch_sha256": logical_digest,
                "rank": str(rank),
                "start": str(start),
                "end": str(end),
            },
        )
        record = {
            "rank": rank,
            "start": start,
            "end": end,
            "filename": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if prefix:
            key = f"{prefix.rstrip('/')}/{logical_digest}/{filename}"
            client.upload_file(str(path), bucket, key)
            record["s3_uri"] = f"s3://{bucket}/{key}"
            path.unlink()
        records.append(record)
        print(f"exported_rank={rank} bytes={record['bytes']} sha256={record['sha256']}", flush=True)
    return records


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    payload = args.work_dir / "ray-payload.bin"
    client = s3_client()
    source_head = client.head_object(Bucket=args.bucket, Key=args.key)
    spill = download_payload(client, args.bucket, args.key, args.offset, payload)
    print("spill=" + json.dumps(spill, sort_keys=True), flush=True)
    restored = restore_payload(payload, spill["payload_length"])
    batch = find_tensor_batch(restored)
    restored_batch_type = f"{type(batch).__module__}.{type(batch).__qualname__}"
    metadata = safe_metadata(getattr(batch, "metadata", None))
    local_summary, local_digest = summarize_batch(batch)
    backing_evidence = None
    if args.use_backing_storage:
        batch, backing_evidence = recover_backing_batch(batch)
    summary, logical_digest = summarize_batch(batch)
    print(f"restored_type={restored_batch_type}", flush=True)
    print(f"local_batch_sha256={local_digest}", flush=True)
    print("local_summary=" + json.dumps(local_summary, sort_keys=True), flush=True)
    if backing_evidence is not None:
        print("backing_storage=" + json.dumps(backing_evidence, sort_keys=True), flush=True)
    print(f"logical_batch_sha256={logical_digest}", flush=True)
    print("summary=" + json.dumps(summary, sort_keys=True), flush=True)
    print("batch_metadata=" + json.dumps(metadata, sort_keys=True), flush=True)

    if args.summary_only:
        ray.shutdown()
        if not args.keep_payload:
            payload.unlink()
            shutil.rmtree(args.work_dir / "plasma", ignore_errors=True)
            shutil.rmtree(args.work_dir / "ray-spill", ignore_errors=True)
        return

    shards = export_shards(
        client,
        batch,
        args.work_dir,
        args.bucket,
        args.upload_prefix,
        logical_digest,
        args.num_shards,
    )
    manifest = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "logical_batch_sha256": logical_digest,
        "source": {
            "job": args.source_job,
            "image": args.source_image,
            "revision": args.source_revision,
            "bucket": args.bucket,
            "key": args.key,
            "offset": args.offset,
            "object_etag": source_head.get("ETag", "").strip('"'),
            "object_size": source_head["ContentLength"],
            "spill": spill,
            "ray_version": ray.__version__,
        },
        "batch": summary,
        "batch_metadata": metadata,
        "local_batch": {"logical_batch_sha256": local_digest, **local_summary},
        "backing_storage": backing_evidence,
        "shards": shards,
    }
    manifest_path = args.work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha = sha256_file(manifest_path)
    if args.upload_prefix:
        key = f"{args.upload_prefix.rstrip('/')}/{logical_digest}/manifest.json"
        client.upload_file(str(manifest_path), args.bucket, key)
        print(f"manifest_s3_uri=s3://{args.bucket}/{key}", flush=True)
    print(f"manifest_sha256={manifest_sha}", flush=True)
    print("manifest=" + json.dumps(manifest, sort_keys=True), flush=True)

    ray.shutdown()
    if not args.keep_payload:
        payload.unlink()
        shutil.rmtree(args.work_dir / "plasma", ignore_errors=True)
        shutil.rmtree(args.work_dir / "ray-spill", ignore_errors=True)


if __name__ == "__main__":
    main()
