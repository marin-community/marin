# /// script
# requires-python = ">=3.12"
# dependencies = ["google-cloud-storage"]
# ///
"""Preserve the four canceled targets with same-bucket, generation-pinned copies."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import storage


DIRECTORY = Path(__file__).resolve().parent
BUCKET = "marin-us-central1"
PLAN_HASH = "30fc30f1a7e618900a2f1dc40fe5650237b06ea979f2cedfcf8ca664c496114b"
DESTINATION_ROOT = f"experiments/tpp10_domain_sweeps/{PLAN_HASH}/paused_target_checkpoints/20260912"
CHECKSUM_FIELDS = ("size", "crc32c", "md5_hash")


def preserve_checkpoint(checkpoint: dict) -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    source_prefix = checkpoint["checkpoint"].removeprefix(f"{BUCKET}/")
    if not checkpoint["checkpoint"].startswith(f"{BUCKET}/tmp/ttl=14d/checkpoints-temp/"):
        raise ValueError(f"Unexpected source: {checkpoint['checkpoint']}")
    step = checkpoint["metadata"]["step"]
    destination_prefix = f"{DESTINATION_ROOT}/{checkpoint['run_name']}/step-{step}"
    receipts = []
    source_names = set()
    for item in checkpoint["objects"]:
        name = item["name"].removeprefix(f"{BUCKET}/")
        if not name.startswith(source_prefix + "/"):
            raise ValueError(f"Object outside checkpoint: {name}")
        relative_name = name.removeprefix(source_prefix + "/")
        source_names.add(relative_name)
        generation = int(item["generation"])
        source = bucket.blob(name, generation=generation)
        source.reload(timeout=60)
        expected = {"size": item["size"], "crc32c": item["crc32c"], "md5_hash": item.get("md5Hash")}
        if any(getattr(source, key) != expected[key] for key in CHECKSUM_FIELDS):
            raise ValueError(f"Source differs from audited inventory: {name}")
        destination_name = f"{destination_prefix}/{relative_name}"
        destination = bucket.get_blob(destination_name, timeout=60)
        if destination is None:
            destination = bucket.blob(destination_name)
            token = None
            while True:
                token, _, _ = destination.rewrite(
                    source,
                    token=token,
                    if_generation_match=0,
                    if_source_generation_match=generation,
                    timeout=60,
                )
                if token is None:
                    break
            destination.reload(timeout=60)
        if any(getattr(destination, key) != expected[key] for key in CHECKSUM_FIELDS):
            raise ValueError(f"Destination checksum mismatch: {destination_name}")
        receipts.append({
            "relative_name": relative_name,
            "source_uri": f"gs://{BUCKET}/{name}",
            "source_generation": generation,
            "destination_uri": f"gs://{BUCKET}/{destination_name}",
            "destination_generation": int(destination.generation),
            **expected,
        })
    destination_names = {
        blob.name.removeprefix(destination_prefix + "/")
        for blob in client.list_blobs(BUCKET, prefix=destination_prefix + "/", timeout=60)
    }
    if destination_names != source_names:
        raise ValueError(f"Checkpoint object sets differ: {checkpoint['run_name']}")
    metadata = json.loads(bucket.blob(destination_prefix + "/metadata.json").download_as_bytes(timeout=60))
    if metadata != checkpoint["metadata"]:
        raise ValueError(f"Checkpoint metadata changed: {checkpoint['run_name']}")
    print(f"Verified {checkpoint['run_name']}: step {step}, {len(receipts)} objects", flush=True)
    return {
        "run_name": checkpoint["run_name"],
        "percent": checkpoint["percent"],
        "step": step,
        "checkpoint_uri": f"gs://{BUCKET}/{destination_prefix}",
        "metadata": metadata,
        "bytes": sum(item["size"] for item in receipts),
        "objects": receipts,
    }


def main() -> None:
    checkpoints = json.loads((DIRECTORY / "checkpoint_preservation_plan.json").read_text())
    if sorted(row["percent"] for row in checkpoints) != [30, 50, 70, 100]:
        raise ValueError("Preservation scope must be exactly the four canceled targets")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(preserve_checkpoint, checkpoints))
    manifest = {
        "verified_at": datetime.now(UTC).isoformat(),
        "status": "verified",
        "method": "Same-bucket server-side rewrite with pinned source generations and destination create-only preconditions",
        "verification": "Exact relative object set, size, CRC32C, MD5 and checkpoint metadata",
        "training_plan_sha256": PLAN_HASH,
        "checkpoints": results,
        "total_bytes": sum(row["bytes"] for row in results),
        "resume_requires": "Explicit user promotion after reviewing proxy results",
    }
    path = DIRECTORY / "checkpoint_preservation_receipt.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    bucket = storage.Client().bucket(BUCKET)
    seal = bucket.blob(DESTINATION_ROOT + "/preservation_receipt.json")
    previous = bucket.get_blob(seal.name, timeout=60)
    if previous is not None:
        old = json.loads(previous.download_as_bytes(timeout=60))
        if old["checkpoints"] != results:
            raise ValueError("Existing preservation receipt describes different checkpoints")
    else:
        seal.upload_from_string(path.read_bytes(), content_type="application/json", if_generation_match=0, timeout=60)
    print(f"Preserved {manifest['total_bytes']} bytes. Receipt: gs://{BUCKET}/{seal.name}", flush=True)


if __name__ == "__main__":
    main()
