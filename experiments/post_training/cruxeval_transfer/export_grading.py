# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export bounded E response text for credential-free local execution grading."""

import argparse
import base64
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.cruxeval_transfer.constants import COLLECTION_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    source = StoragePath(args.source)
    manifest_text = (source / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    complete = json.loads((source / "complete.json").read_text())
    if manifest["requests"] != COLLECTION_SIZE or complete["requests"] != COLLECTION_SIZE:
        raise ValueError("Expected a completed 12800-response collection")
    requests_text = StoragePath(manifest["prompt_source"]).read_text()
    requests = {row["custom_id"]: row for row in map(json.loads, requests_text.splitlines())}
    if set(requests) != set(manifest["prompt_ids"]):
        raise ValueError("Prompt source and collection IDs differ")
    identities = [prompt + f"::{sample}" for prompt in manifest["prompt_ids"] for sample in range(8)]

    def read_response(identity: str) -> dict:
        filename = hashlib.sha256(identity.encode()).hexdigest() + ".json"
        raw = (source / "responses" / filename).read_text()
        record = json.loads(raw)
        if record["custom_id"] != identity or record["status_code"] != 200:
            raise ValueError(f"Invalid response: {identity}")
        choice = record["response"]["choices"][0]
        return {
            "custom_id": identity,
            "prompt_id": record["prompt_id"],
            "sample": record["sample"],
            "task": record["task"],
            "source_id": record["source_id"],
            "response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "finish_reason": choice["finish_reason"],
            "content": choice["message"].get("content") or "",
        }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_text)
        archive.writestr("requests.jsonl", requests_text)
        total = 0
        with ThreadPoolExecutor(16) as pool, archive.open("responses.jsonl", "w") as stream:
            for index, record in enumerate(pool.map(read_response, identities)):
                encoded = (json.dumps(record, separators=(",", ":")) + "\n").encode()
                total += len(encoded)
                if total > 256 * 1024 * 1024:
                    raise ValueError("Grading text exceeds 256 MiB export bound")
                stream.write(encoded)
                if (index + 1) % 1600 == 0:
                    print(f"GRADING_EXPORT_READ {index + 1}/{COLLECTION_SIZE}", flush=True)
    payload = buffer.getvalue()
    if len(payload) > 64 * 1024 * 1024:
        raise ValueError("Compressed grading bundle exceeds 64 MiB")
    encoded = base64.b64encode(payload).decode()
    chunks = [encoded[start : start + 4096] for start in range(0, len(encoded), 4096)]
    print(
        "CRUXEXPORT_META "
        + json.dumps({"bytes": len(payload), "chunks": len(chunks), "sha256": hashlib.sha256(payload).hexdigest()}),
        flush=True,
    )
    for index, chunk in enumerate(chunks):
        print(f"CRUXEXPORT_CHUNK {index} {chunk}", flush=True)
        time.sleep(0.02)
    time.sleep(90)


if __name__ == "__main__":
    main()
