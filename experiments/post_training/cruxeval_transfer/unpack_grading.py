# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a log-export bundle and adapt it to the existing isolated CruxEval grader."""

import argparse
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from experiments.post_training.cruxeval_transfer.constants import COLLECTION_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = None
    chunks = {}
    for line in args.logs.read_text().splitlines():
        if "CRUXEXPORT_META " in line:
            metadata = json.loads(line.split("CRUXEXPORT_META ", 1)[1])
        match = re.search(r"CRUXEXPORT_CHUNK (\d+) ([A-Za-z0-9+/=]+)", line)
        if match:
            index = int(match[1])
            if index in chunks and chunks[index] != match[2]:
                raise ValueError("Conflicting export chunks")
            chunks[index] = match[2]
    if metadata is None or set(chunks) != set(range(metadata["chunks"])):
        raise ValueError(f"Incomplete export: {len(chunks)} chunks, metadata {metadata}")
    payload = base64.b64decode("".join(chunks[index] for index in range(len(chunks))), validate=True)
    if len(payload) != metadata["bytes"] or hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("Export checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        requests = [json.loads(line) for line in archive.read("requests.jsonl").splitlines()]
        responses = [json.loads(line) for line in archive.read("responses.jsonl").splitlines()]
        manifest = json.loads(archive.read("manifest.json"))
    prompts = {row["custom_id"]: row for row in requests}
    identities = {prompt + f"::{sample}" for prompt in prompts for sample in range(8)}
    if len(responses) != COLLECTION_SIZE or {row["custom_id"] for row in responses} != identities:
        raise ValueError("Expected eight distinct responses per prompt")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "bundle.zip").write_bytes(payload)
    (args.output / "bundle-metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (
        (args.output / "requests.jsonl").open("w") as request_stream,
        (args.output / "raw-responses.jsonl").open("w") as response_stream,
        (args.output / "provenance.jsonl").open("w") as provenance,
    ):
        for row in responses:
            request = prompts[row["prompt_id"]] | {"custom_id": row["custom_id"]}
            request_stream.write(json.dumps(request) + "\n")
            response_stream.write(
                json.dumps(
                    {
                        "custom_id": row["custom_id"],
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {"finish_reason": row["finish_reason"], "message": {"content": row["content"]}}
                                ]
                            },
                        },
                    }
                )
                + "\n"
            )
            provenance.write(json.dumps({key: value for key, value in row.items() if key != "content"}) + "\n")
    print(json.dumps({"responses": len(responses), "prompts": len(prompts), "bundle": metadata}))


if __name__ == "__main__":
    main()
