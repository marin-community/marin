# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete a verified CPU schema-3 export's portable SkyRL manifest."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from marinskyrl.model_manifest import MODEL_MANIFEST_FILENAME, ModelManifestFile, build_model_manifest
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath


def publish_manifest(root: StoragePath) -> str:
    completion = json.loads((root / "export-manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    if completion["request"]["export_version"] != 3 or config["grugmoe_artifact_schema_version"] != 3:
        raise ValueError("Mochi handoff requires a completed schema-3 export")
    if config["dtype"] != "bfloat16":
        raise ValueError("Mochi handoff requires BF16 weights")

    def shard_entry(record: dict) -> ModelManifestFile:
        path = record["filename"]
        if (root / path).size() != record["bytes"]:
            raise ValueError(f"Exported shard size differs from its receipt: {path}")
        # The completed exporter hashed each local shard before upload. Retain
        # those receipts instead of downloading a terabyte to hash it again.
        return ModelManifestFile(path=path, size=record["bytes"], sha256=record["sha256"])

    with ThreadPoolExecutor(max_workers=12) as pool:
        entries = list(pool.map(shard_entry, completion["shards"]))
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "source.json",
        "export-manifest.json",
        "export-request.json",
    ):
        data = (root / name).read_bytes()
        entries.append(ModelManifestFile(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    # This runtime supports the SFT's TokenizersBackend metadata directly.
    # Building from entries preserves its tokenizer and chat-template bytes.
    manifest = build_model_manifest(tuple(sorted(entries, key=lambda entry: entry.path)), None, None)
    data = (json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
    target = conditional_object(str(root / MODEL_MANIFEST_FILENAME))
    existing = target.read()
    if existing is None:
        target.write(data, expected_version=None)
    elif existing.data != data:
        raise FileExistsError(f"Refusing to replace the model manifest at {root}")
    return manifest.identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    print(publish_manifest(StoragePath(args.hf_root)))


if __name__ == "__main__":
    main()
