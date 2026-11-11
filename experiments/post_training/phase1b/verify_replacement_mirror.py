# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check a regional expert mirror against its pinned HF file and LFS manifest."""

import argparse
import json
from datetime import UTC, datetime

from huggingface_hub import HfApi
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mirror", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()

    manifest_url = f"{args.mirror.rstrip('/')}/.verified-copy-manifest.json"
    filesystem, path = filesystem_for(manifest_url)
    with filesystem.open(path) as handle:
        manifest = json.load(handle)
    if manifest["source"] != f"hf://{args.repo}@{args.revision}" or manifest["destination"] != args.mirror:
        raise ValueError("mirror manifest does not identify the pinned source and destination")
    if manifest["schema_version"] != 1:
        raise ValueError("unexpected verified-copy manifest schema")
    files = {file["path"]: file for file in manifest["files"]}
    if len(files) != manifest["total_files"] or sum(file["size"] for file in files.values()) != manifest["total_bytes"]:
        raise ValueError("mirror manifest has inconsistent file counts or sizes")

    hf_files = {
        item.path: item
        for item in HfApi().list_repo_tree(args.repo, revision=args.revision, recursive=True, expand=True)
    }
    if files.keys() != hf_files.keys():
        raise ValueError(f"mirror/HF inventory differs: {sorted(files.keys() ^ hf_files.keys())[:10]}")
    _, mirror_root = filesystem_for(args.mirror)
    final_names = {
        object_path.removeprefix(f"{mirror_root.rstrip('/')}/")
        for object_path in filesystem.find(mirror_root)
        if object_path != path
    }
    if final_names != files.keys():
        raise ValueError(f"final mirror inventory differs: {sorted(final_names ^ files.keys())[:10]}")
    shard_count = 0
    for name, file in files.items():
        source = hf_files[name]
        if file["size"] != source.size:
            raise ValueError(f"size mismatch: {name}")
        if name.endswith(".safetensors"):
            if source.lfs is None or file["sha256"] != source.lfs.sha256:
                raise ValueError(f"LFS sha256 mismatch: {name}")
            shard_count += 1
        final_info = filesystem.info(f"{mirror_root.rstrip('/')}/{name}")
        if final_info["size"] != file["size"]:
            raise ValueError(f"final mirror object size mismatch: {name}")

    result = {
        "checked_at": datetime.now(UTC).isoformat(),
        "repo": args.repo,
        "revision": args.revision,
        "mirror": args.mirror,
        "files": len(files),
        "bytes": manifest["total_bytes"],
        "lfs_shards_checked": shard_count,
        "sha256_match": True,
    }
    with open(args.out, "x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
