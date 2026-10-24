"""Finish a CoreWeave-S3 merge copy that `fsutil verified-copy` staged but could not promote, then check every
object against the merge's own manifest by sha256.

On CoreWeave S3 the promote step's server-side copy returns an object whose ETag differs from the staged one, so
verified-copy rejects it ("promoted object does not match verified staging object"). Here each remaining staged
object is copied into place, and correctness comes from re-reading every destination object and comparing its
sha256 and size with merge_manifest.json `objects` (which the merge wrote from the source bytes).

Run next to the data (an Iris CPU job on cw-rno2a):
    python finish_copy.py s3://.../<name>.verified-copy-staging/ s3://.../<name>/ --report s3://.../check.json
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.buckets import filesystem_for

CHUNK = 64 * 1024 * 1024


def sha256(fs, path):
    digest = hashlib.sha256()
    with fs.open(path, "rb", block_size=CHUNK) as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging")
    parser.add_argument("destination")
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    fs, staging = filesystem_for(args.staging)
    _, destination = filesystem_for(args.destination)
    staging, destination = staging.rstrip("/"), destination.rstrip("/")

    staged = {p.removeprefix(staging + "/") for p in fs.find(staging)}
    moved = []
    for name in sorted(staged):
        fs.cp_file(f"{staging}/{name}", f"{destination}/{name}")
        moved.append(name)
    fs.invalidate_cache(destination)

    with fs.open(f"{destination}/merge_manifest.json") as handle:
        manifest = json.load(handle)
    expected = {o["name"]: (o["bytes"], o["sha256"]) for o in manifest["objects"]}
    present = {p.removeprefix(destination + "/"): info["size"] for p, info in fs.find(destination, detail=True).items()}

    def check(name):
        size, digest = expected[name]
        if present.get(name) != size:
            return name, f"size {present.get(name)} != {size}"
        actual = sha256(fs, f"{destination}/{name}")
        return name, None if actual == digest else f"sha256 {actual} != {digest}"

    with ThreadPoolExecutor(args.workers) as pool:
        results = dict(pool.map(check, sorted(expected)))
    failures = {name: why for name, why in results.items() if why}
    report = dict(
        destination=args.destination,
        moved_from_staging=moved,
        expected=len(expected),
        matched=len(expected) - len(failures),
        failures=failures,
        not_in_manifest=sorted(set(present) - set(expected)),
    )
    out_fs, out_path = filesystem_for(args.report)
    with out_fs.open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print("TASK289_COPY_CHECK", json.dumps({k: v for k, v in report.items() if k != "moved_from_staging"}), flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} objects failed the manifest check")
    for name in moved:  # Staging copies are redundant once every destination object matched.
        fs.rm(f"{staging}/{name}")


if __name__ == "__main__":
    main()
