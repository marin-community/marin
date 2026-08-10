# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy a fuzzy-dedup attribute tree to a sibling path in the same bucket.

The global fuzzy-dedup step rewrites ``outputs/`` in place when it runs again,
so a continuation past the iteration cap replaces the attributes the previous
cap produced. This script preserves them first.

The copy is server side: source and destination share one bucket, thus the
object store moves the bytes and no data crosses the network.

Run it in region, next to the bucket::

    uv run iris --cluster=marin job run --no-wait \
        --target-cluster cw-us-east-08a --priority interactive \
        --cpu 4 --memory 8g \
        -- python experiments/datakit/scripts/copy_dedup_outputs.py \
            --src s3://marin-us-east-02a/marin/datakit/dedup_709f5997/outputs \
            --dst s3://marin-us-east-02a/marin/datakit/dedup_709f5997/outputs_it10
"""

import argparse
import hashlib
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_COPY_WORKERS = 32
DEFAULT_VERIFY_SAMPLE = 100
PROGRESS_INTERVAL = 500
# Fixed seed so a repeated verification reads the same objects.
VERIFY_SEED = 20260807


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source prefix, for example s3://bucket/path/outputs")
    parser.add_argument("--dst", required=True, help="Destination prefix in the same bucket")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_COPY_WORKERS,
        help=f"Concurrent copy requests. Default: {DEFAULT_COPY_WORKERS}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the object count and size, and copy nothing.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare the two trees instead of copying: paths, per-object sizes, and a hashed sample of bodies.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_VERIFY_SAMPLE,
        help=f"Objects to read and hash on both sides during --verify. Default: {DEFAULT_VERIFY_SAMPLE}.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.sample < 0:
        parser.error("--sample must not be negative")
    if args.src.rstrip("/") == args.dst.rstrip("/"):
        parser.error("--src and --dst must differ")
    return args


def _relative_sizes(listing: dict, root: str) -> dict[str, int]:
    return {path[len(root) :].lstrip("/"): entry.get("size") or 0 for path, entry in listing.items()}


def verify(fs, src_root: str, dst_root: str, sample: int, workers: int) -> None:
    """Compare both trees and raise on the first difference that matters.

    Object counts alone hide a truncated body or a shard copied to the wrong
    name, so this checks the relative-path sets, every object's size, and the
    bytes of a random sample.
    """
    source = _relative_sizes(fs.find(src_root, detail=True), src_root)
    destination = _relative_sizes(fs.find(dst_root, detail=True), dst_root)
    logger.info("Source holds %d objects, destination holds %d", len(source), len(destination))

    missing = sorted(set(source) - set(destination))
    extra = sorted(set(destination) - set(source))
    if missing:
        raise RuntimeError(f"{len(missing)} object(s) missing at the destination, first: {missing[:3]}")
    if extra:
        raise RuntimeError(f"{len(extra)} unexpected object(s) at the destination, first: {extra[:3]}")
    logger.info("Every source path exists at the destination, and nothing extra is present.")

    resized = [key for key, size in source.items() if destination[key] != size]
    if resized:
        raise RuntimeError(f"{len(resized)} object(s) differ in size, first: {resized[:3]}")
    logger.info("All %d objects match in size (%.1f GiB total).", len(source), sum(source.values()) / 1024**3)

    if sample <= 0:
        return
    keys = sorted(source)
    rng = random.Random(VERIFY_SEED)
    chosen = rng.sample(keys, min(sample, len(keys)))

    def digests(key: str) -> tuple[str, str, str]:
        src_hash = hashlib.sha256(fs.cat_file(f"{src_root}/{key}")).hexdigest()
        dst_hash = hashlib.sha256(fs.cat_file(f"{dst_root}/{key}")).hexdigest()
        return key, src_hash, dst_hash

    sampled_bytes = sum(source[key] for key in chosen)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        mismatched = [key for key, src_hash, dst_hash in pool.map(digests, chosen) if src_hash != dst_hash]
    if mismatched:
        raise RuntimeError(f"{len(mismatched)} sampled object(s) differ in content, first: {mismatched[:3]}")
    logger.info("Content matches for %d sampled objects (%.1f MiB read).", len(chosen), sampled_bytes / 1024**2)
    logger.info("Verification passed.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(logging.INFO)

    fs, src_root = url_to_fs(args.src.rstrip("/"))
    _, dst_root = url_to_fs(args.dst.rstrip("/"))

    if args.verify:
        verify(fs, src_root, dst_root, sample=args.sample, workers=args.workers)
        return

    listing = fs.find(src_root, detail=True)
    sources = sorted(listing)
    total_bytes = sum(entry.get("size") or 0 for entry in listing.values())
    logger.info("Source %s holds %d objects, %.1f GiB", src_root, len(sources), total_bytes / 1024**3)
    if not sources:
        raise FileNotFoundError(f"No objects under {src_root}")
    if args.dry_run:
        return

    existing = fs.find(dst_root, detail=True) if fs.exists(dst_root) else {}

    def copy_one(source: str) -> int:
        relative = source[len(src_root) :].lstrip("/")
        target = f"{dst_root}/{relative}"
        size = listing[source].get("size") or 0
        # A rerun after a partial copy skips objects that already match.
        if (existing.get(target) or {}).get("size") == size:
            return 0
        fs.copy(source, target)
        return size

    copied_objects = 0
    copied_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(copy_one, source): source for source in sources}
        for done, future in enumerate(as_completed(futures), start=1):
            copied_bytes += future.result()
            copied_objects += 1
            if done % PROGRESS_INTERVAL == 0 or done == len(sources):
                logger.info("Copied %d/%d objects (%.1f GiB)", done, len(sources), copied_bytes / 1024**3)

    written = fs.find(dst_root, detail=True)
    logger.info(
        "Destination %s holds %d objects, %.1f GiB",
        dst_root,
        len(written),
        sum(entry.get("size") or 0 for entry in written.values()) / 1024**3,
    )
    if len(written) != len(sources):
        raise RuntimeError(f"Copy incomplete: {len(sources)} source objects, {len(written)} at the destination")
    logger.info("Copy complete: %d objects preserved at %s", copied_objects, dst_root)


if __name__ == "__main__":
    main()
