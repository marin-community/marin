# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["tensorstore==0.1.84"]
# ///

"""Compare uninterrupted and forked Levanter checkpoints byte-for-byte.

Levanter stores every checkpointed state leaf in one logical OCDBT key-value
tree. Comparing all logical keys avoids reconstructing a model-shaped template
and is insensitive to OCDBT's physical object packing.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import tensorstore as ts

REPORT_VERSION = "2026-08-11-exact-state-v2"
DEFAULT_MAX_CONCURRENCY = 4
MAX_REPORTED_MISMATCHES = 20


def _checkpoint_kvstore_spec(checkpoint_root: str) -> dict:
    parsed = urlparse(checkpoint_root)
    if parsed.scheme == "gs":
        if not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"Invalid GCS checkpoint root: {checkpoint_root!r}")
        base = {
            "driver": "gcs",
            "bucket": parsed.netloc,
            "path": parsed.path.strip("/"),
        }
    elif parsed.scheme:
        raise ValueError(f"Unsupported checkpoint scheme: {parsed.scheme!r}")
    else:
        base = {"driver": "file", "path": str(Path(checkpoint_root).resolve())}
    return {"driver": "ocdbt", "base": base}


async def _checkpoint_keys(checkpoint_root: str) -> tuple[ts.KvStore, tuple[bytes, ...]]:
    store = await ts.KvStore.open(_checkpoint_kvstore_spec(checkpoint_root))
    keys = tuple(sorted(await store.list()))
    if not keys:
        raise ValueError(f"Checkpoint has no OCDBT keys: {checkpoint_root}")
    return store, keys


async def compare_checkpoint_pair(
    label: str,
    parent_root: str,
    fork_root: str,
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> dict:
    """Compare every logical OCDBT key in one parent/fork pair."""
    if parent_root == fork_root:
        raise ValueError(f"{label}: parent and fork roots must differ")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")

    parent_store, parent_keys = await _checkpoint_keys(parent_root)
    fork_store, fork_keys = await _checkpoint_keys(fork_root)
    parent_key_set = set(parent_keys)
    fork_key_set = set(fork_keys)
    missing_from_parent = sorted(fork_key_set - parent_key_set)
    missing_from_fork = sorted(parent_key_set - fork_key_set)
    common_keys = sorted(parent_key_set & fork_key_set)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def compare_key(key: bytes) -> tuple[int, dict | None]:
        async with semaphore:
            parent_result, fork_result = await asyncio.gather(parent_store.read(key), fork_store.read(key))
            if str(parent_result.state) != "value" or str(fork_result.state) != "value":
                raise ValueError(f"{label}: failed to read key {key!r}")
            parent_value = parent_result.value
            fork_value = fork_result.value
            if parent_value == fork_value:
                return len(parent_value), None
            return max(len(parent_value), len(fork_value)), {
                "key": key.decode("utf-8", errors="backslashreplace"),
                "parent_bytes": len(parent_value),
                "fork_bytes": len(fork_value),
                "parent_sha256": hashlib.sha256(parent_value).hexdigest(),
                "fork_sha256": hashlib.sha256(fork_value).hexdigest(),
            }

    comparisons = await asyncio.gather(*(compare_key(key) for key in common_keys))
    value_mismatches = [mismatch for _, mismatch in comparisons if mismatch is not None]
    total_bytes = sum(size for size, _ in comparisons)
    exact = not missing_from_parent and not missing_from_fork and not value_mismatches
    return {
        "label": label,
        "parent_root": parent_root,
        "fork_root": fork_root,
        "exact": exact,
        "parent_key_count": len(parent_keys),
        "fork_key_count": len(fork_keys),
        "common_key_count": len(common_keys),
        "compared_bytes": total_bytes,
        "missing_from_parent": [key.decode("utf-8", errors="backslashreplace") for key in missing_from_parent],
        "missing_from_fork": [key.decode("utf-8", errors="backslashreplace") for key in missing_from_fork],
        "value_mismatch_count": len(value_mismatches),
        "value_mismatches": value_mismatches[:MAX_REPORTED_MISMATCHES],
    }


async def _write_report(path: str, payload: bytes) -> None:
    parsed = urlparse(path)
    if parsed.scheme == "gs":
        if not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"Invalid GCS report path: {path!r}")
        store = await ts.KvStore.open({"driver": "gcs", "bucket": parsed.netloc})
        await store.write(parsed.path.strip("/").encode(), payload)
        return
    if parsed.scheme:
        raise ValueError(f"Unsupported report scheme: {parsed.scheme!r}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        required=True,
        metavar=("LABEL", "PARENT_ROOT", "FORK_ROOT"),
        help="Checkpoint pair to compare; repeat for multiple seeds.",
    )
    parser.add_argument("--output", required=True, help="Local or gs:// JSON report path.")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--expected-key-count", type=int)
    parser.add_argument("--expected-compared-bytes", type=int)
    parser.add_argument("--expected-parent-step", type=int)
    parser.add_argument("--expected-fork-step", type=int)
    return parser.parse_args()


def _assert_expected_comparison(comparison: dict, args: argparse.Namespace) -> None:
    if args.expected_key_count is not None:
        observed = (comparison["parent_key_count"], comparison["fork_key_count"])
        expected = (args.expected_key_count, args.expected_key_count)
        if observed != expected:
            raise ValueError(f"{comparison['label']}: checkpoint key counts {observed} != {expected}")
    if args.expected_compared_bytes is not None and comparison["compared_bytes"] != args.expected_compared_bytes:
        raise ValueError(
            f"{comparison['label']}: compared bytes {comparison['compared_bytes']} != " f"{args.expected_compared_bytes}"
        )
    for field, expected_step in (
        ("parent_root", args.expected_parent_step),
        ("fork_root", args.expected_fork_step),
    ):
        if expected_step is not None and not str(comparison[field]).rstrip("/").endswith(f"step-{expected_step}"):
            raise ValueError(f"{comparison['label']}: {field} does not end at step-{expected_step}")


async def _main() -> int:
    args = _parse_args()
    comparisons = []
    for label, parent_root, fork_root in args.pair:
        comparison = await compare_checkpoint_pair(
            label,
            parent_root,
            fork_root,
            max_concurrency=args.max_concurrency,
        )
        _assert_expected_comparison(comparison, args)
        comparisons.append(comparison)
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "exact": all(comparison["exact"] for comparison in comparisons),
        "comparisons": comparisons,
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    await _write_report(args.output, payload)
    print(payload.decode(), end="")
    return 0 if report["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
