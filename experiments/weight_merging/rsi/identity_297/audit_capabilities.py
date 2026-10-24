# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count capability paths in evaluation artifacts without emitting credential values."""

import argparse
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for

PATTERN = re.compile(rb"/proxy/t/[^/\s\"\\]+/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--metadata-only", action="store_true", help="Skip model payloads; scan JSON and text files only."
    )
    args = parser.parse_args()
    fs, prefix = filesystem_for(args.prefix)
    output_fs, output = filesystem_for(args.report)
    report = {"prefix": args.prefix, "scanned": 0, "bytes": 0, "matches": [], "complete": False}
    paths = fs.find(prefix)
    if args.metadata_only:
        paths = [p for p in paths if p.endswith((".json", ".jsonl", ".txt", ".log"))]

    def scan(path: str) -> tuple[str, int, int]:
        payload = fs.cat_file(path)
        size = len(payload)
        if path.endswith(".parquet"):
            table = pq.read_table(io.BytesIO(payload))
            payload = json.dumps(table.to_pylist(), default=str).encode()
        return path, size, len(PATTERN.findall(payload))

    with ThreadPoolExecutor(max_workers=8) as pool:
        for path, size, count in pool.map(scan, paths):
            report["scanned"] += 1
            report["bytes"] += size
            if count:
                report["matches"].append({"path": path, "count": count})
            if report["scanned"] % 25 == 0:
                output_fs.pipe_file(output, json.dumps(report, indent=2).encode())
    report["complete"] = True
    output_fs.pipe_file(output, json.dumps(report, indent=2).encode())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
