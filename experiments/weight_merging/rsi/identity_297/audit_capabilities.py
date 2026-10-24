# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count capability paths in evaluation artifacts without emitting credential values."""

import argparse
import io
import json
import re

import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for

PATTERN = re.compile(rb"/proxy/t/[^/\s\"\\]+/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    fs, prefix = filesystem_for(args.prefix)
    output_fs, output = filesystem_for(args.report)
    report = {"prefix": args.prefix, "scanned": 0, "bytes": 0, "matches": [], "complete": False}
    for path in fs.find(prefix):
        payload = fs.cat_file(path)
        report["scanned"] += 1
        report["bytes"] += len(payload)
        if path.endswith(".parquet"):
            table = pq.read_table(io.BytesIO(payload))
            payload = json.dumps(table.to_pylist(), default=str).encode()
        count = len(PATTERN.findall(payload))
        if count:
            report["matches"].append({"path": path, "count": count})
        if report["scanned"] % 25 == 0:
            output_fs.pipe_file(output, json.dumps(report, indent=2).encode())
    report["complete"] = True
    output_fs.pipe_file(output, json.dumps(report, indent=2).encode())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
