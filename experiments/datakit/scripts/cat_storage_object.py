# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print a storage object to stdout so it can be recovered from an Iris job log.

The CoreWeave object store is reachable from inside the cluster and, for most
people, not from a workstation. That is fine for pipeline data and awkward for a
small artifact — a stats JSON, a calibration, a manifest — that has to be read
somewhere else. This ships it out through the one channel that is always
available: the job log.

The object is gzipped and base64-encoded, then emitted as fixed-width lines
tagged ``OBJ <index> <chunk>`` so the reader can reassemble it after the log
interleaves the lines with everything else:

    iris --cluster=marin job run --target-cluster cw-us-east-02a -- \\
        python -m experiments.datakit.scripts.cat_storage_object --path s3://.../stats.json
    iris --cluster=marin job logs <job> | python -m experiments.datakit.scripts.cat_storage_object --decode > stats.json

Meant for artifacts of a few megabytes at most; anything larger belongs in a job
that reads it in place.
"""

import argparse
import base64
import gzip
import re
import sys

from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3

CHUNK = 2_000
TAG = "OBJ"
LINE = re.compile(rf"{TAG} (\d+) ([A-Za-z0-9+/=]+)\s*$")


def emit(path: str) -> None:
    with StoragePath(path).open("rb") as fh:
        payload = base64.b64encode(gzip.compress(fh.read())).decode()
    print(f"{TAG}-BEGIN {path} chunks={-(-len(payload) // CHUNK)}")
    for i in range(0, len(payload), CHUNK):
        print(f"{TAG} {i // CHUNK} {payload[i : i + CHUNK]}")
    print(f"{TAG}-END {path}")


def decode(stream) -> bytes:
    chunks: dict[int, str] = {}
    for line in stream:
        match = LINE.search(line)
        if match:
            chunks[int(match.group(1))] = match.group(2)
    if not chunks:
        raise ValueError(f"no {TAG} lines on stdin")
    missing = sorted(set(range(max(chunks) + 1)) - set(chunks))
    if missing:
        raise ValueError(f"log is missing chunk(s) {missing[:10]} of {max(chunks) + 1}")
    return gzip.decompress(base64.b64decode("".join(chunks[i] for i in sorted(chunks))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", help="storage path to emit (in-cluster side)")
    p.add_argument("--decode", action="store_true", help="reassemble the object from a log on stdin")
    args = p.parse_args()
    if args.decode:
        sys.stdout.buffer.write(decode(sys.stdin))
        return
    if not args.path:
        raise SystemExit("--path is required unless --decode is given")
    configure_coreweave_s3()
    emit(args.path)


if __name__ == "__main__":
    main()
