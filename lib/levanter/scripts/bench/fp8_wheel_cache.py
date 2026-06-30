# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Stash/fetch the forked jaxlib wheel in object storage so cluster jobs skip the ~11 min build.

The forked-jaxlib build dominates a benchmark job's wall-clock (~11 min build vs ~1.5 min 8-GPU sweep).
Build it once, ``put`` the wheel here, and subsequent jobs ``get`` it and pass ``JAXLIB_WHEEL=`` to
``mixed_fp8_fork_setup.sh`` (which then skips the bazel build) — turning a per-change run into ~3 min.

Storage is the cluster's R2-backed S3 bucket (``MARIN_PREFIX``). In a task container iris injects the R2
creds as ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` + ``AWS_ENDPOINT_URL``, so s3fs picks them up.

    python fp8_wheel_cache.py put /root/jaxsrc/dist/jaxlib-*.whl   # after a build
    python fp8_wheel_cache.py get /tmp/jaxlib.whl                  # before setup, then JAXLIB_WHEEL=/tmp/jaxlib.whl
    python fp8_wheel_cache.py exists
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_artifact_store import fs as _fs  # noqa: E402  (shared R2/S3 plumbing)

# Override with FP8_WHEEL_URI. Lives under MARIN_PREFIX (s3://marin-na/marin), which task pods can write.
WHEEL_URI = os.environ.get("FP8_WHEEL_URI", "s3://marin-na/marin/grug-fp8/wheels/jaxlib-mixfp8-0.10.0-cp312-cw.whl")
# uv/pip require a PEP440-valid wheel filename, so `get` restores the original basename. `put` records it
# in a sidecar; this is the fallback for the first wheel (stashed before the sidecar existed).
_DEFAULT_WHEEL_NAME = "jaxlib-0.10.0.dev0+selfbuilt-cp312-cp312-manylinux_2_27_x86_64.whl"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _wheel_name(fs):
    """Original wheel basename (PEP440-valid), from the sidecar if present else the known default."""
    name_uri = WHEEL_URI + ".name"
    try:
        if fs.exists(name_uri):
            with fs.open(name_uri, "r") as f:
                return f.read().strip() or _DEFAULT_WHEEL_NAME
    except Exception:
        pass
    return _DEFAULT_WHEEL_NAME


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_put = sub.add_parser("put", help="upload a local wheel to the cache (records its basename)")
    p_put.add_argument("local")
    p_get = sub.add_parser("get", help="download the cached wheel into a dir under its valid basename")
    p_get.add_argument("dest_dir")
    sub.add_parser("exists", help="print yes/no whether the cached wheel exists")
    args = ap.parse_args()

    fs = _fs()
    if args.cmd == "put":
        fs.put(args.local, WHEEL_URI)
        with fs.open(WHEEL_URI + ".name", "w") as f:
            f.write(os.path.basename(args.local))
        print(f"FP8_WHEEL_PUT uri={WHEEL_URI} name={os.path.basename(args.local)} sha256={_sha256(args.local)} bytes={os.path.getsize(args.local)}", flush=True)
    elif args.cmd == "get":
        os.makedirs(args.dest_dir, exist_ok=True)
        dest = os.path.join(args.dest_dir, _wheel_name(fs))
        fs.get(WHEEL_URI, dest)
        print(f"FP8_WHEEL_GET uri={WHEEL_URI} path={dest} sha256={_sha256(dest)} bytes={os.path.getsize(dest)}", flush=True)
    elif args.cmd == "exists":
        print("yes" if fs.exists(WHEEL_URI) else "no", flush=True)


if __name__ == "__main__":
    main()
