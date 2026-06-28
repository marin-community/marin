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
import subprocess
import sys

# Override with FP8_WHEEL_URI. Lives under MARIN_PREFIX (s3://marin-na/marin), which task pods can write.
WHEEL_URI = os.environ.get("FP8_WHEEL_URI", "s3://marin-na/marin/grug-fp8/wheels/jaxlib-mixfp8-0.10.0-cp312-cw.whl")


def _fs():
    try:
        import s3fs  # noqa: PLC0415  (optional dep: pip-install on miss in the task container)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "s3fs"], check=True)
        import s3fs  # noqa: PLC0415
    client_kwargs = {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    return s3fs.S3FileSystem(client_kwargs=client_kwargs)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_put = sub.add_parser("put", help="upload a local wheel to the cache")
    p_put.add_argument("local")
    p_get = sub.add_parser("get", help="download the cached wheel to a local path")
    p_get.add_argument("local")
    sub.add_parser("exists", help="print yes/no whether the cached wheel exists")
    args = ap.parse_args()

    fs = _fs()
    if args.cmd == "put":
        fs.put(args.local, WHEEL_URI)
        print(f"FP8_WHEEL_PUT uri={WHEEL_URI} sha256={_sha256(args.local)} bytes={os.path.getsize(args.local)}", flush=True)
    elif args.cmd == "get":
        fs.get(WHEEL_URI, args.local)
        print(f"FP8_WHEEL_GET uri={WHEEL_URI} sha256={_sha256(args.local)} bytes={os.path.getsize(args.local)}", flush=True)
    elif args.cmd == "exists":
        print("yes" if fs.exists(WHEEL_URI) else "no", flush=True)


if __name__ == "__main__":
    main()
