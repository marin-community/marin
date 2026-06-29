# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ship benchmark/profile artifacts to and from the cluster's R2-backed S3 bucket.

The FP8 benchmark produces artifacts that must move between a cluster task container and localhost:
the forked jaxlib wheel (see ``fp8_wheel_cache.py``) and profiler trace tarballs (see the ``--profile``
mode of ``bench_ragged_fp8_autotune.py``). Both need the same object-store plumbing, so it lives here
once: a filesystem handle wired to the R2 creds iris injects into the task container
(``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_ENDPOINT_URL``), plus a generic ``cp``.

    python fp8_artifact_store.py cp /tmp/prof.tgz s3://marin-na/marin/grug-fp8/profiles/prof.tgz
    python fp8_artifact_store.py cp s3://marin-na/marin/grug-fp8/profiles/prof.tgz /tmp/prof.tgz
"""

import argparse
import os


def fs():
    """An s3fs filesystem bound to the injected R2 endpoint (pip-installs s3fs on miss in-container)."""
    try:
        import s3fs  # noqa: PLC0415  (optional dep: pip-install on miss in the task container)
    except ImportError:
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "s3fs"], check=True)
        import s3fs  # noqa: PLC0415
    client_kwargs = {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    return s3fs.S3FileSystem(client_kwargs=client_kwargs)


def cp(src, dst):
    """Copy a single file local<->s3; exactly one of ``src`` / ``dst`` must be an ``s3://`` uri."""
    src_s3, dst_s3 = src.startswith("s3://"), dst.startswith("s3://")
    handle = fs()
    if src_s3 and not dst_s3:
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        handle.get(src, dst)
    elif dst_s3 and not src_s3:
        handle.put(src, dst)
    else:
        raise SystemExit("cp: exactly one of src/dst must be an s3:// uri")
    local = dst if src_s3 else src
    print(f"FP8_CP src={src} dst={dst} bytes={os.path.getsize(local)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_cp = sub.add_parser("cp", help="copy a single file local<->s3")
    p_cp.add_argument("src")
    p_cp.add_argument("dst")
    args = ap.parse_args()
    if args.cmd == "cp":
        cp(args.src, args.dst)


if __name__ == "__main__":
    main()
