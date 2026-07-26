# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert an uploaded xprof capture into a PGLE ProfiledInstructionsProto.

Downloads the xplane dumps from the run's xprof S3 root, aggregates them into XLA's
ProfiledInstructionsProto, and prints the serialized proto gzip+base64 between markers so
the caller can harvest it from job logs (the submitting sandbox has no S3 credentials).

Usage (as an iris CPU job): python -m experiments.grug.moe.pgle_convert <xprof_s3_root>
"""

import base64
import gzip
import sys



import fsspec

from jax._src.lib import _profiler

CHUNK = 3000


def main() -> None:
    root = sys.argv[1].rstrip("/")
    fs = fsspec.filesystem(root.split("://", 1)[0])
    xplanes = [p for p in fs.find(root) if p.endswith(".xplane.pb")]
    if not xplanes:
        raise FileNotFoundError(f"no .xplane.pb under {root}")
    print(f"found {len(xplanes)} xplane files:", *xplanes, sep="\n  ")

    # Mirror jax's PGLEProfiler: extract the FDO profile from each raw xspace, then
    # aggregate the extracted profiles at the given percentile.
    fdo_profiles = []
    for p in xplanes:
        with fs.open(p, "rb") as f:
            data = f.read()
        fdo = _profiler.get_fdo_profile(data)
        print(f"{p}: xspace {len(data)} bytes -> fdo {len(fdo)} bytes")
        if fdo:
            fdo_profiles.append(fdo)
    if not fdo_profiles:
        raise RuntimeError("all fdo profiles empty (CUPTI contention or missing device trace?)")
    proto = _profiler.aggregate_profiled_instructions(fdo_profiles, 90)
    if isinstance(proto, str):
        proto = proto.encode()
    print(f"proto bytes: {len(proto)}")
    payload = base64.b64encode(gzip.compress(proto)).decode()
    print("PGLE_B64_BEGIN")
    for i in range(0, len(payload), CHUNK):
        print(payload[i : i + CHUNK])
    print("PGLE_B64_END")


if __name__ == "__main__":
    main()
