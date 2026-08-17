# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether a cluster can run the datakit dedup stack, and hash identically.

The GB200 fleet is aarch64 while every dedup artifact so far was produced on
amd64. Two things have to hold before work moves there. The native wheels must
import at all, and ``dupekit``'s hashes must be bit-identical across
architectures -- cluster keys, n-gram sets and shard routing are all derived
from them, so a hash that differs by one bit silently repartitions the corpus
and destroys co-partitioning against existing attribute trees.

The expected values below were computed on amd64 and are hard-coded, so this
script fails loudly on a mismatch rather than reporting two numbers for a human
to compare.
"""

import hashlib
import json
import platform
import sys

# Computed on amd64 (cw-us-east-02a) with the pinned dupekit build.
EXPECTED: dict[str, int] = {
    "": 3244421341483603138,
    "a": 16629034431890738719,
    "the quick brown fox": 8136938508107280505,
    "clúster-キー/2026": 14865039483309968794,
    "x" * 4096: 9605164738794217619,
}
PROBE_STRINGS = ("", "a", "the quick brown fox", "clúster-キー/2026", "x" * 4096)


def main() -> None:
    report: dict[str, object] = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }

    imports: dict[str, str] = {}
    for name in ("dupekit", "pyarrow", "polars", "numpy", "zephyr", "rigging"):
        try:
            module = __import__(name)
            imports[name] = getattr(module, "__version__", "ok")
        except Exception as error:
            imports[name] = f"FAILED: {type(error).__name__}: {error}"
    report["imports"] = imports

    dupekit = sys.modules.get("dupekit")
    if dupekit is not None:
        hashes = {text: dupekit.hash_xxh3_64(text.encode("utf-8")) for text in PROBE_STRINGS}
        report["xxh3_64"] = {f"len{len(t)}": v for t, v in hashes.items()}
        batch = dupekit.hash_xxh3_64_batch([t.encode("utf-8") for t in PROBE_STRINGS])
        report["batch_matches_scalar"] = list(batch) == [hashes[t] for t in PROBE_STRINGS]
        # A digest over the whole set makes a single value comparable between runs.
        blob = json.dumps([hashes[t] for t in PROBE_STRINGS], sort_keys=True).encode()
        report["xxh3_digest"] = hashlib.sha256(blob).hexdigest()[:32]
        if EXPECTED:
            bad = {t: (hashes[t], EXPECTED[t]) for t in EXPECTED if hashes[t] != EXPECTED[t]}
            report["hash_mismatches"] = bad
            if bad:
                print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
                raise SystemExit(f"dupekit hashes differ from amd64 on {platform.machine()}: {bad}")

    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
