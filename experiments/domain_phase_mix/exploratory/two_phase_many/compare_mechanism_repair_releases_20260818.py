# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["google-cloud-storage"]
# ///
"""Do two mechanism-repair releases produce the same science, or only the same provenance?

v9 differs from v8 only in the audit's acceptance rule: it accepts a cosine that is undefined because one
side is the zero vector, which is what happens at the `final` checkpoint once the schedule has decayed the
learning rate to zero. Tracing the call graph, `_assert_defined_statistic` is reached only through
`_validate_scientific_document`, which has exactly one caller, `audit_outputs`, and that runs only under
`--mode audit`. The children's compute-and-write path never reaches it.

That is an argument, not a measurement. This turns it into one. Both releases ran the same canary rows
from the same permanent checkpoints, so every scientific field should agree; only provenance fields that
name the release or its result path may differ. Anything else that differs is a real behavioural change
and the equivalence claim fails.

Floats are compared exactly first and then with a tolerance, because exact agreement is the strong claim
and small XLA-level nondeterminism would still be compatible with the audit-only change while making
"bit-identical" the wrong word. Both numbers are reported rather than one being chosen in advance.

Usage: ``uv run python ... --left <v8 dir name> --right <v9 dir name> [--scope canary]``
"""

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from google.cloud import storage  # noqa: E402

REFERENCE = SCRIPT_DIR / "reference_outputs"
# Fields that name the release or where its outputs live. These are EXPECTED to differ and are the only
# differences allowed; the allowlist is deliberately narrow, since a broad one would prove nothing.
PROVENANCE_FIELDS = frozenset({"release_sha256", "identity_sha256", "payload_sha256"})
# A sentinel object rather than a string, so a literal "<absent>" in a document cannot masquerade as one.
ABSENT = type("Absent", (), {"__repr__": lambda self: "<absent>"})()


def _client() -> storage.Client:
    return storage.Client()


def _row_blob(client: storage.Client, result_root: str, scope: str, row_id: str):
    """Locate a row's document without assuming the artifact version, which the release does not record."""
    _, _, rest = result_root.partition("gs://")
    bucket_name, _, prefix = rest.partition("/")
    group = row_id.replace("mechanism_", "mechanism_group_")
    bucket = client.bucket(bucket_name)
    matches = [
        blob
        for blob in client.list_blobs(bucket, prefix=f"{prefix}/{scope}/{group}/")
        if blob.name.endswith(f"/rows/{row_id}.json")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one document for {row_id} under {result_root}, found {len(matches)}")
    return json.loads(matches[0].download_as_text())


def differences(left, right, path: str = "") -> list[tuple[str, object, object]]:
    """Every leaf where the two documents disagree, with the path that reaches it.

    Two subtleties, both found by review and both in the direction that would UNDER-report:

    The allowlist applies only at the TOP LEVEL. Applied at every depth it would exempt a nested field
    that merely shares a name with a provenance key -- `checkpoint_metadata` is parsed from an externally
    authored file, so that is a real possibility -- and, because the skip preceded the presence check, it
    also hid a provenance key appearing on one side and not the other.

    Equality alone is not enough at a leaf. Python makes `1 == 1.0`, `True == 1` and `-0.0 == 0.0` all
    true, so a writer-level type change -- exactly the class of behavioural change this tool exists to
    rule out -- would compare equal. The canary documents carry 82 integer and 861 boolean leaves, so the
    surface is not hypothetical. Types must match, and signed zero counts as a difference.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(set(left) | set(right)):
            # Exempt only a provenance key that is present on BOTH sides: its VALUE may legitimately
            # differ between releases, but its disappearance is a schema change and must be reported.
            if path == "" and key in PROVENANCE_FIELDS and key in left and key in right:
                continue
            if key not in left or key not in right:
                found.append((f"{path}/{key}", left.get(key, ABSENT), right.get(key, ABSENT)))
                continue
            found.extend(differences(left[key], right[key], f"{path}/{key}"))
        return found
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [(f"{path}[len]", len(left), len(right))]
        return [d for i, (a, b) in enumerate(zip(left, right, strict=True)) for d in differences(a, b, f"{path}[{i}]")]
    if type(left) is not type(right):
        return [(path, f"{left!r} ({type(left).__name__})", f"{right!r} ({type(right).__name__})")]
    if isinstance(left, float):
        if math.isnan(left) and math.isnan(right):
            return []
        if left == right and math.copysign(1.0, left) == math.copysign(1.0, right):
            return []
        return [(path, left, right)]
    return [] if left == right else [(path, left, right)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="starcoder_wsd80_gradient_mechanism_repair_v8_20260818")
    parser.add_argument("--right", default="starcoder_wsd80_gradient_mechanism_repair_v9_20260818")
    parser.add_argument("--scope", default="canary")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    releases = {}
    for side in ("left", "right"):
        directory = REFERENCE / getattr(args, side)
        releases[side] = json.loads((directory / "release.json").read_text())
    with (REFERENCE / args.left / f"{args.scope}_mechanism_manifest.csv").open() as handle:
        manifest_rows = [row["row_id"] for row in csv.DictReader(handle)]
    print(f"comparing {len(manifest_rows)} {args.scope} rows")
    print(f"  left  {releases['left']['release_version']}  {releases['left']['release_sha256'][:16]}")
    print(f"  right {releases['right']['release_version']}  {releases['right']['release_sha256'][:16]}\n")

    client = _client()

    def compare(row_id: str):
        pair = [_row_blob(client, releases[side]["result_root"], args.scope, row_id) for side in ("left", "right")]
        return row_id, differences(*pair)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(compare, manifest_rows))

    exact = [row for row, diffs in results if not diffs]
    numeric, structural = [], []
    for row, diffs in results:
        for path, left, right in diffs:
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                scale = max(abs(float(left)), abs(float(right)), 1.0)
                numeric.append((row, path, abs(float(left) - float(right)) / scale))
            else:
                structural.append((row, path, left, right))

    print(f"rows identical on every non-provenance field: {len(exact)}/{len(results)}")
    print(f"structural differences (field present/absent, type, or string): {len(structural)}")
    for row, path, left, right in structural[:10]:
        print(f"   {row} {path}: {str(left)[:60]} != {str(right)[:60]}")
    if numeric:
        worst = max(numeric, key=lambda item: item[2])
        within = sum(1 for _, _, d in numeric if d <= args.tolerance)
        print(f"numeric differences: {len(numeric)}, {within} within {args.tolerance:g} relative")
        print(f"   largest relative deviation {worst[2]:.3e} at {worst[0]} {worst[1]}")
    verdict = not structural and (not numeric or max(d for _, _, d in numeric) <= args.tolerance)
    print(f"\nEQUIVALENT: {verdict}")
    if not verdict:
        print("The releases do not produce the same science, so the v8 outputs cannot be reused under v9.")


if __name__ == "__main__":
    main()
