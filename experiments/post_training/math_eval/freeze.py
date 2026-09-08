# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze audited evaluation batteries from an immutable candidate pool."""

import argparse
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.math_eval.audit_overlay import VERIFIER_REVISION, VERIFIER_SOURCES_SHA256
from experiments.post_training.math_eval.pool import canonical_json
from experiments.post_training.math_eval.pool_audit import audit_manifest
from experiments.post_training.math_eval.sample import sample

QWEN_DEV = {
    "g03-gsm8k": 64,
    "g05-math-l12": 48,
    "g06-math-l3": 48,
    "g07-math-l4": 48,
    "g00-rg-sum-easy": 16,
    "g01-rg-sum-med": 16,
    "g02-rg-sum-hard": 16,
}
QWEN_HELDOUT = {**{f"heldout-math-level-{level}": 200 for level in range(1, 5)}, "heldout-gsm8k-platinum": 1209}


def select_components(manifest, selection, overlay, quotas: Mapping[str, int], *, model: str, split: str, seed: int):
    """Sample each prescribed bin and freeze the combined order by question hash."""
    components = {
        name: sample(manifest, selection, [name], count, seed, model=model, split=split, audit_overlay=overlay)
        for name, count in sorted(quotas.items())
    }
    rows = sorted([row for result in components.values() for row in result.rows], key=lambda row: row["prompt_sha256"])
    ids = [row["prompt_sha256"] for row in rows]
    if len(ids) != len(set(ids)) or len(ids) != sum(quotas.values()):
        raise ValueError("Battery quotas overlap or have missing members")
    return rows, {
        "seed": seed,
        "components": {name: result.receipt for name, result in components.items()},
        "prompt_sha256": ids,
        "combined_sorted_sha256": hashlib.sha256(canonical_json(ids).encode()).hexdigest(),
    }


def freeze_batteries(pool_uri: str, output_uri: str) -> dict[str, Any]:
    """Keep candidate failures visible while requiring every selected row to pass."""
    if not all(uri.startswith("s3://marin-us-east-02a/") for uri in (pool_uri, output_uri)):
        raise ValueError("Evaluation batteries stay in east-02a")
    if StoragePath(output_uri + "/selection.json").exists():
        raise ValueError("A frozen battery is immutable")
    manifest = pq.read_table(pa.BufferReader(StoragePath(pool_uri + "/manifest.parquet").read_bytes())).to_pylist()
    selection = json.loads(StoragePath(pool_uri + "/selection.json").read_bytes())
    audit_uri = output_uri + "/pool-audit-mechanical.jsonl"
    candidate_summary = audit_manifest(pool_uri + "/manifest.parquet", audit_uri, fail_on_evaluation=False)
    audit_content = StoragePath(audit_uri).read_bytes()
    audit_rows = [json.loads(line) for line in audit_content.decode().splitlines()]
    overlay = {
        "manifest_sha256": selection["manifest_sha256"],
        "verifier_revision": VERIFIER_REVISION,
        "verifier_sources_sha256": VERIFIER_SOURCES_SHA256,
        "audit_source_sha256": hashlib.sha256(audit_content).hexdigest(),
        "statuses": {row["prompt_sha256"]: "accept" if row["passed"] else "reject" for row in audit_rows},
        "audit_stage": "mechanical_only",
        "llm_audit": "pending_KE9",
    }
    # The rated Snowball hard battery is deliberately absent until KE7 calibration.
    protocols = {
        "qwen/dev": ("qwen", "dev", QWEN_DEV),
        "qwen/heldout": ("qwen", "heldout", QWEN_HELDOUT),
        "qwen/ood": ("qwen", "ood", {"val-math500": 500}),
        "snowball/ood": ("snowball", "ood", {"val-math500": 500}),
        "snowball/platinum": ("snowball", "heldout", {"heldout-gsm8k-platinum": 1209}),
        "snowball/provisional-dev": ("snowball", "dev", {"g07-math-l4": 128, "g08-math-l5": 128}),
    }
    receipts = {}
    for name, (model, split, quotas) in protocols.items():
        selected, receipt = select_components(manifest, selection, overlay, quotas, model=model, split=split, seed=17)
        if not all(overlay["statuses"][row["prompt_sha256"]] == "accept" for row in selected):
            raise ValueError("Selected evaluation row failed mechanical audit")
        view = pq.read_table(
            pa.BufferReader(StoragePath(f"{pool_uri}/{model}/{split}.parquet").read_bytes())
        ).to_pylist()
        lookup = {row["extra_info"]["prompt_sha256"]: row for row in view}
        data = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist([lookup[row["prompt_sha256"]] for row in selected]), data)
        content = data.getvalue().to_pybytes()
        StoragePath(f"{output_uri}/{name}.parquet").write_bytes(content)
        receipts[name] = {
            **receipt,
            "parquet_sha256": hashlib.sha256(content).hexdigest(),
            "rows": len(selected),
            "status": "provisional_before_KE7" if name.endswith("provisional-dev") else "frozen",
        }
    final = {
        "candidate_pool_uri": pool_uri,
        "candidate_manifest_sha256": selection["manifest_sha256"],
        "candidate_audit": candidate_summary,
        "audit_overlay_sha256": hashlib.sha256(canonical_json(overlay).encode()).hexdigest(),
        "protocols": receipts,
    }
    StoragePath(output_uri + "/audit-overlay.json").write_bytes(canonical_json(overlay).encode())
    StoragePath(output_uri + "/selection.json").write_bytes(canonical_json(final).encode())
    summary = {
        "candidate_audit": candidate_summary,
        "protocols": {
            name: {key: row[key] for key in ("rows", "combined_sorted_sha256", "parquet_sha256", "status")}
            for name, row in receipts.items()
        },
        "audit_overlay_sha256": final["audit_overlay_sha256"],
    }
    print("KE2_FINAL_BATTERY_PASS " + json.dumps(summary))
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    args = parser.parse_args()
    freeze_batteries(args.pool_uri, args.output_uri)
