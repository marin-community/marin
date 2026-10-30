# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover native Harbor evidence for incomplete normalized sample tables."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for

rows = json.loads(Path("experiments/evaluation/mini_policy/assembly_audit.json").read_text())
harbor = {
    "bfclparity-pi",
    "bixbench-pi",
    "tau3-pi",
    "ot-tblite-recovery",
    "tb2-recovery",
    "ds-1000-local",
    "swebench-verified",
}
targets = [
    r
    for r in rows
    if not r["source"].endswith(".json")
    and r["benchmark"] in harbor
    and (
        r["status"] == "score_mismatch"
        or r.get("needs_adapter")
        or (r["benchmark"] == "ds-1000-local" and r.get("row_count", 0) < 180)
    )
]
output = Path(os.environ["IRIS_OUTPUT_DIR"])
output.mkdir(exist_ok=True, parents=True)
for cell in targets:
    fs, root = filesystem_for(cell["source"])
    paths = fs.glob(root.rstrip("/") + "/harbor_jobs/*/*/result.json")

    def read(path, filesystem=fs):
        return {"path": path, "result": json.loads(filesystem.cat(path))}

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        trials = list(pool.map(read, paths))
    key = hashlib.sha256((cell["model"] + "\n" + cell["benchmark"]).encode()).hexdigest()[:16]
    payload = json.dumps({"cell": cell, "trials": trials}, default=str).encode()
    url = f"s3://marin-us-east-02a/marin/experiments/mini-eval-policy/inventory-20261030/native_evidence/{key}.json"
    target_fs, target_path = filesystem_for(url)
    target_fs.pipe_file(target_path, payload)
    receipt = {
        "model": cell["model"],
        "benchmark": cell["benchmark"],
        "normalized_count": cell.get("row_count"),
        "native_count": len(trials),
        "evidence_url": url,
        "trial_keys": list(trials[0]["result"]) if trials else [],
    }
    (output / f"{key}.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
