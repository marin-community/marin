# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import numpy as np

p = Path("/home/ahmad/oa-data/captures/async-rl-v2-optimizer/e62-retained-thinking-v2-question-diagnostics.json")
assert hashlib.sha256(p.read_bytes()).hexdigest() == "d156a8d2cee3ef5b0bc72e9fd957f03014e623ff7c07972b5a460f5ede011f99"
q = json.loads(p.read_text())["questions"]
results = {}
for name, selected, n, k in [
    ("original_k4", [r for r in q if r["stage"].startswith("initial")], 3000, 4),
    ("fresh_k8_legacy_extremes", [r for r in q if r["stage"] == "fresh_k8_legacy_extremes"], 1025, 8),
]:
    selected.sort(key=lambda r: r["prompt_sha256"])
    assert len(selected) == n
    fields = ["original_completed", "full_diagnostic_completed", "prefix_diagnostic_completed"]
    values = np.array([[r[f] for f in fields] for r in selected], dtype=np.float64) / k
    values = np.column_stack([values, values[:, 1] - values[:, 0], values[:, 2] - values[:, 1]])
    rng = np.random.default_rng(17)
    draws = []
    for _ in range(100):
        draws.append(values[rng.integers(0, n, size=(100, n))].mean(axis=1))
    draws = np.concatenate(draws)
    ci = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
    counts = {f: sum(r[f] for r in selected) for f in fields}
    results[name] = {
        "questions": n,
        "responses": n * k,
        "counts": counts,
        "means": values.mean(axis=0).tolist(),
        "ci95": ci.T.tolist(),
        "fields": [*fields, "full_minus_original", "prefix_minus_full"],
        "artificial_prefix_cutoffs": sum(r["artificial_prefix_cutoffs"] for r in selected),
    }
out = {
    "question_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "method": (
        "paired question bootstrap, PCG64 seed17 reset per population,10000 draws,linear "
        "percentile95,conditional on observed responses/selection"
    ),
    "populations": results,
}
p = Path("/home/ahmad/.cache/oa/root-retained-diagnostic-intervals.json")
p.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(results, indent=2))
print("E62_RETAINED_ROOT_INTERVALS_PASS sha256=" + hashlib.sha256(p.read_bytes()).hexdigest())
