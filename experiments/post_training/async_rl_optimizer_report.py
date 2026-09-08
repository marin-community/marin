# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize the audited three-arm optimizer qualification without rescoring text."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def report(root: Path, costs_path: Path) -> dict:
    inputs = {
        name: json.loads((root / name).read_text())
        for name in ("audit.json", "history.json", "native-audit.json", "finelog-raw.json", "terminal-exports.json")
    }
    audit, history = inputs["audit.json"], inputs["history.json"]
    assert audit["clean_end_to_end"] and not audit["errors"]
    costs = json.loads(costs_path.read_text())
    result = {
        "arms": {},
        "contrasts": [],
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in inputs},
    }
    result["source_sha256"][str(costs_path)] = hashlib.sha256(costs_path.read_bytes()).hexdigest()
    for arm, run in audit["runs"].items():
        rows = history[arm]
        assert rows[-1]["policy/updates_completed_valid"] == 1
        assert rows[-1]["policy/updates_completed"] == rows[-1]["policy/updates_attempted"] == 8
        ev = run["eval_dumps"][-1]["metrics"]
        native = inputs["finelog-raw.json"]["runs"][arm]
        ages = [json.loads(e["body_json"]) for e in native["events"] if e["name"] == "consumed_age"]
        tokens = sum(x["response_tokens"] for x in ages)
        assert tokens == sum(x["value"] for x in native["work"] if x["work_type"] == "consumed_response_token")
        ordered = sorted(ages, key=lambda x: x["age"])
        cumulative = 0
        p95 = None
        for event in ordered:
            cumulative += event["response_tokens"]
            if cumulative >= 0.95 * tokens:
                p95 = event["age"]
                break
        ess = [
            v
            for row in rows
            for k, v in row.items()
            if k.startswith("policy/by_update/") and k.endswith("/log_ratio_ess_fraction")
        ]
        result["arms"][arm] = {
            "successful_updates": rows[-1]["policy/updates_completed"],
            "tokens": tokens,
            "core_seconds": sum(r.get("timing/step", 0) for r in rows),
            "task_gpu_hours": costs["runs"][arm]["task_gpu_hours"],
            "raw_score": ev["eval/all/avg_score"],
            "completed_score": ev["eval/all/completed_stop_score_contribution"],
            "semantic_score": None,
            "truncation_fraction": ev["eval/all/length_stop_fraction"],
            "response_tokens_mean": ev["eval/all/response_tokens_mean"],
            "token_weighted_age_mean": sum(x["age"] * x["response_tokens"] for x in ages) / tokens,
            "token_weighted_age_p95": p95,
            "worker_ess_range": [min(ess), max(ess)],
            "wandb_url": run["wandb_url"],
        }
    reference = audit["vectors"]["sync_n1"]["8"]
    for arm, step in (("async_n1", "8"), ("sync_n4", "2")):
        candidate = audit["vectors"][arm][step]
        assert len(reference) == len(candidate) == 128
        assert [r[:2] for r in reference] == [r[:2] for r in candidate]
        assert all(r[2] in (0, 1) for r in reference + candidate)

        def scores(rs):
            return np.array([r[2] * (r[3] in {"stop", "complete", "end_turn", "eos"}) for r in rs])

        delta = scores(candidate) - scores(reference)
        draws = delta[np.random.default_rng(20260908).integers(0, 128, (10000, 128))].mean(axis=1)
        result["contrasts"].append(
            {
                "candidate": arm,
                "reference": "sync_n1",
                "metric": "score_contract_completed",
                "delta_pp": float(delta.mean() * 100),
                "candidate_only_correct": int((delta > 0).sum()),
                "reference_only_correct": int((delta < 0).sum()),
                "individual_97_5_percent_interval_pp": (np.quantile(draws, [0.0125, 0.9875]) * 100).tolist(),
                "interpretation": "descriptive paired-question bootstrap; nominal simultaneous 95% over two contrasts",
                "resamples": 10000,
                "seed": 20260908,
            }
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("costs", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.costs), indent=2))
