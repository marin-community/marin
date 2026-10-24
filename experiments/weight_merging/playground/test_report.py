# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys

from experiments.weight_merging.playground.report import main


def test_report_preserves_seed_pairs_and_behavior_denominators(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps([{"name": name} for name in ("instruct", "merged")]))
    rows = []
    for candidate in ("instruct", "merged", "ignore"):
        for dataset in ("gsm8k", "mbpp"):
            for mode, seed in (("greedy", 0), ("sample", 17), ("sample", 29)):
                correct = (dataset == "gsm8k") if candidate == "instruct" else (dataset == "mbpp")
                if mode == "sample" and candidate == "merged":
                    correct = seed == 17
                rows.append(
                    {
                        "candidate": candidate,
                        "dataset": dataset,
                        "item_id": dataset,
                        "mode": mode,
                        "seed": seed,
                        "correct": correct,
                        "completion": "<think>unfinished" if candidate == "merged" else "plain",
                        "finish_reason": "length" if candidate == "merged" else "stop",
                    }
                )
    (tmp_path / "outputs.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (tmp_path / "run-metadata.json").write_text(json.dumps({"max_new_tokens": 8192}))
    output = tmp_path / "report.md"
    monkeypatch.setattr(
        sys, "argv", ["report", "--run", str(tmp_path), "--candidates", str(candidates), "--output", str(output)]
    )
    main()
    summaries = json.loads(output.with_suffix(".json").read_text())
    merged = {(r["dataset"], r["mode"], r["seed"]): r for r in summaries if r["candidate"] == "merged"}
    assert merged["gsm8k", "greedy", 0]["losses"] == 1
    assert merged["mbpp", "greedy", 0]["wins"] == 1
    assert merged["gsm8k", "greedy", 0]["unclosed_think"] == 1
    assert merged["gsm8k", "greedy", 0]["truncated"] == 1
    assert merged["gsm8k", "sample", 17]["correct"] == 1
    assert merged["gsm8k", "sample", 29]["correct"] == 0
    assert "max_tokens=8192" in output.read_text()
    assert all(r["candidate"] != "ignore" for r in summaries)
