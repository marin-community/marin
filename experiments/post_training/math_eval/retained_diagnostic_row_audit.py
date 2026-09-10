# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

root = Path("/home/ahmad/oa-data/captures/async-rl-v2-optimizer")
r = json.loads(Path("/home/ahmad/.cache/oa/e62-retained-thinking-v2-receipt.json").read_text())
q = json.loads((root / "e62-retained-thinking-v2-question-diagnostics.json").read_text())["questions"]
expected = {(x["stage"], x["prompt_sha256"]): x for x in q}
total = 0
seen_questions = set()
accepted = {"complete", "end_turn", "eos", "stop"}
for stage, receipt in r["stages"].items():
    p = root / f"e62-retained-{stage}-diagnostics.parquet"
    raw = p.read_bytes()
    assert len(raw) == receipt["diagnostic_records_bytes"]
    assert hashlib.sha256(raw).hexdigest() == receipt["diagnostic_records_sha256"]
    rows = pq.read_table(p).to_pylist()
    assert len(rows) == receipt["bins"]["all"]["responses"]
    assert [x["row_ordinal"] for x in rows] == list(range(len(rows)))
    grouped = defaultdict(list)
    for x in rows:
        assert x["stage"] == stage and x["diagnostic_version"] == "legacy-retained-postthinking-diagnostic-v1"
        assert x["actual_corrected_native_generation"] is False and x["actual_4096_serving_equivalence"] is False
        for k in ["response_ids_sha256", "prompt_token_ids_sha256", "source_record_sha256", "prompt_sha256"]:
            assert len(x[k]) == 64 and all(c in "0123456789abcdef" for c in x[k])
        assert 0 < x["original_response_tokens"] <= 8192
        assert x["original_contract_correct"] in [0, 1]
        assert x["original_score_contract_completed"] == x["original_contract_correct"] * int(
            x["original_stop_reason"] in accepted
        )
        full = x["retained_full_response"]
        prefix = x["censored_prefix_4096"]
        cut = x["original_response_tokens"] > 4096
        assert prefix["artificial_cutoff"] == cut
        assert full["effective_stop_diagnostic"] == x["original_stop_reason"]
        assert prefix["effective_stop_diagnostic"] == ("synthetic_prefix_cutoff" if cut else x["original_stop_reason"])
        for d in [full, prefix]:
            reward = d["verifier_reward_diagnostic"]
            correct = d["contract_correct_diagnostic"]
            assert correct in [0, 1] and correct == int(reward >= 1)
            assert d["completed_correct_diagnostic"] == correct * int(d["effective_stop_diagnostic"] in accepted)
        if cut:
            assert prefix["completed_correct_diagnostic"] == 0
        else:
            assert {k: v for k, v in prefix.items() if k != "artificial_cutoff"} == full
        grouped[x["prompt_sha256"]].append(x)
    for uid, items in grouped.items():
        assert len(items) == (8 if stage == "fresh_k8_legacy_extremes" else 4)
        assert len({x["bin"] for x in items}) == 1
        counts = dict(
            stage=stage,
            prompt_sha256=uid,
            bin=items[0]["bin"],
            responses=len(items),
            original_completed=sum(x["original_score_contract_completed"] for x in items),
            original_semantic_resolved=sum(x["original_score_semantic"] is not None for x in items),
            original_semantic_sum=sum(x["original_score_semantic"] or 0 for x in items),
            original_semantic_status=dict(Counter(x["original_semantic_status"] for x in items)),
            full_diagnostic_completed=sum(x["retained_full_response"]["completed_correct_diagnostic"] for x in items),
            prefix_diagnostic_completed=sum(x["censored_prefix_4096"]["completed_correct_diagnostic"] for x in items),
            artificial_prefix_cutoffs=sum(x["censored_prefix_4096"]["artificial_cutoff"] for x in items),
            full_boundary_status=dict(Counter(x["retained_full_response"]["boundary_status"] for x in items)),
            original_stop_counts=dict(Counter(x["original_stop_reason"] for x in items)),
        )
        assert counts == expected[(stage, uid)], (stage, uid)
        seen_questions.add((stage, uid))
    total += len(rows)
assert seen_questions == set(expected) and total == 20200
print(
    "E62_RETAINED_ROOT_ROW_REPLAY_PASS responses=20200 stage_questions=4025 "
    "exact_parquet_hashes=true completion_products=true prefix_cutoffs=true "
    "row_to_question_all_fields=true original_semantics_retained=true"
)
