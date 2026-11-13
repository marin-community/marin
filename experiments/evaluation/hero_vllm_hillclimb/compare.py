# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade an arm's gates (G2 within the arm; G1b/G4/G5 against a control arm) and tabulate load.

    uv run python experiments/evaluation/hero_vllm_hillclimb/compare.py <control-run> [<candidate-run>]

Runs are names under the hill-climb TTL output root or full URIs.
"""

import argparse
import json
import math

from rigging.filesystem.factory import open_url

OUTPUT_ROOT = "s3://marin-us-east-02a/tmp/ttl=14d/hero-vllm-hillclimb/runs"
GPUS = 16
# Thresholds (see the gate proposal): relative probability error vs the accepted baseline, the
# same-rank co-batching log-prob bound, and the near-tie margin that excuses a greedy divergence.
RELATIVE_PROBABILITY_BOUND = 0.005
SAME_RANK_LOGPROB_BOUND = 0.02
NEAR_TIE_MARGIN = 0.05
MAX_NEAR_TIE_DIVERGENCES = 2


def _load(run: str, phase: str) -> dict | None:
    uri = run if "://" in run else f"{OUTPUT_ROOT}/{run}"
    try:
        with open_url(f"{uri}/{phase}.json", "r") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def _position_errors(a: list[list], b: list[list]) -> tuple[float, float, bool]:
    """Max probability error and max log-prob error over tokens present in both top-k; winner agreement."""
    da = dict((int(t), lp) for t, lp in a)
    db = dict((int(t), lp) for t, lp in b)
    shared = set(da) & set(db)
    prob = max((abs(math.exp(da[t]) - math.exp(db[t])) for t in shared), default=float("inf"))
    logp = max((abs(da[t] - db[t]) for t in shared), default=float("inf"))
    # Tokens missing from one side: bound their probability by that side's k-th probability.
    for t in set(da) ^ set(db):
        present, other = (da, db) if t in da else (db, da)
        prob = max(prob, abs(math.exp(present[t]) - math.exp(min(other.values()))))
    return prob, logp, a[0][0] == b[0][0]


def _compare_scores(control: dict, candidate: dict) -> dict:
    worst_prob, worst_logp, winner_mismatch, positions = 0.0, 0.0, [], 0
    for case_id, ref in control.items():
        for index, (a, b) in enumerate(zip(ref["positions"], candidate[case_id]["positions"], strict=True)):
            if not a or not b:
                continue
            prob, logp, same = _position_errors(a, b)
            positions += 1
            worst_prob = max(worst_prob, prob)
            worst_logp = max(worst_logp, logp)
            if not same:
                margin = math.exp(a[0][1]) - math.exp(a[1][1]) if len(a) > 1 else 1.0
                winner_mismatch.append({"case": case_id, "position": index, "control_margin": margin})
    return {
        "positions": positions,
        "max_probability_error": worst_prob,
        "max_logprob_error": worst_logp,
        "winner_mismatches": winner_mismatch,
    }


def _bitwise(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _greedy(control: dict, candidate: dict) -> dict:
    divergences = []
    for case_id, ref in control.items():
        got = candidate[case_id]
        for index, (x, y) in enumerate(zip(ref["tokens"], got["tokens"], strict=True)):
            if x != y:
                top2 = ref["top2"][index]
                margin = top2[0][1] - top2[1][1] if len(top2) > 1 else float("inf")
                divergences.append({"case": case_id, "position": index, "control_logprob_margin": margin})
                break
    near_ties = [d for d in divergences if d["control_logprob_margin"] < NEAR_TIE_MARGIN]
    passed = len(divergences) == len(near_ties) and len(near_ties) <= MAX_NEAR_TIE_DIVERGENCES
    return {"prompts": len(control), "divergences": divergences, "pass": passed}


def g2(gates: dict) -> dict:
    """Within one arm: engine-0 results alone vs with peer-rank load vs with same-rank load."""
    alone_score = {k: gates["score"][k] for k in gates["invariance"]["peer_load"]["score"]}
    alone_greedy = {k: {"tokens": gates["greedy"][k]["tokens"][:64]} for k in alone_score}
    out = {}
    for label in ("peer_load", "same_rank_load", "alone_after"):
        probe = gates["invariance"][label]
        greedy_tokens = {k: {"tokens": v["tokens"]} for k, v in probe["greedy"].items()}
        scores = _compare_scores(alone_score, probe["score"])
        out[label] = {
            "score_bitwise": all(_bitwise(alone_score[k], probe["score"][k]) for k in alone_score),
            "greedy_tokens_equal": _bitwise(alone_greedy, greedy_tokens),
            "max_logprob_error": scores["max_logprob_error"],
            "max_probability_error": scores["max_probability_error"],
            "busy_before": probe.get("busy_before"),
        }
    out["pass_peer_exact"] = out["peer_load"]["score_bitwise"] and out["peer_load"]["greedy_tokens_equal"]
    out["pass_same_rank"] = out["same_rank_load"]["max_logprob_error"] <= SAME_RANK_LOGPROB_BOUND
    return out


def load_table(load: dict) -> list[dict]:
    rows = []
    for key, summary in load.items():
        if not key.startswith("c") or not isinstance(summary, dict):
            continue
        tps = summary.get("window_output_tokens_per_second")
        rows.append(
            {
                "C": summary["concurrency"],
                "out_tok_s": round(tps, 1) if tps else None,
                "out_tok_s_gpu": round(tps / GPUS, 2) if tps else None,
                "ttft_p50_s": summary["ttft_p50"],
                "ttft_p90_s": summary["ttft_p90"],
                "itl_p50_ms": summary["itl_p50"] * 1000 if summary["itl_p50"] else None,
                "itl_p90_ms": summary["itl_p90"] * 1000 if summary["itl_p90"] else None,
                "window_s": summary.get("window_seconds"),
                "requests": summary["requests"],
            }
        )
    return sorted(rows, key=lambda row: row["C"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("candidate", nargs="?")
    args = parser.parse_args()
    report: dict = {}
    for role, run in (("control", args.control), ("candidate", args.candidate)):
        if run is None:
            continue
        gates = _load(run, "gates")
        load = _load(run, "load")
        report[role] = {
            "run": run,
            "g2": g2(gates) if gates and "invariance" in gates else gates,
            "load": load_table(load) if load and "error" not in load else load,
        }
    if args.candidate:
        control_gates, candidate_gates = _load(args.control, "gates"), _load(args.candidate, "gates")
        if control_gates and candidate_gates and "score" in control_gates and "score" in candidate_gates:
            g1b = _compare_scores(control_gates["score"], candidate_gates["score"])
            g1b["pass"] = g1b["max_probability_error"] <= RELATIVE_PROBABILITY_BOUND
            report["g1b_vs_control"] = g1b
            report["g4_greedy_vs_control"] = _greedy(control_gates["greedy"], candidate_gates["greedy"])
        control_long, candidate_long = _load(args.control, "long"), _load(args.candidate, "long")
        if control_long and candidate_long:
            report["g5_vs_control"] = {
                case: {
                    "score": _compare_scores({case: control_long[case]["score"]}, {case: candidate_long[case]["score"]}),
                    "greedy": _greedy({case: control_long[case]["greedy"]}, {case: candidate_long[case]["greedy"]}),
                }
                for case in control_long
                if case.startswith("long") and case in candidate_long
            }
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
