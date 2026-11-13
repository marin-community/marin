# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade an arm's gates (G2 within the arm; G1b/G4/G5 against a control arm) and tabulate load.

    uv run python experiments/evaluation/hero_vllm_hillclimb/compare.py <control-run> [<candidate-run>]

Runs are names under the hill-climb TTL output root or full URIs.
"""

import argparse
import io
import json
import math

import numpy as np
from rigging.filesystem.factory import open_url

OUTPUT_ROOT = "s3://marin-us-east-02a/tmp/ttl=14d/hero-vllm-hillclimb/runs"
REFERENCE = "s3://marin-us-east-02a/tmp/ttl=14d/hero-vllm-hillclimb/reference/ref-a4"
# NextTokenObservation family screen (vLLM #110/#119/#129): probability error bound vs JAX.
ABSOLUTE_PROBABILITY_BOUND = 0.075
GPUS = 16
# Thresholds (see the gate proposal): relative probability error vs the accepted baseline, the
# same-rank co-batching log-prob bound, and the near-tie margin that excuses a greedy divergence.
RELATIVE_PROBABILITY_BOUND = 0.005
SAME_RANK_LOGPROB_BOUND = 0.02
NEAR_TIE_MARGIN = 0.05
MAX_NEAR_TIE_DIVERGENCES = 2


def _load(run: str, phase: str) -> dict | None:
    uri = run if "://" in run or run.startswith("/") else f"{OUTPUT_ROOT}/{run}"
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


def load_reference(uri: str = REFERENCE) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """(case_id, position) -> (top token ids, top log-probs), sorted by log-prob descending."""
    with open_url(f"{uri}/reference.npz", "rb") as handle:
        arrays = np.load(io.BytesIO(handle.read()))
    with open_url(f"{uri}/reference.json", "r") as handle:
        case_ids = json.load(handle)["case_ids"]
    rows = {}
    for case, position, ids, logprobs in zip(
        arrays["row_case"], arrays["row_position"], arrays["top_ids"], arrays["top_logprobs"], strict=True
    ):
        order = np.argsort(-logprobs)
        rows[(case_ids[int(case)], int(position))] = (ids[order], logprobs[order])
    return rows


def g1a(scores: dict, reference: dict) -> dict:
    """Score each vLLM position against JAX: probability error, reference top-5 coverage, supported winner."""
    worst = {"max_probability_error": 0.0}
    failures = []
    positions = 0
    for case_id, result in scores.items():
        for position, returned in enumerate(result["positions"]):
            key = (case_id, position)
            if key not in reference or not returned:
                continue
            ids, logprobs = reference[key]
            ref = dict(zip(ids.tolist(), logprobs.tolist(), strict=True))
            floor = math.exp(float(logprobs[-1]))
            actual = {int(t): lp for t, lp in returned}
            errors = [abs(math.exp(lp) - (math.exp(ref[t]) if t in ref else floor)) for t, lp in actual.items()]
            top5 = ids[:5].tolist()
            missing = [t for t in top5 if t not in actual]
            winner = max(actual, key=actual.get)
            gap = math.exp(float(logprobs[0])) - (math.exp(ref[winner]) if winner in ref else 0.0)
            max_error = max(errors)
            positions += 1
            if max_error > worst["max_probability_error"]:
                worst = {"max_probability_error": max_error, "case": case_id, "position": position}
            if missing or max_error > ABSOLUTE_PROBABILITY_BOUND or gap > 2 * max_error + 1e-6:
                failures.append(
                    {"case": case_id, "position": position, "max_error": max_error, "missing_top5": missing, "gap": gap}
                )
    return {
        "positions": positions,
        "worst": worst,
        "failures": len(failures),
        "first_failures": failures[:10],
        "pass": not failures,
    }


def position_errors(scores: dict, reference: dict) -> dict[str, np.ndarray]:
    """Per case: max probability error over vLLM's returned tokens at each referenced position."""
    out = {}
    for case_id, result in scores.items():
        errors = []
        for position, returned in enumerate(result["positions"]):
            key = (case_id, position)
            if key not in reference or not returned:
                continue
            ids, logprobs = reference[key]
            ref = dict(zip(ids.tolist(), logprobs.tolist(), strict=True))
            floor = math.exp(float(logprobs[-1]))
            errors.append(
                max(abs(math.exp(lp) - (math.exp(ref[int(t)]) if int(t) in ref else floor)) for t, lp in returned)
            )
        if errors:
            out[case_id] = np.asarray(errors)
    return out


GATE_STATS = {
    "p50": lambda e: float(np.quantile(e, 0.5)),
    "p90": lambda e: float(np.quantile(e, 0.9)),
    "p99": lambda e: float(np.quantile(e, 0.99)),
    "frac_gt_0.075": lambda e: float(np.mean(e > ABSOLUTE_PROBABILITY_BOUND)),
}


def distribution(errors: dict[str, np.ndarray], *, replicates: int = 2000, seed: int = 0) -> dict:
    """Point statistics plus a 95% bootstrap interval that resamples whole histories."""
    cases = sorted(errors)
    pooled = np.concatenate([errors[c] for c in cases])
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in GATE_STATS}
    for _ in range(replicates):
        draw = np.concatenate([errors[cases[i]] for i in rng.integers(0, len(cases), len(cases))])
        for name, stat in GATE_STATS.items():
            samples[name].append(stat(draw))
    return {
        name: {
            "value": stat(pooled),
            "ci95": [float(np.quantile(samples[name], 0.025)), float(np.quantile(samples[name], 0.975))],
        }
        for name, stat in GATE_STATS.items()
    } | {"positions": int(pooled.size), "histories": len(cases)}


def gate_v2_distributional(control: dict, candidate: dict) -> dict:
    """Fail when a candidate statistic falls outside the control's 95% bootstrap band (worse side only)."""
    verdict = {}
    for name in GATE_STATS:
        low, high = control[name]["ci95"]
        value = candidate[name]["value"]
        verdict[name] = {"candidate": value, "control_band": [low, high], "pass": value <= high}
    verdict["pass"] = all(v["pass"] for v in verdict.values() if isinstance(v, dict))
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("candidate", nargs="?")
    parser.add_argument("--reference", default=REFERENCE)
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
    reference = load_reference(args.reference)
    for role in ("control", "candidate"):
        run = getattr(args, role)
        gates = _load(run, "gates") if run else None
        if gates and "score" in gates:
            report[role]["g1a_vs_jax"] = g1a(gates["score"], reference)
        long = _load(run, "long") if run else None
        if long:
            report[role]["g5_vs_jax"] = {
                case: g1a({case: value["score"]}, reference) for case, value in long.items() if case.startswith("long")
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
