# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a candidate run against its same-node control with the #11430 correctness gates.

- (a) prompt scoring on engine 0 is bitwise identical alone, under peer load and alone again;
- (b) per-position max probability error against the JAX reference: the candidate's p50/p90/p99
  and fraction above 0.075 must not exceed the control's 95% bootstrap band (short histories, and
  long histories in 32-position blocks when both runs have ``long.json``);
- (e) decode self-consistency (``decode.json``): |p_decode - p_prefill| of each greedy token;
- D1 (tracked, not a gate): engine 0's greedy top-1 drift under peer load.

Runs are names under the hill-climb output root or full URIs.

Usage: uv run python experiments/evaluation/hero_vllm_hillclimb/grade.py CONTROL CANDIDATE
"""

import argparse
import json
import math

import numpy as np

from experiments.evaluation.hero_vllm_hillclimb import compare

LONG_BLOCK = 32
GREEDY_D1_TOKENS = 64


def _quantiles(values: list[float]) -> dict:
    array = np.asarray(values)
    return {
        "n": int(array.size),
        "p50": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "frac_gt_0.075": float((array > compare.ABSOLUTE_PROBABILITY_BOUND).mean()),
    }


def _top(entry: list) -> dict[int, float]:
    return {int(token): logprob for token, logprob in entry}


def d1(gates: dict) -> dict:
    """Engine 0 greedy drift with the other engines loaded, on the shared prefix of each case."""
    diffs, bitwise, flips = [], 0, 0
    for case, loaded in gates["invariance"]["peer_load"]["greedy"].items():
        alone = gates["greedy"][case]
        for index, (token, other) in enumerate(zip(alone["tokens"][:GREEDY_D1_TOKENS], loaded["tokens"], strict=False)):
            if token != other:
                flips += 1
                break
            a, b = _top(alone["top2"][index])[token], _top(loaded["top2"][index])[token]
            diffs.append(abs(math.exp(a) - math.exp(b)))
            bitwise += a == b
    return {**_quantiles(diffs), "bitwise_frac": bitwise / len(diffs), "greedy_flips": flips}


def decode_consistency(decode: dict) -> dict:
    """Gate (e): decode-time against prefill-scored probability of each greedy token."""
    report = {}
    for mode in ("alone", "batched"):
        diffs = [
            abs(math.exp(a) - math.exp(b))
            for case in decode[mode].values()
            for a, b in zip(case["decode_lp"], case["prefill_lp"], strict=True)
            if b is not None
        ]
        report[mode] = _quantiles(diffs)
    return report


def _long_blocks(long: dict, reference: dict) -> dict:
    scores = {case: value["score"] for case, value in long.items() if case.startswith("long")}
    errors = compare.position_errors(scores, reference)
    return {f"{case}:{i}": e[i : i + LONG_BLOCK] for case, e in errors.items() for i in range(0, len(e), LONG_BLOCK)}


def grade(control: str, candidate: str, reference_uri: str) -> dict:
    reference = compare.load_reference(reference_uri)
    report: dict = {}
    for role, run in (("control", control), ("candidate", candidate)):
        gates = compare._load(run, "gates")
        invariance = compare.g2(gates)
        decode = compare._load(run, "decode")
        report[role] = {
            "run": run,
            "a_bitwise_under_peer_load": invariance["peer_load"]["score_bitwise"],
            "a_bitwise_alone_again": invariance["alone_after"]["score_bitwise"],
            "d1": d1(gates),
            "e_decode_consistency": decode_consistency(decode) if decode else None,
        }
    control_gates, candidate_gates = compare._load(control, "gates"), compare._load(candidate, "gates")
    report["b_short_vs_jax"] = compare.gate_v2_distributional(
        compare.distribution(compare.position_errors(control_gates["score"], reference)),
        compare.distribution(compare.position_errors(candidate_gates["score"], reference)),
    )
    control_long, candidate_long = compare._load(control, "long"), compare._load(candidate, "long")
    if control_long and candidate_long:
        report["b_long_vs_jax"] = compare.gate_v2_distributional(
            compare.distribution(_long_blocks(control_long, reference)),
            compare.distribution(_long_blocks(candidate_long, reference)),
        )
    report["prompt_scores_bitwise_equal_to_control"] = all(
        compare._bitwise(control_gates["score"][case], candidate_gates["score"][case]) for case in control_gates["score"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("control")
    parser.add_argument("candidate")
    parser.add_argument("--reference", default=compare.REFERENCE)
    args = parser.parse_args()
    print(json.dumps(grade(args.control, args.candidate, args.reference), indent=1, default=str))


if __name__ == "__main__":
    main()
