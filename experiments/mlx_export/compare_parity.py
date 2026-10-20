# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare an MLX logprob dump against the vLLM goldens and gate the build.

Per scored position:
  top1_match  MLX argmax == vLLM argmax (the best of vLLM's top-K)
  prob_err    max over vLLM's top-K ids of |p_mlx - p_vllm|
  decisive    vLLM top-1 prob - top-2 prob >= DECISIVE_MARGIN

Two groups are gated separately, because pooling lets the long case's many
positions hide errors on the short chat and tool cases:
  short  every position of the cases no longer than the sliding window
  long   positions at or past the sliding window in the longer cases

Usage (Mac):
  uv run python compare_parity.py --dump out/mlx_mixed4.json --golden out/golden_vllm.json \
      --model out/mlx-mixed4 --recipe mixed4 --report out/parity_mixed4.md
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from convert import Recipe
from dump_logprobs import token_id

DECISIVE_MARGIN = 0.2


@dataclass(frozen=True)
class Gate:
    top1_decisive: float
    p95_prob_err: float


GATES = {
    Recipe.EIGHT_BIT: Gate(top1_decisive=0.99, p95_prob_err=0.075),
    Recipe.MIXED_4BIT: Gate(top1_decisive=0.97, p95_prob_err=0.15),
}


@dataclass(frozen=True)
class Position:
    case_id: str
    past_window: bool
    top1_match: bool
    prob_err: float
    decisive: bool


@dataclass(frozen=True)
class Summary:
    positions: int
    top1_all: float
    top1_decisive: float
    p50_prob_err: float
    p95_prob_err: float
    max_prob_err: float


def _int_keys(lp_map: dict) -> dict[int, float]:
    return {token_id(k): lp for k, lp in lp_map.items()}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def scored_positions(dump: dict, goldens: dict, sliding_window: int) -> list[Position]:
    positions = []
    for case in dump["cases"]:
        golden = goldens[case["id"]]
        prompt_len = len(golden["prompt_token_ids"])
        for position_str, mlx_map in case["positions"].items():
            position = int(position_str)
            # Logits index i predicts token i + 1, scored by vLLM's prompt entry
            # i + 1 or continuation entry i - (P - 1).
            if position < prompt_len - 1:
                vllm_map = _int_keys(golden["prompt_top_logprobs"][position + 1])
            else:
                vllm_map = _int_keys(golden["continuation_top_logprobs"][position - (prompt_len - 1)])
            vllm_probs = {tid: math.exp(lp) for tid, lp in vllm_map.items()}
            mlx_lps = _int_keys(mlx_map)
            top2 = sorted(vllm_probs.values(), reverse=True)[:2]
            positions.append(
                Position(
                    case_id=case["id"],
                    past_window=prompt_len > sliding_window and position >= sliding_window,
                    top1_match=case["argmax"][position_str] == max(vllm_probs, key=vllm_probs.get),
                    prob_err=max(abs(math.exp(mlx_lps[tid]) - p) for tid, p in vllm_probs.items()),
                    decisive=len(top2) == 2 and top2[0] - top2[1] >= DECISIVE_MARGIN,
                )
            )
    return positions


def summarize(positions: list[Position]) -> Summary:
    assert positions, "no positions in this group"
    decisive = [p for p in positions if p.decisive]
    errors = [p.prob_err for p in positions]
    return Summary(
        positions=len(positions),
        top1_all=sum(p.top1_match for p in positions) / len(positions),
        top1_decisive=sum(p.top1_match for p in decisive) / max(len(decisive), 1),
        p50_prob_err=percentile(errors, 0.50),
        p95_prob_err=percentile(errors, 0.95),
        max_prob_err=max(errors),
    )


def report_lines(
    groups: dict[str, Summary], cases: dict[str, Summary], gate: Gate, header: str
) -> tuple[list[str], bool]:
    lines = [
        header,
        "",
        "| group | positions | top-1 all | top-1 decisive | p50 prob_err | p95 prob_err | max prob_err |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, s in {**groups, **cases}.items():
        lines.append(
            f"| {label} | {s.positions} | {s.top1_all:.4f} | {s.top1_decisive:.4f} | "
            f"{s.p50_prob_err:.5f} | {s.p95_prob_err:.5f} | {s.max_prob_err:.5f} |"
        )
    lines += ["", f"Gate: top-1 decisive >= {gate.top1_decisive}, p95 prob_err <= {gate.p95_prob_err}.", ""]
    passed = True
    for label, s in groups.items():
        ok = s.top1_decisive >= gate.top1_decisive and s.p95_prob_err <= gate.p95_prob_err
        passed = passed and ok
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {label}")
    return lines, passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="MLX build directory, for the sliding window")
    parser.add_argument("--recipe", required=True, type=Recipe, choices=list(Recipe))
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    goldens_list = json.loads(args.golden.read_text())
    goldens = {g["id"]: g for g in goldens_list}
    sliding_window = json.loads((args.model / "config.json").read_text())["sliding_window"]
    positions = scored_positions(json.loads(args.dump.read_text()), goldens, sliding_window)

    long_ids = {g["id"] for g in goldens_list if len(g["prompt_token_ids"]) > sliding_window}
    groups = {
        "short cases (gated)": summarize([p for p in positions if p.case_id not in long_ids]),
        f"long cases, positions >= {sliding_window} (gated)": summarize([p for p in positions if p.past_window]),
    }
    cases = {g["id"]: summarize([p for p in positions if p.case_id == g["id"]]) for g in goldens_list}
    header = (
        f"# MLX {args.recipe} parity vs vLLM {goldens_list[0]['vllm_release_tag']} "
        f"(source revision {goldens_list[0]['source_revision']})"
    )
    lines, passed = report_lines(groups, cases, GATES[args.recipe], header)
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nresult: {'PASS' if passed else 'FAIL'} (report: {args.report})")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
