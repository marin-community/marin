# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score an MLX build on the parity cases, teacher-forced on vLLM's continuation.

For each case, one forward pass over prompt + vLLM's greedy continuation gives
float32 logprobs at every position; the dump keeps vLLM's top-K ids plus the
MLX argmax there. A 64-token MLX greedy decode is stored alongside.

Usage (Mac):
  uv run python dump_logprobs.py --model out/mlx-mixed4 --prompts out/prompts.json \
      --golden out/golden_vllm.json --out out/mlx_mixed4.json
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

FREE_GREEDY_TOKENS = 64


def token_id(key) -> int:
    """Token id from a logprob map key: a JSON string or vLLM's ``token_id:N``."""
    return int(str(key).removeprefix("token_id:"))


def logprobs_for_positions(model, token_ids: list[int], wanted: dict[int, set[int]]):
    """Return ({position: {token_id: logprob}}, {position: argmax}) for each wanted position.

    Logits at position i predict token i + 1.
    """
    logits = model(mx.array([token_ids]), cache=model.make_cache())
    logprobs = nn.log_softmax(logits.astype(mx.float32), axis=-1)[0]
    scores, argmaxes = {}, {}
    for position, ids in wanted.items():
        row = logprobs[position]
        argmax = int(mx.argmax(row).item())
        keep = sorted(ids | {argmax})
        scores[position] = dict(zip(keep, row[mx.array(keep)].tolist(), strict=True))
        argmaxes[position] = argmax
    return scores, argmaxes


def free_greedy(model, tokenizer, prompt_ids: list[int]) -> list[int]:
    cache = model.make_cache()
    logits = model(mx.array([prompt_ids]), cache=cache)
    generated = []
    for _ in range(FREE_GREEDY_TOKENS):
        next_id = int(mx.argmax(logits[0, -1]).item())
        if next_id in tokenizer.eos_token_ids:
            break
        generated.append(next_id)
        logits = model(mx.array([[next_id]]), cache=cache)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="MLX build directory or repo")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text())
    goldens = {g["id"]: g for g in json.loads(args.golden.read_text())}
    model, tokenizer = load(args.model, trust_remote_code=True)

    results = []
    for case in prompts["cases"]:
        golden = goldens[case["id"]]
        prompt_ids = list(case["prompt_token_ids"])
        assert golden["prompt_token_ids"] == prompt_ids, case["id"]
        assert golden["source_revision"] == prompts["source_revision"], case["id"]

        # vLLM's prompt entry j scores prompt token j (logits at j - 1); its
        # continuation entry k scores generated token k (logits at P - 1 + k).
        wanted = {
            position - 1: {token_id(k) for k in lp_map}
            for position, lp_map in enumerate(golden["prompt_top_logprobs"])
            if lp_map is not None
        }
        for offset, lp_map in enumerate(golden["continuation_top_logprobs"]):
            wanted[len(prompt_ids) - 1 + offset] = {token_id(k) for k in lp_map}

        scores, argmaxes = logprobs_for_positions(model, prompt_ids + golden["continuation_token_ids"], wanted)
        free = free_greedy(model, tokenizer, prompt_ids)
        results.append(
            {
                "id": case["id"],
                "prompt_token_ids": prompt_ids,
                "positions": {str(p): m for p, m in scores.items()},
                "argmax": {str(p): a for p, a in argmaxes.items()},
                "free_greedy": free,
            }
        )
        print(f"{case['id']}: scored {len(scores)} positions, free greedy {len(free)} tokens")

    dump = {"model": args.model, "source_revision": prompts["source_revision"], "cases": results}
    args.out.write_text(json.dumps(dump) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
