# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the paged cache with a small checkpoint and fixed token histories."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical", type=int, choices=[0, 1], default=1)
    parser.add_argument("--quantization", choices=["fp8"])
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--pcp", type=int, default=1)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())["cases"]
    speculative = None
    if args.speculative:
        speculative = {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 1, "prompt_lookup_min": 1}
    llm = LLM(
        model=args.model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=4,
        block_size=16,
        max_num_batched_tokens=16,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_logprobs=128,
        kv_cache_memory_bytes=128 * 1024**2,
        quantization=args.quantization,
        additional_config={"grug_logical_kv_cache": bool(args.logical)},
        prefill_context_parallel_size=args.pcp,
        speculative_config=speculative,
        disable_log_stats=False,
    )
    sample = SamplingParams(temperature=0, max_tokens=1, logprobs=128, ignore_eos=True, detokenize=False)
    records = []

    def observe(case, position, variant):
        tokens = case["token_ids"][: position + 1]
        result = llm.generate([{"prompt_token_ids": tokens}], sample, use_tqdm=False)[0]
        output = result.outputs[0]
        assert output.logprobs is not None
        scores = output.logprobs[0]
        assert len(scores) == 128
        logprobs = [float(scores[token].logprob) for token in range(128)]
        records.append(
            {
                "case_id": case["id"],
                "position": position,
                "variant": variant,
                "token_ids": tokens,
                "num_cached_tokens": result.num_cached_tokens,
                "output_token_id": output.token_ids[0],
                "logprobs": logprobs,
            }
        )

    for case in cases:
        for position in case["positions"]:
            llm.reset_prefix_cache()
            observe(case, position, "cold")
            llm.reset_prefix_cache()
            boundary = position // 16 * 16
            if boundary:
                llm.generate([{"prompt_token_ids": case["token_ids"][:boundary]}], sample, use_tqdm=False)
            observe(case, position, "warm")
    llm.reset_prefix_cache()
    observe(cases[0], cases[0]["positions"][-1], "branch-seed")
    observe(cases[1], cases[1]["positions"][-1], "branch")
    # Seed one history, visit an unrelated session, then return to the first.
    for case in (cases[0], cases[-1], cases[0]):
        observe(case, case["positions"][-1], "isolated")
    prompt = [value for token in range(128) for value in (token, 0, 1, 2)]
    trajectory = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(temperature=0, max_tokens=8, logprobs=128, ignore_eos=True, detokenize=False),
        use_tqdm=False,
    )[0].outputs[0]
    history = prompt + list(trajectory.token_ids)
    assert trajectory.logprobs is not None
    for index, scores in enumerate(trajectory.logprobs):
        records.append(
            {
                "case_id": "generated-history",
                "position": len(prompt) + index - 1,
                "variant": "decode",
                "token_ids": history[: len(prompt) + index],
                "output_token_id": trajectory.token_ids[index],
                "logprobs": [float(scores[token].logprob) for token in range(128)],
            }
        )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "observations.json").write_text(json.dumps(records) + "\n")
    generated = {
        "cases": [
            {
                "id": "generated-history",
                "token_ids": history,
                "positions": [len(prompt) + index - 1 for index in range(8)],
            }
        ]
    }
    (args.output / "generated-cases.json").write_text(json.dumps(generated) + "\n")
    metrics = [asdict(metric) for metric in llm.get_metrics() if "spec_decode" in metric.name]
    (args.output / "speculative-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if args.speculative:
        assert any(metric.get("value", 0) > 0 for metric in metrics if "num_drafts" in metric["name"]), metrics
    assert all(np.isfinite(record["logprobs"]).all() for record in records)
    assert any(x["num_cached_tokens"] for x in records if x["variant"] == "warm")
    print(
        json.dumps(
            {
                "observations": len(records),
                "warm_cached_tokens": [x["num_cached_tokens"] for x in records if x["variant"] == "warm"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
