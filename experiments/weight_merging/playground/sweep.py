# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge into one resident model and record paired outputs without saving copies."""

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from experiments.weight_merging.playground.candidates import Candidate, merged_weights, read_candidates
from experiments.weight_merging.playground.evaluation import (
    GSM_SMOKE_IDS,
    grade_completion,
    paired_summary,
    prompt,
    read_rows,
    validate_vocabulary,
)

DECODE_SEEDS = (17, 29)


def evaluate(llm, tokenizer, items: list[dict], candidate: Candidate, args) -> list[dict]:
    rows = []
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(item)}], tokenize=False, add_generation_prompt=True
        )
        for item in items
    ]
    modes = [("greedy", 0), *[("sample", seed) for seed in DECODE_SEEDS]]
    for mode, seed in modes:
        params = [
            SamplingParams(
                temperature=0 if mode == "greedy" else 0.7,
                top_p=1 if mode == "greedy" else 0.95,
                max_tokens=args.max_new_tokens,
                seed=seed + index,
                stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")],
            )
            for index in range(len(items))
        ]
        began = time.monotonic()
        outputs = llm.generate(texts, params, use_tqdm=True)
        elapsed = time.monotonic() - began
        with ThreadPoolExecutor(max_workers=4) as pool:
            grades = list(
                pool.map(
                    lambda pair: grade_completion(pair[0], pair[1], defer_code=args.defer_code_scoring),
                    zip(items, [o.outputs[0].text for o in outputs], strict=True),
                )
            )
        for index, (item, output, grade) in enumerate(zip(items, outputs, grades, strict=True)):
            completion = output.outputs[0]
            rows.append(
                {
                    "candidate": candidate.name,
                    "item_id": item["id"],
                    "dataset": item["dataset"],
                    "mode": mode,
                    "seed": seed,
                    "request_seed": seed + index,
                    "completion": completion.text,
                    "tokens": list(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "generation_wall_seconds": elapsed,
                    **grade,
                }
            )
        print(f"{candidate.name} {mode} seed={seed}: {len(items)} scored", flush=True)
    return rows


def engine(paths: dict[str, Path], source: str, rope_theta: float, args):
    return LLM(
        model=str(paths[source]),
        tokenizer=str(paths["instruct"]),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.8,
        swap_space=0,
        enable_prefix_caching=False,
        rope_theta=rope_theta,
        enforce_eager=True,
    )


def reload_weights(llm, paths: dict[str, Path], candidate: Candidate) -> None:
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    loaded = model.load_weights(merged_weights(paths, candidate))
    missing = set(dict(model.named_parameters())) - loaded
    if missing:
        raise ValueError(f"Weight reload omitted parameters: {sorted(missing)}")


def reload_probe(llm, tokenizer, items: list[dict], max_tokens: int) -> list[list[int]]:
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(item)}], tokenize=False, add_generation_prompt=True
        )
        for item in items
    ]
    params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")],
    )
    return [list(output.outputs[0].token_ids) for output in llm.generate(texts, params)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Source revision when the Iris workspace has no .git directory")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--defer-code-scoring", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reload-check-every", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit this experiment to Iris/CoreWeave")
    panel_path = args.run / "panel.json"
    panel = json.loads(panel_path.read_text())
    paths = {key: Path(value) for key, value in json.loads((args.run / "model-paths.json").read_text()).items()}
    tokenizer = AutoTokenizer.from_pretrained(paths["instruct"], padding_side="left", local_files_only=True)
    expected_vocab = tokenizer.get_vocab()
    for path in paths.values():
        other = AutoTokenizer.from_pretrained(path, local_files_only=True)
        validate_vocabulary(expected_vocab, other.get_vocab(), other.pad_token)
    selected = [
        candidate for candidate in read_candidates(args.candidates) if args.only is None or candidate.name in args.only
    ]
    if args.only is not None and {c.name for c in selected} != set(args.only):
        raise ValueError("Unknown candidate in --only")
    metadata = {
        "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "models": json.loads((args.run / "model-sources.json").read_text()),
        "git_revision": args.revision or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "vllm": vllm.__version__,
        "max_num_seqs": args.max_num_seqs,
        "dtype": args.dtype,
        "defer_code_scoring": args.defer_code_scoring,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "decode_seeds": DECODE_SEEDS,
        "prompt_policy": "all use pinned Instruct tokenizer and template",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
        },
        "candidates": [asdict(c) for c in selected],
    }
    output = args.run / "outputs.jsonl"
    existing = read_rows(output) if output.exists() else []
    metadata_path = args.run / "run-metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        for field in (
            "panel_sha256",
            "models",
            "torch",
            "transformers",
            "vllm",
            "max_num_seqs",
            "dtype",
            "defer_code_scoring",
            "max_new_tokens",
            "max_model_len",
            "source_sha256",
        ):
            if previous[field] != metadata[field]:
                raise ValueError(f"Cannot resume with different {field}")
        known = {c["name"]: c for c in previous["candidates"]}
        for candidate in metadata["candidates"]:
            if candidate["name"] in known and known[candidate["name"]] != candidate:
                raise ValueError(f"Cannot resume with a changed recipe: {candidate['name']}")
        known.update({c["name"]: c for c in metadata["candidates"]})
        metadata["candidates"] = list(known.values())
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    expected_count = len(panel["items"]) * 3
    expected_keys = {(item["id"], "greedy", 0) for item in panel["items"]}
    expected_keys.update((item["id"], "sample", seed) for item in panel["items"] for seed in DECODE_SEEDS)
    keys = [(r["candidate"], r["item_id"], r["mode"], r["seed"]) for r in existing]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate item/decoding keys in existing output")
    probe_ids = {f"gsm8k/test/{index}" for index in GSM_SMOKE_IDS}
    probe_items = [item for item in panel["items"] if item["id"] in probe_ids]
    probe_items.extend([item for item in panel["items"] if item["dataset"] == "mbpp"][:8])
    model = engine(paths, "instruct", 1_000_000, args)
    expected_tokens = reload_probe(model, tokenizer, probe_items, args.max_new_tokens)
    del model
    cleanup_dist_env_and_memory()
    model = engine(paths, "anchor", 1_000_000, args)
    reload_weights(model, paths, Candidate("reload-check", "instruct"))
    actual_tokens = reload_probe(model, tokenizer, probe_items, args.max_new_tokens)
    if actual_tokens != expected_tokens:
        raise ValueError("Reloaded Instruct differs from fresh Instruct on smoke panel")
    checks = [{"after_candidate": "initial", "items": len(probe_items), "exact_tokens": True}]
    check_path = args.run / "reload-checks.json"
    check_path.write_text(json.dumps(checks, indent=2) + "\n")
    configs = {"rope": 1_000_000}
    for candidate_index, candidate in enumerate(selected):
        count = sum(r["candidate"] == candidate.name for r in existing)
        if count == expected_count:
            observed = {(r["item_id"], r["mode"], r["seed"]) for r in existing if r["candidate"] == candidate.name}
            if observed != expected_keys:
                raise ValueError(f"Candidate output has different item/decoding keys: {candidate.name}")
            print(f"skip completed {candidate.name}", flush=True)
            continue
        if count:
            raise ValueError(f"Partial candidate output requires inspection: {candidate.name}")
        began = time.monotonic()
        # Recreate the engine for a new RoPE value: modules cache their frequencies.
        if model is None or configs.get("rope") != candidate.rope_theta:
            if model is not None:
                del model
                cleanup_dist_env_and_memory()
            model = engine(paths, "anchor", candidate.rope_theta, args)
            configs["rope"] = candidate.rope_theta
        reload_weights(model, paths, candidate)
        rows = evaluate(model, tokenizer, panel["items"], candidate, args)
        with output.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        existing.extend(rows)
        summary = paired_summary([r for r in existing if not r.get("pending_execution")], "instruct")
        (args.run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "candidate": candidate.name,
                    "seconds": time.monotonic() - began,
                    "scores": [s for s in summary if s["candidate"] == candidate.name],
                }
            ),
            flush=True,
        )
        if (candidate_index + 1) % args.reload_check_every == 0 and candidate.rope_theta == 1_000_000:
            reload_weights(model, paths, Candidate("reload-check", "instruct"))
            if reload_probe(model, tokenizer, probe_items, args.max_new_tokens) != expected_tokens:
                raise ValueError("Mid-sweep reload differs from fresh Instruct")
            checks.append({"after_candidate": candidate.name, "items": len(probe_items), "exact_tokens": True})
            check_path.write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
