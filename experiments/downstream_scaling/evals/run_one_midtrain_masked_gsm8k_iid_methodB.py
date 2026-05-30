# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Method-B single-cell launcher: marin.inference.distributed.inference().

Bypasses the Marin ExecutorStep framework entirely. Inline grading. Calls
`inference()` directly, reads the resulting shard files, groups by prompt id,
runs lm-eval `flexible_extract` + `strict_match` filters inline, writes
`grades.jsonl.gz` in the canonical Marin schema.

Output paths:
  gs://marin-us-east5/downstream_scaling/evals/delphi_midtrain/masked_gsm8k_mask00_methodB_distributed/{slug}/completions.jsonl.gz
  gs://marin-us-east5/downstream_scaling/evals/delphi_midtrain/masked_gsm8k_mask00_methodB_distributed/{slug}/grades.jsonl.gz

Usage:
  uv run iris ... job run --no-wait \\
    --job-name aa-mtxB-1e22-p33m67-lr0p5 \\
    --zone us-east5-a \\
    --cpu 4 --memory 16GB --disk 30GB \\
    --priority interactive \\
    -e MARIN_PREFIX gs://marin-us-east5 \\
    -e WANDB_API_KEY "$WANDB_API_KEY" \\
    -- python experiments/downstream_scaling/evals/run_one_midtrain_masked_gsm8k_iid_methodB.py --slug 1e22_p33m67_lr0.5
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import sys
import time
from typing import Any

import fsspec
from marin.evaluation.utils import discover_hf_checkpoints
from marin.inference.distributed import (
    InferenceConfig,
    ModelSpec,
    SamplingParams,
    inference,
)

from experiments.downstream_scaling.evals.framework.schema import read_prompt_rows
from experiments.downstream_scaling.models.midtrain import DELPHI_MIDTRAIN_MATRIX

logger = logging.getLogger(__name__)

# Sampling config — byte-identical to Rohith's mask_00 launcher except n=1
# (we replicate prompts 32x to recover the n=32 sampling).
N_SAMPLES = 32
N_PROBLEMS = 256
TEMPERATURE = 0.6
TOP_P = 1.0
TOP_K = 1000
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

# Shared prompt artifact — same one Rohith's matrix + Method A use.
PROMPTS_URI = "gs://marin-us-east5/downstream_scaling/evals/prompts/masked_gsm8k-42fd11/prompts.jsonl.gz"

# Method B output prefix (parallel to masked_gsm8k_mask00_methodA_cache/).
OUTPUT_PREFIX = "downstream_scaling/evals/delphi_midtrain/masked_gsm8k_mask00_methodB_distributed"


def _materialize_records(prompts_uri: str) -> list[dict[str, Any]]:
    """Read prompts and replicate 32x per problem with id=`{base}#{sample_idx}`."""
    records: list[dict[str, Any]] = []
    for row in read_prompt_rows(prompts_uri):
        base = row["id"]
        prompt = row["prompt"]
        for j in range(N_SAMPLES):
            records.append({"id": f"{base}#{j}", "payload": {"kind": "text", "prompt": prompt}})
    logger.info("Materialized %d PromptRecord rows (%d problems x %d samples)", len(records), len(records) // N_SAMPLES, N_SAMPLES)
    return records


def _read_shard_outputs(results_uri: str) -> list[dict[str, Any]]:
    """Read all shard-NNNNNNNN.jsonl.gz under results_uri, return list of dicts."""
    fs, _ = fsspec.core.url_to_fs(results_uri)
    pattern = f"{results_uri.rstrip('/')}/shard-*.jsonl.gz"
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shard outputs under {results_uri!r}")

    out: list[dict[str, Any]] = []
    protocol = results_uri.split("://", 1)[0] + "://" if "://" in results_uri else ""
    for p in paths:
        full = p if "://" in p else f"{protocol}{p}"
        with fsspec.open(full, "rb") as f:
            with gzip.open(f, "rt", encoding="utf-8") as gz:
                for line in gz:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
    logger.info("Read %d response records from %d shard files", len(out), len(paths))
    return out


def _group_completions(shard_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group shard responses by prompt-id-base (strip `#sample_idx` suffix)."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for rec in shard_records:
        rid = rec["id"]
        if "#" not in rid:
            raise ValueError(f"Expected id to contain '#': {rid!r}")
        base, j_str = rid.rsplit("#", 1)
        j = int(j_str)
        grouped.setdefault(base, []).append((j, rec))
    return {base: [item[1] for item in sorted(entries, key=lambda x: x[0])] for base, entries in grouped.items()}


def _write_completions_jsonl_gz(grouped: dict[str, list[dict[str, Any]]], output_uri: str) -> None:
    """Write old-schema {id, completions:[{text, metadata}, ...]} JSONL.gz to output_uri."""
    buffer = io.BytesIO()
    with gzip.open(buffer, "wt", encoding="utf-8") as gz:
        for base in sorted(grouped):
            comps = [
                {"text": e.get("response", ""), "metadata": e.get("extra", {})}
                for e in grouped[base]
            ]
            gz.write(json.dumps({"id": base, "completions": comps}, ensure_ascii=False) + "\n")
    with fsspec.open(output_uri, "wb") as f:
        f.write(buffer.getvalue())
    logger.info("Wrote completions to %s", output_uri)


def _grade_inline(
    prompts_uri: str,
    grouped_completions: dict[str, list[dict[str, Any]]],
    output_uri: str,
) -> tuple[int, int]:
    """Grade with lm-eval flexible_extract / strict_match filters, write grades.jsonl.gz.

    Returns (pass_at_32_count, total_problems) for a quick summary log.
    """
    import lm_eval.tasks
    from lm_eval.api.instance import Instance

    task = lm_eval.tasks.get_task_dict(["gsm8k"])["gsm8k"]
    filter_names = [f.name for f in task._filters]

    prompts_by_id = {row["id"]: row for row in read_prompt_rows(prompts_uri)}

    grade_rows: list[dict[str, Any]] = []
    pass_count = 0
    for prompt_id in sorted(grouped_completions):
        if prompt_id not in prompts_by_id:
            logger.warning("Completion id %s has no matching prompt; skipping.", prompt_id)
            continue
        prompt_row = prompts_by_id[prompt_id]
        doc = {"question": prompt_row["metadata"]["problem"], "answer": f"#### {prompt_row['ground_truth']}"}

        grades_for_prompt: list[dict[str, Any]] = []
        any_correct = False
        for completion_index, comp in enumerate(grouped_completions[prompt_id]):
            inst = Instance(
                request_type="generate_until",
                doc=doc,
                arguments=("", {}),
                idx=completion_index,
                task_name="gsm8k",
            )
            inst.resps = [comp.get("response", "")]
            task._instances = [inst]
            task.apply_filters()

            metadata: dict[str, Any] = {}
            score = 0.0
            for name in filter_names:
                key = name.replace("-", "_")
                filtered = inst.filtered_resps[name]
                correct = bool(task.process_results(doc, [filtered])["exact_match"])
                metadata[f"extraction_{key}"] = filtered
                metadata[f"correct_{key}"] = correct
                if key == "flexible_extract":
                    score = 1.0 if correct else 0.0
                    if correct:
                        any_correct = True
            if "correct_flexible_extract" not in metadata and filter_names:
                first_correct_key = f"correct_{filter_names[0].replace('-', '_')}"
                score = 1.0 if metadata[first_correct_key] else 0.0
                if score > 0:
                    any_correct = True
            grades_for_prompt.append({"score": score, "metadata": metadata})

        if any_correct:
            pass_count += 1
        grade_rows.append({"id": prompt_id, "grades": grades_for_prompt})

    buffer = io.BytesIO()
    with gzip.open(buffer, "wt", encoding="utf-8") as gz:
        for row in grade_rows:
            gz.write(json.dumps(row, ensure_ascii=False) + "\n")
    with fsspec.open(output_uri, "wb") as f:
        f.write(buffer.getvalue())
    logger.info("Wrote %d grade rows to %s", len(grade_rows), output_uri)
    return pass_count, len(grade_rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    args = parser.parse_args()

    if args.slug not in DELPHI_MIDTRAIN_MATRIX:
        raise ValueError(f"Unknown slug {args.slug!r}. Available: {sorted(DELPHI_MIDTRAIN_MATRIX)}")

    marin_prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-east5")
    rel_path = DELPHI_MIDTRAIN_MATRIX[args.slug]
    base_path = f"{marin_prefix}/{rel_path}"
    resolved = discover_hf_checkpoints(base_path)[-1]
    logger.info("Resolved %s -> %s", base_path, resolved)

    cell_root = f"{marin_prefix}/{OUTPUT_PREFIX}/{args.slug}"
    completions_uri = f"{cell_root}/completions.jsonl.gz"
    grades_uri = f"{cell_root}/grades.jsonl.gz"

    t0 = time.time()
    records = _materialize_records(PROMPTS_URI)

    sampling = SamplingParams(
        n=1,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        stop=STOP_TOKENS,
        seed=SEED,
    )
    model_spec = ModelSpec(
        model=resolved,
        engine_kwargs={
            "hf_overrides": {"architectures": ["Qwen3ForCausalLM"]},
            "trust_remote_code": True,
            "load_format": "runai_streamer",
        },
    )
    job_name_for_lib = f"mtxB-{args.slug.replace('_', '-').replace('.', 'p')}"
    config = InferenceConfig(
        regions=["us-east5"],
        results_region="us-east5",
        tpu_shapes=("v5p-8",),
        max_workers_per_region=8,
        shard_size=1024,
        sampling=sampling,
        job_name=job_name_for_lib,
        worker_preemptible=True,
        # Arch-shared compile cache (NOT per-model). Matches Method A's
        # shared-dir choice so the A/B comparison is apples-to-apples on
        # the compile-cache axis.
        compile_cache_uri_template="{region_prefix}/tmp/ttl=30d/vllm-cache/marin-eval",
    )

    logger.info("Calling inference() ...")
    t_inf_start = time.time()
    result = inference(model_spec, records, config)
    t_inf_end = time.time()
    logger.info(
        "inference() returned in %.1f s; results_uri=%s missing_shards=%s",
        t_inf_end - t_inf_start,
        result.results_uri,
        result.missing_shards,
    )
    if result.missing_shards:
        raise RuntimeError(f"Missing shards after inference: {result.missing_shards}")

    shard_records = _read_shard_outputs(result.results_uri)
    grouped = _group_completions(shard_records)
    _write_completions_jsonl_gz(grouped, completions_uri)

    pass_count, total = _grade_inline(PROMPTS_URI, grouped, grades_uri)

    t1 = time.time()
    pass_at_32 = (pass_count / total * 100.0) if total else 0.0
    logger.info("=" * 60)
    logger.info("CELL %s DONE", args.slug)
    logger.info("  pass@%d = %.2f%% (%d/%d)", N_SAMPLES, pass_at_32, pass_count, total)
    logger.info("  completions: %s", completions_uri)
    logger.info("  grades: %s", grades_uri)
    logger.info("  inference() wall-clock: %.1f s", t_inf_end - t_inf_start)
    logger.info("  total wall-clock: %.1f s", t1 - t0)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
