# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SWE-ZERO multi-language scaling experiment (marin-community/marin#4653).

Samples K PRs per language across all 20 languages in SWE-rebench V2 (default
5 PRs per language → 100 PRs total) and runs N rollouts per PR (default 3) on
the same v6e-4 TP=4 + 32K-context recipe that the Python MVP landed on. Saves
the rollouts and a per-language report so we can see how the recipe transfers
beyond Python.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def save_json(data, output_path: str) -> None:
    if output_path.startswith("gs://"):
        import fsspec

        with fsspec.open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def _sample_prs_per_language(
    loader,
    n_per_language: int,
    seed: int,
):
    """Group all SWE-rebench V2 instances by language and sample n_per_language from each.

    Returns a list of PRRecord objects, plus a per-language sampling summary.
    The seeded sampler is per-language so adding/removing a language doesn't
    perturb the others' samples.
    """
    by_lang: dict[str, list[str]] = defaultdict(list)
    for row in loader._ds:
        by_lang[row.get("language", "unknown")].append(row["instance_id"])

    summary: dict[str, dict] = {}
    sampled_prs = []
    for lang in sorted(by_lang.keys()):
        ids = by_lang[lang]
        rng = random.Random(seed + hash(lang) % 10_000)
        n = min(n_per_language, len(ids))
        chosen = rng.sample(ids, n)
        for iid in chosen:
            sampled_prs.append(loader.get(iid))
        summary[lang] = {
            "available": len(ids),
            "sampled": n,
            "instance_ids": chosen,
        }
    return sampled_prs, summary


def _per_language_metrics(rollouts) -> dict:
    """Aggregate per-language stats from a list of Rollout objects."""
    by_lang: dict[str, list] = defaultdict(list)
    for r in rollouts:
        # We attached the language to the PRRecord; look it up via the worktree
        # info we stored as `extra` (see `Rollout.steps`)... actually we don't
        # store language on the rollout, so we look it up via instance_id later.
        by_lang.setdefault(r.instance_id, []).append(r)
    return {}  # filled below by the caller that has the language map


def _run_with_vllm(
    model_name: str,
    model_path: str,
    output_dir: str,
    n_per_language: int,
    n_rollouts_per_pr: int,
    seed: int,
    concurrency: int,
    max_num_seqs: int,
    max_model_len: int,
    tensor_parallel_size: int,
    max_total_tokens: int,
) -> None:
    from marin.evaluation.evaluators.evaluator import ModelConfig
    from marin.inference.vllm_server import VllmEnvironment
    from marin.utils import remove_tpu_lockfile_on_exit

    from experiments.swe_zero.data_loader import SWERebenchV2Loader
    from experiments.swe_zero.diversity import measure_diversity
    from experiments.swe_zero.rollout_generator import (
        Rollout,
        RolloutBatch,
        run_rollouts_concurrently,
    )

    os.environ.setdefault("VLLM_TPU_SKIP_PRECOMPILE", "1")
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    os.environ.setdefault("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1")

    # Load the WHOLE dataset (no language filter — we want all 20).
    loader = SWERebenchV2Loader()
    sampled_prs, summary = _sample_prs_per_language(loader, n_per_language, seed)
    logger.info("Sampling summary: %d total PRs across %d languages", len(sampled_prs), len(summary))
    for lang, info in sorted(summary.items()):
        logger.info("  %-10s: %d sampled / %d available", lang, info["sampled"], info["available"])

    # Save the sampling plan up front so we can reproduce / inspect even if
    # the rollout phase crashes mid-run.
    save_json(
        {
            "n_per_language": n_per_language,
            "n_rollouts_per_pr": n_rollouts_per_pr,
            "seed": seed,
            "languages": summary,
            "total_prs": len(sampled_prs),
        },
        os.path.join(output_dir, "sampling_plan.json"),
    )

    pr_to_language = {pr.instance_id: pr.language for pr in sampled_prs}
    batches = [RolloutBatch(pr=pr, n_rollouts=n_rollouts_per_pr) for pr in sampled_prs]
    total = sum(b.n_rollouts for b in batches)
    logger.info("Will run %d total rollouts (concurrency=%d)", total, concurrency)

    model_config = ModelConfig(
        name=model_name,
        path=model_path,
        engine_kwargs={"max_model_len": max_model_len},
    )
    extra_args = [
        "--max-num-seqs",
        str(max_num_seqs),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--enforce-eager",
    ]
    logger.info(
        "vLLM config: max_model_len=%d, max_num_seqs=%d, TP=%d, max_total_tokens=%d, concurrency=%d",
        max_model_len,
        max_num_seqs,
        tensor_parallel_size,
        max_total_tokens,
        concurrency,
    )

    completed_rollouts: list[Rollout] = []
    save_interval = max(10, min(50, total // 20))

    def _on_rollout_done(done: int, total_n: int, rollout: Rollout) -> None:
        completed_rollouts.append(rollout)
        if done % save_interval == 0 or done == total_n:
            save_json(
                [r.to_dict() for r in completed_rollouts],
                os.path.join(output_dir, "rollouts.json"),
            )
            logger.info("Checkpoint: saved %d/%d rollouts", done, total_n)

    with remove_tpu_lockfile_on_exit(), VllmEnvironment(model_config, extra_args=extra_args) as env:
        logger.info("vLLM server ready at %s", env.server_url)
        rollouts = run_rollouts_concurrently(
            api_base=env.server_url,
            api_key="EMPTY",
            model=model_name,
            batches=batches,
            concurrency=concurrency,
            temperature=1.0,
            max_total_tokens=max_total_tokens,
            progress_callback=_on_rollout_done,
        )

    save_json([r.to_dict() for r in rollouts], os.path.join(output_dir, "rollouts.json"))

    # Per-language report
    by_lang_rollouts: dict[str, list] = defaultdict(list)
    for r in rollouts:
        lang = pr_to_language.get(r.instance_id, "unknown")
        by_lang_rollouts[lang].append(r)

    per_lang_report = {}
    for lang in sorted(by_lang_rollouts.keys()):
        rs = by_lang_rollouts[lang]
        n = len(rs)
        n_finished = sum(1 for r in rs if r.finished)
        n_errored = sum(1 for r in rs if r.error)
        mean_turns = sum(sum(1 for s in r.steps if s.role == "assistant") for r in rs) / n
        mean_completion = sum(r.total_completion_tokens for r in rs) / n
        # Top first-words
        first_words: Counter = Counter()
        for r in rs:
            for s in r.steps:
                if s.role == "assistant" and s.bash_command:
                    fw = s.bash_command.strip().split()[0] if s.bash_command.strip() else ""
                    first_words[fw] += 1
        # cmd not found
        n_cnf = 0
        n_obs = 0
        for r in rs:
            for s in r.steps:
                if s.role == "user" and (s.content or "").startswith("Observation:"):
                    n_obs += 1
                    if "command not found" in s.content:
                        n_cnf += 1
        # diversity
        try:
            div = measure_diversity(rs)
            div_unique = div.n_unique
            div_mean_jacc = div.mean_pairwise_jaccard
        except Exception as e:
            logger.warning("diversity failed for %s: %s", lang, e)
            div_unique = None
            div_mean_jacc = None
        per_lang_report[lang] = {
            "n_rollouts": n,
            "n_prs": len({r.instance_id for r in rs}),
            "submission_rate": n_finished / n if n else 0.0,
            "n_finished": n_finished,
            "n_errored": n_errored,
            "mean_turns_per_rollout": round(mean_turns, 1),
            "mean_completion_tokens_per_rollout": round(mean_completion, 0),
            "command_not_found_observations": n_cnf,
            "command_not_found_rate": round(n_cnf / n_obs, 4) if n_obs else 0.0,
            "top_first_words": first_words.most_common(15),
            "diversity_unique_at_jaccard_05": div_unique,
            "diversity_mean_pairwise_jaccard": round(div_mean_jacc, 4) if div_mean_jacc is not None else None,
        }

    # Aggregate
    total_n = len(rollouts)
    total_finished = sum(1 for r in rollouts if r.finished)
    summary_payload = {
        "total_rollouts": total_n,
        "total_prs": len({r.instance_id for r in rollouts}),
        "languages_covered": len(by_lang_rollouts),
        "overall_submission_rate": total_finished / total_n if total_n else 0.0,
        "per_language": per_lang_report,
    }
    save_json(summary_payload, os.path.join(output_dir, "multilang_report.json"))
    logger.info("=== Per-language summary ===")
    for lang in sorted(per_lang_report.keys()):
        r = per_lang_report[lang]
        logger.info(
            "  %-10s: subs=%d/%d (%.0f%%) turns=%.1f compl_tok=%.0f cnf=%d/%d top=%s",
            lang,
            r["n_finished"],
            r["n_rollouts"],
            100 * r["submission_rate"],
            r["mean_turns_per_rollout"],
            r["mean_completion_tokens_per_rollout"],
            r["command_not_found_observations"],
            r["n_rollouts"],
            r["top_first_words"][:3],
        )
    logger.info("Overall submission rate: %d/%d (%.1f%%)", total_finished, total_n, 100 * total_finished / total_n)


def main():
    parser = argparse.ArgumentParser(description="SWE-ZERO multi-language scaling experiment")
    parser.add_argument("--model", default="ricdomolm/mini-coder-1.7b")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--n-per-language", type=int, default=5)
    parser.add_argument("--n-rollouts", type=int, default=3, help="Rollouts per PR")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--output_dir", default="gs://marin-us-central2/experiments/swe_zero_multilang")
    parser.add_argument("--local", action="store_true", help="Run locally with VllmEnvironment")
    args = parser.parse_args()

    model_path = args.model_path or args.model

    if not args.local:
        raise SystemExit("Pass --local to run on a TPU worker via VllmEnvironment.")

    _run_with_vllm(
        model_name=args.model,
        model_path=model_path,
        output_dir=args.output_dir,
        n_per_language=args.n_per_language,
        n_rollouts_per_pr=args.n_rollouts,
        seed=args.seed,
        concurrency=args.concurrency,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_total_tokens=args.max_total_tokens,
    )


if __name__ == "__main__":
    main()
