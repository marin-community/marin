# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SWE-ZERO multi-language scaling experiment (marin-community/marin#4653).

Samples K PRs per language across all 20 languages in SWE-rebench V2 (default
5 PRs per language → 100 PRs total) and runs N rollouts per PR (default 3) on
the same v6e-4 TP=4 + 32K-context recipe that the Python MVP landed on. Saves
the rollouts and a per-language report so we can see how the recipe transfers
beyond Python.

Important: this script intentionally avoids importing
``marin.evaluation.evaluators.evaluator`` and ``marin.inference.vllm_server``
because their import chain pulls in ``transformers → torch`` which on some
worker images is the CUDA build and crashes at module init on TPU. We start
the vLLM server directly via ``subprocess.Popen(["vllm", "serve", ...])``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
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
    languages: list[str] | None = None,
):
    """Group all SWE-rebench V2 instances by language and sample n_per_language from each.

    Returns a list of PRRecord objects, plus a per-language sampling summary.
    The seeded sampler is per-language so adding/removing a language doesn't
    perturb the others' samples.

    ``languages`` (optional) filters to a specific subset of languages.
    """
    import hashlib

    by_lang: dict[str, list[str]] = defaultdict(list)
    for row in loader._ds:
        by_lang[row.get("language", "unknown")].append(row["instance_id"])

    target_langs = sorted(by_lang.keys()) if languages is None else sorted(languages)
    summary: dict[str, dict] = {}
    sampled_prs = []
    for lang in target_langs:
        if lang not in by_lang:
            logger.warning("Skipping unknown language %s", lang)
            continue
        ids = sorted(by_lang[lang])  # stable order across runs
        # Use a stable hash of the language so the per-language seed is
        # reproducible across Python invocations. Built-in hash() randomizes
        # via PYTHONHASHSEED and would shuffle the per-language seed each run,
        # which breaks --resume-from.
        lang_seed = seed + int(hashlib.sha256(lang.encode()).hexdigest()[:8], 16) % 10_000
        rng = random.Random(lang_seed)
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


def _sample_all_prs_sharded(
    loader,
    seed: int,
    shard_index: int,
    total_shards: int,
):
    """Sample EVERY PR in SWE-rebench V2 (no per-language cap), partitioned across shards.

    Round-robin assignment over a globally-shuffled list of all instance_ids,
    so each shard sees a roughly proportional mix of all 20 languages and
    similar total work. Use ``shard_index`` 0..total_shards-1 to pick which
    slice this worker runs.

    Returns ``(sampled_prs, summary)`` matching ``_sample_prs_per_language``.
    """
    all_ids = sorted(loader._id_to_idx.keys())
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    my_ids = all_ids[shard_index::total_shards]

    by_lang: dict[str, list[str]] = defaultdict(list)
    sampled_prs = []
    for iid in my_ids:
        pr = loader.get(iid)
        sampled_prs.append(pr)
        by_lang[pr.language].append(iid)

    summary: dict[str, dict] = {}
    for lang, ids in by_lang.items():
        summary[lang] = {
            "available": -1,  # not meaningful in shard mode
            "sampled": len(ids),
            "instance_ids": ids,
        }
    summary["__shard__"] = {
        "shard_index": shard_index,
        "total_shards": total_shards,
        "total_prs_in_dataset": len(all_ids),
        "sampled": len(my_ids),
    }
    return sampled_prs, summary


def _remove_tpu_lockfile() -> None:
    """Best-effort delete of stale TPU lockfiles left by a prior aborted vLLM run.

    Mirrors ``marin.utils.remove_tpu_lockfile_on_exit`` but as a one-shot
    pre-start step so we recover after `--max-retries` re-attempts on the
    same Iris worker. The lockfile is the most common cause of "TPU device
    busy" / "open(/dev/vfio/N): Device or resource busy" failures.
    """
    for path in ("/tmp/libtpu_lockfile", "/tmp/libtpu.so_lockfile"):
        try:
            if os.path.exists(path):
                os.unlink(path)
                logger.info("Removed stale TPU lockfile: %s", path)
        except OSError as e:
            logger.warning("Could not remove %s: %s", path, e)


def _start_vllm_server(
    *,
    model_name_or_path: str,
    max_num_seqs: int,
    max_model_len: int,
    tensor_parallel_size: int,
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout_seconds: int = 1800,
) -> tuple[subprocess.Popen, str, str]:
    """Spawn `vllm serve` as a subprocess and wait until it answers /v1/models.

    Replicates ``marin.inference.vllm_server._start_vllm_native_server`` but
    in a way that doesn't transitively import ``transformers`` / ``torch``
    into THIS Python process. Returns ``(process, server_url, log_dir)``.
    """
    _remove_tpu_lockfile()
    vllm_bin = shutil.which("vllm") or "vllm"
    cmd = [
        vllm_bin,
        "serve",
        model_name_or_path,
        "--trust-remote-code",
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--enforce-eager",
    ]

    log_dir = tempfile.mkdtemp(prefix="vllm_server_")
    stdout_path = os.path.join(log_dir, "stdout.log")
    stderr_path = os.path.join(log_dir, "stderr.log")
    logger.info("Starting vLLM: %s", " ".join(cmd))
    logger.info("vLLM logs: %s", log_dir)

    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")
    process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
    server_url = f"http://{host}:{port}/v1"
    models_url = f"{server_url}/models"

    start = time.monotonic()
    while True:
        if process.poll() is not None:
            stdout_f.close()
            stderr_f.close()
            with open(stderr_path) as f:
                # 400 lines = enough to capture the *original* TPU/jax/sharding
                # error from a child engine-core process, not just the wrapper
                # tracebacks from the API server in the last few lines.
                stderr_tail = "".join(f.readlines()[-400:])
            raise RuntimeError(
                f"vLLM exited before becoming ready (rc={process.returncode}).\nstderr tail:\n{stderr_tail}"
            )
        try:
            with urllib.request.urlopen(models_url, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("vLLM ready at %s after %.1fs", server_url, time.monotonic() - start)
                    return process, server_url, log_dir
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.monotonic() - start > timeout_seconds:
            process.kill()
            raise TimeoutError(f"vLLM did not become ready within {timeout_seconds}s")
        time.sleep(5)


def _load_existing_rollouts(resume_from: str) -> dict[str, int]:
    """Load a partial rollouts.json and return ``{instance_id: count}``.

    Used when resuming from a previous (preempted) run — we look up how many
    rollouts each PR already has and only run the shortfall.
    """
    if resume_from.startswith("gs://"):
        import fsspec

        with fsspec.open(resume_from, "r") as f:
            data = json.load(f)
    else:
        with open(resume_from) as f:
            data = json.load(f)
    counts: dict[str, int] = defaultdict(int)
    for r in data:
        counts[r["instance_id"]] += 1
    logger.info("Resume: loaded %d existing rollouts across %d PRs from %s", len(data), len(counts), resume_from)
    return dict(counts)


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
    resume_from: str | None = None,
    languages: list[str] | None = None,
    all_prs: bool = False,
    shard_index: int = 0,
    total_shards: int = 1,
    dataset_id: str = "nebius/SWE-rebench-V2",
) -> None:
    # Local imports of swe_zero modules only — no marin.evaluation /
    # marin.inference imports, to avoid pulling in transformers / torch.
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

    loader = SWERebenchV2Loader(dataset_id=dataset_id)
    if all_prs:
        sampled_prs, summary = _sample_all_prs_sharded(
            loader, seed=seed, shard_index=shard_index, total_shards=total_shards
        )
        logger.info(
            "Sharded ALL-PR sampling: shard %d/%d, %d PRs",
            shard_index,
            total_shards,
            len(sampled_prs),
        )
    else:
        sampled_prs, summary = _sample_prs_per_language(loader, n_per_language, seed, languages=languages)
        logger.info("Sampling summary: %d total PRs across %d languages", len(sampled_prs), len(summary))
        for lang, info in sorted(summary.items()):
            logger.info("  %-10s: %d sampled / %d available", lang, info["sampled"], info["available"])

    # Auto-resume from output_dir/rollouts.json if it exists and no explicit
    # --resume-from was given. This makes sharded jobs idempotent under preemption:
    # a re-launched shard reads its own previous incremental save and only runs
    # the per-PR shortfall.
    auto_resume_path = os.path.join(output_dir, "rollouts.json")
    if resume_from is None:
        try:
            if auto_resume_path.startswith("gs://"):
                import fsspec

                fs, _ = fsspec.core.url_to_fs(auto_resume_path)
                if fs.exists(auto_resume_path):
                    resume_from = auto_resume_path
                    logger.info("Auto-resume: found existing rollouts at %s", auto_resume_path)
            elif os.path.exists(auto_resume_path):
                resume_from = auto_resume_path
                logger.info("Auto-resume: found existing rollouts at %s", auto_resume_path)
        except Exception as e:
            logger.warning("Auto-resume probe failed (will run full shard): %s", e)

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

    # Resume mode: subtract already-completed rollouts per PR. Only PRs with
    # shortfall vs n_rollouts_per_pr get a batch in the new run.
    # Merge counts from both rollouts.json and rollouts_resume.json so that
    # preempted retries don't redo work already saved in the resume file.
    existing_counts: dict[str, int] = {}
    if resume_from:
        existing_counts = _load_existing_rollouts(resume_from)
        resume_path = os.path.join(output_dir, "rollouts_resume.json")
        if resume_path != resume_from:
            try:
                resume_counts = _load_existing_rollouts(resume_path)
                for k, v in resume_counts.items():
                    existing_counts[k] = existing_counts.get(k, 0) + v
                logger.info("Resume: merged %d additional rollouts from %s", sum(resume_counts.values()), resume_path)
            except Exception:
                pass  # no resume file yet, that's fine

    batches = []
    for pr in sampled_prs:
        already = existing_counts.get(pr.instance_id, 0)
        remaining = max(0, n_rollouts_per_pr - already)
        if remaining > 0:
            batches.append(RolloutBatch(pr=pr, n_rollouts=remaining))
    total = sum(b.n_rollouts for b in batches)
    logger.info("Will run %d total rollouts (concurrency=%d, %d PRs need work)", total, concurrency, len(batches))
    if resume_from:
        skipped = sum(1 for pr in sampled_prs if existing_counts.get(pr.instance_id, 0) >= n_rollouts_per_pr)
        logger.info("Resume: skipping %d PRs that are already complete", skipped)
    logger.info(
        "vLLM config: max_model_len=%d, max_num_seqs=%d, TP=%d, max_total_tokens=%d, concurrency=%d",
        max_model_len,
        max_num_seqs,
        tensor_parallel_size,
        max_total_tokens,
        concurrency,
    )

    # In resume mode write the new rollouts to a separate file so we don't
    # clobber the partial we're resuming from. Pre-load any existing resume
    # file so we append rather than overwrite on preemption retries.
    output_filename = "rollouts_resume.json" if resume_from else "rollouts.json"
    prior_dicts: list[dict] = []
    if resume_from:
        resume_out_path = os.path.join(output_dir, output_filename)
        try:
            if resume_out_path.startswith("gs://"):
                import fsspec

                fs, _ = fsspec.core.url_to_fs(resume_out_path)
                if fs.exists(resume_out_path):
                    with fs.open(resume_out_path) as f:
                        prior_dicts = json.load(f)
            elif os.path.exists(resume_out_path):
                with open(resume_out_path) as f:
                    prior_dicts = json.load(f)
            if prior_dicts:
                logger.info("Resume: pre-loaded %d rollouts from %s for append", len(prior_dicts), resume_out_path)
        except Exception:
            pass
    completed_rollouts: list[Rollout] = []
    save_interval = max(1, min(10, total // 20)) if total else 1

    def _on_rollout_done(done: int, total_n: int, rollout: Rollout) -> None:
        completed_rollouts.append(rollout)
        if done % save_interval == 0 or done == total_n:
            save_json(
                prior_dicts + [r.to_dict() for r in completed_rollouts],
                os.path.join(output_dir, output_filename),
            )
            logger.info("Checkpoint: saved %d/%d rollouts", done, total_n)

    process, server_url, log_dir = _start_vllm_server(
        model_name_or_path=model_path,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )
    try:
        rollouts = run_rollouts_concurrently(
            api_base=server_url,
            api_key="EMPTY",
            model=model_name,
            batches=batches,
            concurrency=concurrency,
            temperature=1.0,
            max_total_tokens=max_total_tokens,
            progress_callback=_on_rollout_done,
        )
    finally:
        logger.info("Stopping vLLM (logs at %s)", log_dir)
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()

    save_json(prior_dicts + [r.to_dict() for r in rollouts], os.path.join(output_dir, output_filename))

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
    if total_n > 0:
        logger.info("Overall submission rate: %d/%d (%.1f%%)", total_finished, total_n, 100 * total_finished / total_n)
    else:
        logger.info("No rollouts produced (all PRs already at target via auto-resume)")


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
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path (local or gs://) to a partial rollouts.json from a previous run. "
        "Per-PR shortfalls will be filled in; PRs already at n-rollouts are skipped.",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated list of languages to sample (e.g. scala,swift,ts). "
        "Default: all 20 languages in SWE-rebench V2.",
    )
    parser.add_argument(
        "--dataset",
        default="nebius/SWE-rebench-V2",
        help="HuggingFace dataset ID. Also supports nebius/SWE-rebench-V2-PRs (126K PRs).",
    )
    parser.add_argument(
        "--all-prs",
        action="store_true",
        help="Sample EVERY PR in the dataset instead of n-per-language. "
        "Combine with --shard-index/--total-shards or Iris --replicas to partition "
        "the corpus across workers.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based shard index for --all-prs mode. If omitted, auto-detected "
        "from IRIS_TASK_ID env var (set by Iris --replicas). PRs are round-robin "
        "assigned to shards over a globally-shuffled instance_id list.",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=None,
        help="Total number of shards for --all-prs mode. If omitted, auto-detected "
        "from IRIS_NUM_TASKS env var (set by Iris --replicas).",
    )
    args = parser.parse_args()

    model_path = args.model_path or args.model

    if not args.local:
        raise SystemExit("Pass --local to run on a TPU worker via VllmEnvironment.")

    # Resolve shard identity: explicit CLI flags take precedence; fall back to
    # Iris env vars (set automatically by `iris job run --replicas N`); finally
    # default to single-shard mode.
    shard_index = args.shard_index
    total_shards = args.total_shards
    if shard_index is None or total_shards is None:
        iris_task_id = os.environ.get("IRIS_TASK_ID", "")
        iris_num_tasks = os.environ.get("IRIS_NUM_TASKS", "")
        if iris_task_id and iris_num_tasks:
            # IRIS_TASK_ID format: /user/job-name/INDEX or /user/job-name/INDEX:attempt
            idx_part = iris_task_id.rsplit("/", 1)[-1].split(":")[0]
            shard_offset = int(os.environ.get("SHARD_OFFSET", "0"))
            shard_index = shard_index if shard_index is not None else int(idx_part) + shard_offset
            total_shards = total_shards if total_shards is not None else int(iris_num_tasks)
            logger.info(
                "Auto-detected shard identity from Iris: shard_index=%d, total_shards=%d (IRIS_TASK_ID=%s)",
                shard_index,
                total_shards,
                iris_task_id,
            )
        else:
            shard_index = shard_index if shard_index is not None else 0
            total_shards = total_shards if total_shards is not None else 1

    # In --all-prs mode with multiple shards, auto-suffix the output_dir so
    # each shard writes to its own prefix (e.g., .../shard_003_of_050/).
    output_dir = args.output_dir
    if args.all_prs and total_shards > 1:
        output_dir = os.path.join(output_dir, f"shard_{shard_index:03d}_of_{total_shards:03d}")

    _run_with_vllm(
        model_name=args.model,
        model_path=model_path,
        output_dir=output_dir,
        n_per_language=args.n_per_language,
        n_rollouts_per_pr=args.n_rollouts,
        seed=args.seed,
        concurrency=args.concurrency,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_total_tokens=args.max_total_tokens,
        resume_from=args.resume_from,
        languages=[s.strip() for s in args.languages.split(",")] if args.languages else None,
        all_prs=args.all_prs,
        shard_index=shard_index,
        total_shards=total_shards,
        dataset_id=args.dataset,
    )


if __name__ == "__main__":
    main()
