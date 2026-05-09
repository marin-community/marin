# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-region parquets staged in GCS, dedup, and upload as a clean HF dataset.

Multi-region refresh round flow:

  1. Per-region jobs (`upload_trajectories_12m.py --parquet-gcs-out gs://...`) build
     their partition's parquets and stage them in their REGIONAL bucket. No HF push.
  2. This aggregator runs in any single region (us-east5 by default), pulls parquets
     from all 4 regional buckets (small per-round transfer - parquets are dense),
     dedups at trajectory-hash level, and uploads a single clean
     `data/train-NNNNN.parquet` set to HF - replacing whatever was there.

Cross-region invariant: this aggregator is the ONE place per refresh round where
cross-region GCS reads happen, bounded to ~5-8 GB per round (parquet form). The
generation pipeline itself never crosses regions.

Usage::

    uv run iris --cluster marin job run \\
        --memory 64GB --cpu 8 --disk 200GB --priority interactive --no-wait \\
        --enable-extra-resources \\
        --job-name swe-zero-aggregator-vN \\
        --region us-east5 \\
        -e HF_TOKEN ${HF_TOKEN} \\
        -- python scripts/aggregate_and_upload_hf.py \\
        --hf-repo AlienKevin/SWE-ZERO-12M-trajectories \\
        --round-id N
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationDelete, HfApi

# Region → GCS bucket mapping. Fixed by the pipeline contract; mirrors
# experiments/swe_zero/run_swe_zero_swarm_multiregion.py:PINNED_REGIONS.
REGIONS = ["us-east5", "us-east1", "us-west4", "us-central1"]
PARQUET_ROWS_PER_SHARD = 10_000

logger = logging.getLogger(__name__)


def _rollout_hash(instance_id: str, messages) -> str:
    """Match `_rollout_hash` from upload_trajectories_12m.py - md5 of (id, sorted messages)."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    payload = json.dumps({"instance_id": instance_id, "messages": msgs}, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _staging_path(region: str, round_id: str) -> str:
    return f"gs://marin-{region}/experiments/swe_zero_100b/parquet-round-{round_id}/"


def _download_region_parquets(region: str, round_id: str, dest: Path) -> list[Path]:
    """Pull all parquet files from a region's staging path into `dest`."""
    import fsspec

    src = _staging_path(region, round_id)
    fs = fsspec.filesystem("gs")
    if not fs.exists(src):
        logger.warning("%s does not exist (region had no staged parquets this round)", src)
        return []
    region_dest = dest / region
    region_dest.mkdir(parents=True, exist_ok=True)
    files = [f for f in fs.ls(src) if f.endswith(".parquet")]
    logger.info("  %s: %d parquets", src, len(files))
    pulled: list[Path] = []
    for f in files:
        local = region_dest / Path(f).name
        with fs.open(f, "rb") as src_f, open(local, "wb") as out:
            while True:
                chunk = src_f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        pulled.append(local)
    return pulled


MAX_ROLLOUTS_PER_PR = 100


def stream_dedup(parquet_files: list[Path], out_dir: Path) -> dict:
    """Read parquets, dedup by trajectory hash, cap at MAX_ROLLOUTS_PER_PR per PR,
    write `train-NNNNN.parquet` chunks.

    The static `shard_idx % 4` partition guarantees each PR's rollouts live in
    exactly one region, so capping here is equivalent to capping per-region -
    no cross-region overlap to worry about.
    """
    seen: set[str] = set()
    pr_counts: Counter = Counter()
    pr_repos: dict[str, str] = {}
    qwen3_tokens_total = 0
    qwen3_tokens_per_row: list[int] = []  # for median
    by_exit: Counter = Counter()
    n_raw = 0
    n_dups = 0
    n_capped = 0
    n_errors = 0
    schema: pa.Schema | None = None
    buffer: list[dict] = []
    out_idx = 0

    out_dir.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        nonlocal out_idx, schema
        if not buffer:
            return
        if schema is None:
            schema = pa.Table.from_pylist(buffer).schema
        table = pa.Table.from_pylist(buffer, schema=schema)
        path = out_dir / f"train-{out_idx:05d}.parquet"
        pq.write_table(table, path, compression="zstd", compression_level=3)
        logger.info("  wrote %s (%d rows)", path.name, len(buffer))
        buffer.clear()
        out_idx += 1

    for pf in parquet_files:
        table = pq.read_table(pf)
        if schema is None:
            schema = table.schema
        rows = table.to_pylist()
        n_raw += len(rows)
        for r in rows:
            iid = r.get("instance_id") or ""
            messages = r.get("messages") or []
            h = _rollout_hash(iid, messages)
            if h in seen:
                n_dups += 1
                continue
            seen.add(h)
            if iid and pr_counts[iid] >= MAX_ROLLOUTS_PER_PR:
                n_capped += 1
                continue
            buffer.append(r)
            pr_counts[iid] += 1
            pr_repos.setdefault(iid, r.get("repo") or "")
            t = int(r.get("qwen3_tokens") or 0)
            qwen3_tokens_total += t
            qwen3_tokens_per_row.append(t)
            by_exit[r.get("exit_status") or ""] += 1
            if not messages or not any(m.get("role") == "assistant" for m in messages):
                n_errors += 1
            if len(buffer) >= PARQUET_ROWS_PER_SHARD:
                flush()
    flush()

    import statistics

    n_kept = sum(pr_counts.values())
    n_prs_at_target_100 = sum(1 for c in pr_counts.values() if c >= 100)
    n_repos = len({v for v in pr_repos.values() if v})
    submitted_pct = (by_exit.get("Submitted", 0) / n_kept * 100.0) if n_kept else 0.0
    median_qwen3_tokens = int(statistics.median(qwen3_tokens_per_row)) if qwen3_tokens_per_row else 0

    return {
        "n_raw": n_raw,
        "n_kept": n_kept,
        "n_dups": n_dups,
        "n_capped": n_capped,
        "n_prs": len(pr_counts),
        "n_prs_at_target_100": n_prs_at_target_100,
        "n_repos": n_repos,
        "qwen3_tokens": qwen3_tokens_total,
        "median_qwen3_tokens": median_qwen3_tokens,
        "submitted_pct": submitted_pct,
        "n_errors": n_errors,
        "n_parquet_shards": out_idx,
    }


def render_readme(stats: dict) -> str:
    n_rollouts = stats["n_kept"]
    qwen3_tokens_b = stats["qwen3_tokens"] / 1e9
    size_cat = "1M<n<10M" if n_rollouts >= 1_000_000 else "100K<n<1M"
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- swe-zero
- code
- agentic
- pre-training
- qwen3
size_categories:
- {size_cat}
---

# SWE-ZERO 12M Trajectories

Execution-free agentic code-editing trajectories from the SWE-ZERO swarm pipeline
([marin-community/marin#4898](https://github.com/marin-community/marin/issues/4898)),
tokenized with the Qwen3-0.6B-Base tokenizer.

## Snapshot

| Metric | Value |
|--------|-------|
| Rollouts | {n_rollouts:,} |
| Unique PRs | {stats['n_prs']:,} of 126,000 source ({stats['n_prs'] / 126_000 * 100:.1f}%) |
| PRs at target (≥100 rollouts) | {stats['n_prs_at_target_100']:,} |
| Repos covered | {stats['n_repos']:,} |
| Total tokens | {qwen3_tokens_b:.2f} B |
| Median trace length (Qwen3 tokens, chat-templated) | {stats.get('median_qwen3_tokens', 0):,} |
| Submission rate (`exit_status == Submitted`) | {stats['submitted_pct']:.2f}% |
| Per-PR cap (rollouts dropped beyond 100/PR) | {stats.get('n_capped', 0):,} |
| Source | `nebius/SWE-rebench-V2-PRs` |
| Generator | `ricdomolm/mini-coder-1.7b` (vLLM, TPU, max_seq_len=32768) |
| Format | `mini-swe-agent-1` |

## Schema

| field | type |
|-------|------|
| `instance_id` | string - PR identifier |
| `repo` | string - `owner/name` |
| `messages` | list[{{role, content}}] - multi-turn agentic trajectory |
| `trajectory_format` | string - `mini-swe-agent-1` |
| `exit_status` | string - `Submitted`, `incomplete`, etc. |
| `qwen3_tokens` | int64 - exact Qwen3-tokenizer count over the chat-templated trajectory |
| `prompt_tokens` | int64 - generator prompt tokens |
| `completion_tokens` | int64 - generator completion tokens |
| `duration_sec` | float64 - wall-clock generation time |

## Sampling settings

| Setting | Value |
|---------|-------|
| Max turns per rollout | 15 |
| Rollouts per PR (target) | 100 |
| Sampling temperature | 1.0 |
| Max model length (vLLM) | 32,768 tokens |
| Max sequences (vLLM) | 256 |
| Max total tokens per rollout | 32,768 |

## Related datasets

- [`SWE-ZERO-96K-trajectories`][96k] - 96 K rollouts from the 1B scaling run (#4666)
- [`SWE-ZERO-multilang-300-trajectories`][300] - 300 rollouts from multilang validation (#4653)

[96k]: https://huggingface.co/datasets/AlienKevin/SWE-ZERO-96K-trajectories
[300]: https://huggingface.co/datasets/AlienKevin/SWE-ZERO-multilang-300-trajectories
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="AlienKevin/SWE-ZERO-12M-trajectories")
    parser.add_argument(
        "--round-id",
        required=True,
        help=(
            "Round identifier; per-region jobs staged at "
            "gs://marin-{region}/experiments/swe_zero_100b/parquet-round-{round_id}/"
        ),
    )
    parser.add_argument("--workdir", default="/tmp/hf-aggregate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set")
        return 1

    api = HfApi(token=token)
    workdir = Path(args.workdir)
    snap_dir = workdir / "regional"
    out_dir = workdir / "deduped"

    logger.info("Step 1/4: pulling parquets from %d regional staging paths (round=%s)", len(REGIONS), args.round_id)
    all_parquets: list[Path] = []
    for region in REGIONS:
        all_parquets.extend(_download_region_parquets(region, args.round_id, snap_dir))
    logger.info("  pulled %d parquet files total", len(all_parquets))

    if not all_parquets:
        logger.error("No parquets found across any region for round %s - nothing to upload", args.round_id)
        return 1

    logger.info("Step 2/4: streaming dedup")
    stats = stream_dedup(all_parquets, out_dir)
    logger.info("  raw=%d kept=%d dups=%d", stats["n_raw"], stats["n_kept"], stats["n_dups"])

    logger.info("Step 3/4: replacing data/ on HF (delete + upload)")
    api.create_commit(
        repo_id=args.hf_repo,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo="data/", is_folder=True)],
        commit_message=f"Round {args.round_id}: clear old parquets before aggregating",
    )
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=args.hf_repo,
        repo_type="dataset",
        path_in_repo="data",
        commit_message=(
            f"Round {args.round_id}: {stats['n_kept']:,} unique rollouts, "
            f"{stats['qwen3_tokens']/1e9:.2f}B Qwen3 tokens, "
            f"{stats['n_dups']:,} dups removed"
        ),
    )

    logger.info("Step 4/4: writing README")
    readme_text = render_readme(stats)
    readme_path = workdir / "README.md"
    readme_path.write_text(readme_text)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.hf_repo,
        repo_type="dataset",
        commit_message=f"Round {args.round_id}: update README",
    )

    print("=" * 70)
    print(f"AGGREGATE ROUND {args.round_id} SUMMARY")
    print(json.dumps(stats, indent=2))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
