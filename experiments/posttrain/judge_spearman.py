#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-statement Spearman rank correlation among GPT-4.1, GPT-5.1, and GPT-oss-120B judges.

Why Spearman and not Pearson
----------------------------
Our judges produce integer ratings on a 1-10 ordinal scale (see the scoring
guide in `lib/marin/src/marin/alignment/prompts/judge.py` — 1=clearly non-
compliant, 10=exemplary). Nothing in the rubric actually commits to "the
distance between a 6 and a 7 equals the distance between 2 and 3", which is
the interval-scale assumption Pearson implicitly requires. Spearman only
looks at rank order, so it's agnostic to that assumption. It's also the
metric the EXP-025 terminology note in validate_bloom_claude.md should have
started with.

What "per-statement" means here
-------------------------------
For each of the 46 statements in the spec, we collect every item where both
judges emitted a non-parse-failure score (pooling across the 4 target models:
sft, full_dpo_beta01_b64_step1699, lora_lr1e5_b64_step1699,
lora_lr5e6_b64_step1699). That gives ~680 paired items per statement (4
targets x ~170 prompts). We compute one Spearman per (judge-pair, statement).
The headline number is the median of the 46 per-statement Spearmans.

This matches the unit of analysis in EXP-026 (pool across targets within a
statement) — NOT a single pooled correlation across all 46 x 4 x 170 items,
which would inflate the number with between-statement rank agreement.

Data layout (persistent, ~/judge_correlations is not /tmp)
-----------------------------------------------------------
    ~/judge_correlations/
        inputs/
            gpt41/{target}/judged_results.jsonl     copied from gpt51_batch
            gpt51/{target}/judged_results.jsonl     copied from gpt51_batch (post-collect)
            goss/{target}/shard_*.jsonl.gz          downloaded from GCS
        outputs/
            spearman_per_statement.json             full per-statement table
            score_histograms.json                   per-judge score distribution

Usage
-----
    # One-time setup: populate inputs/ from local gpt51_batch + GCS
    uv run python experiments/posttrain/judge_spearman.py download

    # Run the analysis (2-way works with just gpt41 + goss; 3-way once gpt51 is collected)
    uv run python experiments/posttrain/judge_spearman.py analyze
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "lib" / "marin" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "lib" / "rigging" / "src"))

from rigging.filesystem import url_to_fs  # noqa: E402
from together_paths import DATA_ROOT as TOGETHER_DATA_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_ROOT = Path.home() / "judge_correlations"
GPT51_BATCH_ROOT = Path.home() / "gpt51_batch"
GEM3F_BATCH_ROOT = Path.home() / "gemini3flash_batch"
GEM31P_BATCH_ROOT = Path.home() / "gemini31pro_batch"
# EXP-030: Together-hosted judges now archive runs under
# ~/together_batch/{run_id}/{model_slug}/{target}/ and publish the latest full
# run through ~/together_batch/latest/{model_slug}/{target}/.
TOGETHER_LATEST_ROOT = TOGETHER_DATA_ROOT / "latest"
TOGETHER_LEGACY_ROOT = TOGETHER_DATA_ROOT
KIMI25_BATCH_ROOT = TOGETHER_LATEST_ROOT / "moonshotai_Kimi-K2.5"
QWEN235_BATCH_ROOT = TOGETHER_LATEST_ROOT / "Qwen_Qwen3-235B-A22B-Instruct-2507-tput"
GLM51_BATCH_ROOT = TOGETHER_LATEST_ROOT / "zai-org_GLM-5.1"
GLM5_BATCH_ROOT = TOGETHER_LATEST_ROOT / "zai-org_GLM-5"
QWEN397_BATCH_ROOT = TOGETHER_LATEST_ROOT / "Qwen_Qwen3.5-397B-A17B"
MM25_BATCH_ROOT = TOGETHER_LATEST_ROOT / "MiniMaxAI_MiniMax-M2.5"
MM27_BATCH_ROOT = TOGETHER_LATEST_ROOT / "MiniMaxAI_MiniMax-M2.7"

TARGETS = [
    "sft",
    "full_dpo_beta01_b64_step1699",
    "lora_lr1e5_b64_step1699",
    "lora_lr5e6_b64_step1699",
    "gpt41_target",
]

# GPT-oss-120B judge output (sharded JSONL.GZ per target) lives here from EXP-021.
GOSS_GCS_PREFIX = "gs://marin-us-central1/eval/judge_goss120b_batch-fd3ffe"

JUDGES = (
    "gpt41",
    "gpt51",
    "goss",
    "gem3f",
    "gem31p",
    "kimi25",
    "qwen235",
    "glm51",
    "glm5",
    "qwen397",
    "mm25",
    "mm27",
)

# Statements excluded from cross-judge pair analysis.
#
# `support_programmatic_use` and `formatting` are excluded from ALL pairs:
# Gemini has JSON-escaping bugs on code/markdown content there (26-58%
# parse failure on Flash). Dropping them for gpt41<->gpt51 too keeps all
# pairs on the same statement basis so the numbers are directly
# comparable. These must match the SKIP_STATEMENTS set in both
# judge_gemini3flash.py and judge_gemini31pro.py.
UNIVERSAL_SKIP_STATEMENTS = frozenset(
    {
        "support_programmatic_use",
        "formatting",
    }
)

# Additional exclusion that applies only to pairs involving a Gemini
# judge (gem3f or gem31p). Both models' safety filters refuse on
# `sexual_content_involving_minors`, producing empty content. GPT-4.1
# and GPT-5.1 don't have this problem, so we keep the statement for the
# gpt41<->gpt51 pair.
GEMINI_JUDGES = frozenset({"gem3f", "gem31p"})
GEMINI_ONLY_SKIP_STATEMENTS = frozenset(
    {
        "sexual_content_involving_minors",
    }
)
SCORE_MIN = 1
SCORE_MAX = 10


# --------------------------------------------------------------------------- #
# Download: populate DATA_ROOT/inputs/
# --------------------------------------------------------------------------- #


def cmd_download() -> None:
    """Populate ~/judge_correlations/inputs/ for all judges we can find."""
    inputs_root = DATA_ROOT / "inputs"
    for sub in JUDGES:
        (inputs_root / sub).mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        _populate_gpt41(target, inputs_root)
        _populate_gpt51(target, inputs_root)
        _populate_goss(target, inputs_root)
        _populate_gem3f(target, inputs_root)
        _populate_gem31p(target, inputs_root)
        _populate_kimi25(target, inputs_root)
        _populate_together_model("qwen235", QWEN235_BATCH_ROOT, target, inputs_root)
        _populate_together_model("glm51", GLM51_BATCH_ROOT, target, inputs_root)
        _populate_together_model("glm5", GLM5_BATCH_ROOT, target, inputs_root)
        _populate_together_model("qwen397", QWEN397_BATCH_ROOT, target, inputs_root)
        _populate_together_model("mm25", MM25_BATCH_ROOT, target, inputs_root)
        _populate_together_model("mm27", MM27_BATCH_ROOT, target, inputs_root)


def _populate_gpt41(target: str, inputs_root: Path) -> None:
    src = GPT51_BATCH_ROOT / target / "input_gpt41.jsonl"
    dst = inputs_root / "gpt41" / target / "judged_results.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        logger.warning(
            "gpt41/%s: source missing at %s (run `judge_gpt51_batch.py download` first)",
            target,
            src,
        )
        return
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        logger.info("gpt41/%s: already copied (%d bytes)", target, dst.stat().st_size)
        return
    shutil.copyfile(src, dst)
    logger.info("gpt41/%s: copied %d bytes from %s", target, dst.stat().st_size, src)


def _populate_gpt51(target: str, inputs_root: Path) -> None:
    # The gpt51_batch.py collect step writes judged_results.jsonl directly in
    # the per-target dir, alongside input_gpt41.jsonl.
    src = GPT51_BATCH_ROOT / target / "judged_results.jsonl"
    dst = inputs_root / "gpt51" / target / "judged_results.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        logger.info("gpt51/%s: not yet available (batch collect pending)", target)
        return
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        logger.info("gpt51/%s: already copied (%d bytes)", target, dst.stat().st_size)
        return
    shutil.copyfile(src, dst)
    logger.info("gpt51/%s: copied %d bytes from %s", target, dst.stat().st_size, src)


def _populate_goss(target: str, inputs_root: Path) -> None:
    gcs_dir = f"{GOSS_GCS_PREFIX}/{target}"
    local_dir = inputs_root / "goss" / target
    local_dir.mkdir(parents=True, exist_ok=True)
    fs, fs_prefix = url_to_fs(gcs_dir)
    try:
        shards = sorted(fs.glob(f"{fs_prefix}/shard_*.jsonl.gz"))
    except FileNotFoundError:
        logger.warning("goss/%s: no shards under %s", target, gcs_dir)
        return
    if not shards:
        logger.warning("goss/%s: empty shard list under %s", target, gcs_dir)
        return
    logger.info("goss/%s: %d shards to check", target, len(shards))
    n_downloaded = 0
    n_skipped = 0
    total_bytes = 0
    for shard_remote in shards:
        shard_name = shard_remote.split("/")[-1]
        local_path = local_dir / shard_name
        if local_path.exists() and local_path.stat().st_size > 0:
            n_skipped += 1
            total_bytes += local_path.stat().st_size
            continue
        with fs.open(shard_remote, "rb") as src, local_path.open("wb") as dst:
            data = src.read()
            dst.write(data)
        n_downloaded += 1
        total_bytes += local_path.stat().st_size
    logger.info(
        "goss/%s: downloaded %d new, skipped %d cached, %.1f MB total",
        target,
        n_downloaded,
        n_skipped,
        total_bytes / (1024 * 1024),
    )


def _populate_gem3f(target: str, inputs_root: Path) -> None:
    src = GEM3F_BATCH_ROOT / target / "judged_results.jsonl"
    dst = inputs_root / "gem3f" / target / "judged_results.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        logger.info("gem3f/%s: not yet available", target)
        return
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        logger.info("gem3f/%s: already copied (%d bytes)", target, dst.stat().st_size)
        return
    shutil.copyfile(src, dst)
    logger.info("gem3f/%s: copied %d bytes from %s", target, dst.stat().st_size, src)


def _populate_gem31p(target: str, inputs_root: Path) -> None:
    src = GEM31P_BATCH_ROOT / target / "judged_results.jsonl"
    dst = inputs_root / "gem31p" / target / "judged_results.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        logger.info("gem31p/%s: not yet available", target)
        return
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        logger.info("gem31p/%s: already copied (%d bytes)", target, dst.stat().st_size)
        return
    shutil.copyfile(src, dst)
    logger.info("gem31p/%s: copied %d bytes from %s", target, dst.stat().st_size, src)


def _populate_kimi25(target: str, inputs_root: Path) -> None:
    src = KIMI25_BATCH_ROOT / target / "judged_results.jsonl"
    dst = inputs_root / "kimi25" / target / "judged_results.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        logger.info("kimi25/%s: not yet available", target)
        return
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        logger.info("kimi25/%s: already copied (%d bytes)", target, dst.stat().st_size)
        return
    shutil.copyfile(src, dst)
    logger.info("kimi25/%s: copied %d bytes from %s", target, dst.stat().st_size, src)


def _populate_together_model(label: str, batch_root: Path, target: str, inputs_root: Path) -> None:
    """Generic populator for any Together-hosted judge (EXP-030)."""
    legacy_root = TOGETHER_LEGACY_ROOT / batch_root.name
    src = batch_root / target / "judged_results.jsonl"
    if not src.exists():
        src = legacy_root / target / "judged_results.jsonl"
    dst = inputs_root / label / target / "judged_results.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        logger.info("%s/%s: not yet available", label, target)
        return
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        logger.info("%s/%s: already copied (%d bytes)", label, target, dst.stat().st_size)
        return
    shutil.copyfile(src, dst)
    logger.info("%s/%s: copied %d bytes from %s", label, target, dst.stat().st_size, src)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_gpt41_records(target: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / "inputs" / "gpt41" / target / "judged_results.jsonl"
    return _load_jsonl(path) if path.exists() else []


def load_gpt51_records(target: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / "inputs" / "gpt51" / target / "judged_results.jsonl"
    return _load_jsonl(path) if path.exists() else []


def load_goss_records(target: str) -> list[dict[str, Any]]:
    """Load GPT-oss sharded JSONL.GZ output for one target."""
    shard_dir = DATA_ROOT / "inputs" / "goss" / target
    if not shard_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for shard in sorted(shard_dir.glob("shard_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def load_gem3f_records(target: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / "inputs" / "gem3f" / target / "judged_results.jsonl"
    return _load_jsonl(path) if path.exists() else []


def load_gem31p_records(target: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / "inputs" / "gem31p" / target / "judged_results.jsonl"
    return _load_jsonl(path) if path.exists() else []


def load_kimi25_records(target: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / "inputs" / "kimi25" / target / "judged_results.jsonl"
    return _load_jsonl(path) if path.exists() else []


def _load_by_label(label: str, target: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / "inputs" / label / target / "judged_results.jsonl"
    return _load_jsonl(path) if path.exists() else []


def load_qwen235_records(target: str) -> list[dict[str, Any]]:
    return _load_by_label("qwen235", target)


def load_glm51_records(target: str) -> list[dict[str, Any]]:
    return _load_by_label("glm51", target)


def load_glm5_records(target: str) -> list[dict[str, Any]]:
    return _load_by_label("glm5", target)


def load_qwen397_records(target: str) -> list[dict[str, Any]]:
    return _load_by_label("qwen397", target)


def load_mm25_records(target: str) -> list[dict[str, Any]]:
    return _load_by_label("mm25", target)


def load_mm27_records(target: str) -> list[dict[str, Any]]:
    return _load_by_label("mm27", target)


# --------------------------------------------------------------------------- #
# Score extraction (EXP-027 parse-failure semantics + legacy artifact filtering)
# --------------------------------------------------------------------------- #


def extract_score(record: dict[str, Any], judge: str) -> int | None:
    """Return the judge's integer score, or None on parse failure.

    Legacy parse-failure markers (pre-EXP-027, still present in existing
    GCS artifacts):
      - gpt41 CLI: score=5 with "Failed to parse" in the explanation
      - goss executor: score=0 with "Parse failure" in the explanation
    Post-EXP-027 (new runs): all judges emit score=None on parse failure.

    This function uniformly maps all parse failures to None so downstream
    aggregation can skip them instead of coercing to a biasing default.
    """
    j = record.get("judgment", {}) or {}
    score = j.get("score")
    if score is None:
        return None
    explanation = j.get("explanation", "") or ""
    if judge == "gpt41" and score == 5 and "Failed to parse" in explanation:
        return None
    if judge == "goss" and score == 0 and "Parse failure" in explanation:
        return None
    try:
        return int(score)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Join key
# --------------------------------------------------------------------------- #


def join_key(record: dict[str, Any], target: str) -> tuple[str, str, str, str]:
    """(target, prompt_id, response_text, behavior_id).

    EXP-026 widened the join from 2-tuple to 3-tuple to guard against the rare
    case where the same response text appears under different statements.
    Target is added on top because judges were run per-target and pooling the
    targets at the row level means key collisions are a real concern.
    """
    return (
        target,
        record.get("prompt_id", ""),
        record.get("response_text", "") or "",
        record.get("behavior_id", "") or "",
    )


# --------------------------------------------------------------------------- #
# Statistics (no scipy dependency — simple rank-based Spearman + quadratic kappa)
# --------------------------------------------------------------------------- #


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation without scipy."""
    if len(xs) < 3:
        return None
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    return _pearson(rx, ry)


def _average_ranks(vs: list[float]) -> list[float]:
    """Ranks with ties averaged — the canonical Spearman preprocessing."""
    n = len(vs)
    indexed = sorted(range(n), key=lambda i: vs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vs[indexed[j + 1]] == vs[indexed[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-indexed rank, averaged over the tie group
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    dxs = [x - mx for x in xs]
    dys = [y - my for y in ys]
    num = sum(dx * dy for dx, dy in zip(dxs, dys, strict=True))
    sx = (sum(dx * dx for dx in dxs)) ** 0.5
    sy = (sum(dy * dy for dy in dys)) ** 0.5
    if sx == 0 or sy == 0:
        return None  # degenerate; one judge gave the same score to everything
    return num / (sx * sy)


def weighted_kappa_quadratic(
    xs: list[int],
    ys: list[int],
    min_score: int = SCORE_MIN,
    max_score: int = SCORE_MAX,
) -> float | None:
    """Quadratic-weighted Cohen's kappa for ordinal ratings.

    Disagreements are weighted by squared distance, normalized so the worst
    possible disagreement in range has weight 1. Reported alongside Spearman
    because the two measure different things: Spearman says "do judges agree
    on ordering?" and kappa says "do judges give the same scores?" (adjusted
    for chance).
    """
    n = len(xs)
    if n < 3:
        return None
    n_scores = max_score - min_score + 1
    idx = lambda v: v - min_score  # noqa: E731

    observed = [[0] * n_scores for _ in range(n_scores)]
    for xi, yi in zip(xs, ys, strict=True):
        if xi < min_score or xi > max_score or yi < min_score or yi > max_score:
            continue
        observed[idx(xi)][idx(yi)] += 1

    total = sum(sum(row) for row in observed)
    if total == 0:
        return None

    weights = [[((i - j) ** 2) / ((n_scores - 1) ** 2) for j in range(n_scores)] for i in range(n_scores)]
    row_sums = [sum(row) for row in observed]
    col_sums = [sum(observed[i][j] for i in range(n_scores)) for j in range(n_scores)]
    expected = [[row_sums[i] * col_sums[j] / total for j in range(n_scores)] for i in range(n_scores)]

    num = sum(weights[i][j] * observed[i][j] for i in range(n_scores) for j in range(n_scores))
    den = sum(weights[i][j] * expected[i][j] for i in range(n_scores) for j in range(n_scores))
    if den == 0:
        return None
    return 1 - num / den


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def cmd_analyze() -> None:
    # Load every judge's records across every target, index by join key.
    by_judge: dict[str, dict[tuple[str, str, str, str], dict[str, Any]]] = {j: {} for j in JUDGES}
    raw_counts: dict[str, int] = {j: 0 for j in JUDGES}
    score_hist: dict[str, dict[int, int]] = {j: collections.Counter() for j in JUDGES}

    for target in TARGETS:
        loaders = {
            "gpt41": load_gpt41_records,
            "gpt51": load_gpt51_records,
            "goss": load_goss_records,
            "gem3f": load_gem3f_records,
            "gem31p": load_gem31p_records,
            "kimi25": load_kimi25_records,
            "qwen235": load_qwen235_records,
            "glm51": load_glm51_records,
            "glm5": load_glm5_records,
            "qwen397": load_qwen397_records,
            "mm25": load_mm25_records,
            "mm27": load_mm27_records,
        }
        for judge, loader in loaders.items():
            records = loader(target)
            for r in records:
                key = join_key(r, target)
                by_judge[judge][key] = r
                s = extract_score(r, judge)
                if s is not None:
                    score_hist[judge][s] += 1
            raw_counts[judge] += len(records)
            if records:
                logger.info("%s/%s: %d records loaded", judge, target, len(records))

    active_judges = [j for j in JUDGES if by_judge[j]]
    if len(active_judges) < 2:
        raise SystemExit(f"need at least 2 judges with records; found {active_judges}. " "Did you run `download` first?")
    logger.info("active judges: %s", active_judges)

    # Enumerate all pairwise comparisons.
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(active_judges):
        for b in active_judges[i + 1 :]:
            pairs.append((a, b))
    logger.info("pairs: %s", pairs)

    # For each pair: collect per-statement (list_a, list_b) of paired integer scores.
    pair_stmt_scores: dict[
        tuple[str, str],
        dict[str, tuple[list[int], list[int]]],
    ] = {pair: collections.defaultdict(lambda: ([], [])) for pair in pairs}
    pair_skip_stats: dict[tuple[str, str], dict[str, int]] = {pair: collections.defaultdict(int) for pair in pairs}

    for pair in pairs:
        ja, jb = pair
        pair_involves_gemini = bool(GEMINI_JUDGES & {ja, jb})
        shared = set(by_judge[ja].keys()) & set(by_judge[jb].keys())
        logger.info("pair %s<->%s: %d shared keys", ja, jb, len(shared))
        for key in shared:
            bid = key[3]
            # Universal exclusions apply to every pair so all correlation
            # numbers are on the same 43-statement basis.
            if bid in UNIVERSAL_SKIP_STATEMENTS:
                pair_skip_stats[pair]["universal_excluded_statement"] += 1
                continue
            # Safety-filter exclusion (sexual_content_involving_minors)
            # applies only to pairs that involve a Gemini judge. Kept in
            # the gpt41<->gpt51 pair since those judges aren't
            # safety-filtered on this content.
            if pair_involves_gemini and bid in GEMINI_ONLY_SKIP_STATEMENTS:
                pair_skip_stats[pair]["gemini_excluded_statement"] += 1
                continue
            ra = by_judge[ja][key]
            rb = by_judge[jb][key]
            sa = extract_score(ra, ja)
            sb = extract_score(rb, jb)
            if sa is None and sb is None:
                pair_skip_stats[pair]["both_parse_fail"] += 1
                continue
            if sa is None:
                pair_skip_stats[pair][f"{ja}_parse_fail"] += 1
                continue
            if sb is None:
                pair_skip_stats[pair][f"{jb}_parse_fail"] += 1
                continue
            list_a, list_b = pair_stmt_scores[pair][bid]
            list_a.append(sa)
            list_b.append(sb)

    # Per-statement Spearman + quadratic-weighted kappa per pair.
    results: dict[str, Any] = {
        "judges_present": active_judges,
        "raw_record_counts": raw_counts,
        "score_histograms": {j: {int(k): int(v) for k, v in sorted(score_hist[j].items())} for j in active_judges},
        "pairs": {},
    }

    for pair in pairs:
        ja, jb = pair
        per_statement: dict[str, dict[str, Any]] = {}
        for bid, (xs, ys) in sorted(pair_stmt_scores[pair].items()):
            rho = spearman([float(x) for x in xs], [float(y) for y in ys])
            qk = weighted_kappa_quadratic(xs, ys)
            per_statement[bid] = {
                "n": len(xs),
                "spearman": rho,
                "weighted_kappa": qk,
                "mean_a": sum(xs) / len(xs) if xs else None,
                "mean_b": sum(ys) / len(ys) if ys else None,
            }

        spearmans = [v["spearman"] for v in per_statement.values() if v["spearman"] is not None]
        kappas = [v["weighted_kappa"] for v in per_statement.values() if v["weighted_kappa"] is not None]
        spearmans_sorted = sorted(spearmans)
        kappas_sorted = sorted(kappas)

        def _median(xs: list[float]) -> float | None:
            if not xs:
                return None
            mid = len(xs) // 2
            if len(xs) % 2:
                return xs[mid]
            return (xs[mid - 1] + xs[mid]) / 2

        def _fraction_ge(xs: list[float], threshold: float) -> float | None:
            if not xs:
                return None
            return sum(1 for v in xs if v >= threshold) / len(xs)

        summary = {
            "n_statements": len(per_statement),
            "n_paired_items": sum(v["n"] for v in per_statement.values()),
            "spearman": {
                "mean": sum(spearmans) / len(spearmans) if spearmans else None,
                "median": _median(spearmans_sorted),
                "min": spearmans_sorted[0] if spearmans_sorted else None,
                "max": spearmans_sorted[-1] if spearmans_sorted else None,
                "frac_ge_0.5": _fraction_ge(spearmans, 0.5),
                "frac_ge_0.7": _fraction_ge(spearmans, 0.7),
                "frac_ge_0.9": _fraction_ge(spearmans, 0.9),
            },
            "weighted_kappa": {
                "mean": sum(kappas) / len(kappas) if kappas else None,
                "median": _median(kappas_sorted),
                "min": kappas_sorted[0] if kappas_sorted else None,
                "max": kappas_sorted[-1] if kappas_sorted else None,
            },
            "skipped": dict(pair_skip_stats[pair]),
        }

        results["pairs"][f"{ja}_vs_{jb}"] = {
            "per_statement": per_statement,
            "summary": summary,
        }

    # Write JSON output.
    output_dir = DATA_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "spearman_per_statement.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("wrote %s", json_path)

    hist_path = output_dir / "score_histograms.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(results["score_histograms"], f, indent=2)
        f.write("\n")
    logger.info("wrote %s", hist_path)

    # Print human-readable summary.
    _print_summary(results)


def _print_summary(results: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(" Per-judge raw record counts (before any filtering)")
    print("=" * 78)
    for judge, count in results["raw_record_counts"].items():
        if count > 0:
            print(f"  {judge:<8}  {count:>8} records")
    print()

    print("=" * 78)
    print(" Score distribution histograms (post parse-failure filter)")
    print("=" * 78)
    for judge, hist in results["score_histograms"].items():
        if not hist:
            continue
        total = sum(hist.values())
        if total == 0:
            continue
        print(f"  {judge}  (n={total})")
        for score in range(SCORE_MIN, SCORE_MAX + 1):
            count = hist.get(score, 0)
            pct = count / total * 100 if total else 0
            bar = "#" * int(pct / 2)
            print(f"    {score:>2}: {count:>7}  {pct:5.1f}%  {bar}")
        print()

    for pair_name, pair_data in results["pairs"].items():
        summary = pair_data["summary"]
        per_statement = pair_data["per_statement"]
        s_sum = summary["spearman"]
        k_sum = summary["weighted_kappa"]

        print("=" * 78)
        print(f" Pair: {pair_name}")
        print("=" * 78)
        print(f"  n_statements = {summary['n_statements']}    n_paired_items = {summary['n_paired_items']}")
        if summary["skipped"]:
            print(f"  skipped: {summary['skipped']}")
        print()
        print("  Spearman rank correlation (primary metric)")
        if s_sum["mean"] is not None:
            print(
                f"    mean = {s_sum['mean']:.4f}    median = {s_sum['median']:.4f}"
                f"    min = {s_sum['min']:.4f}    max = {s_sum['max']:.4f}"
            )
            print(
                f"    frac ≥ 0.5: {s_sum['frac_ge_0.5'] * 100:.1f}%    "
                f"frac ≥ 0.7: {s_sum['frac_ge_0.7'] * 100:.1f}%    "
                f"frac ≥ 0.9: {s_sum['frac_ge_0.9'] * 100:.1f}%"
            )
        print()
        print("  Quadratic-weighted Cohen's kappa (secondary metric)")
        if k_sum["mean"] is not None:
            print(
                f"    mean = {k_sum['mean']:.4f}    median = {k_sum['median']:.4f}"
                f"    min = {k_sum['min']:.4f}    max = {k_sum['max']:.4f}"
            )
        print()

        # Top-5 / bottom-5 by Spearman.
        sorted_stmts = sorted(
            [(bid, st) for bid, st in per_statement.items() if st["spearman"] is not None],
            key=lambda kv: kv[1]["spearman"],
        )
        print("  Bottom 5 statements (Spearman)")
        for bid, st in sorted_stmts[:5]:
            k_str = f"kappa={st['weighted_kappa']:.3f}" if st["weighted_kappa"] is not None else "kappa=n/a"
            print(
                f"    {bid:<40}  rho={st['spearman']:.4f}  {k_str}  n={st['n']}  "
                f"mean_a={st['mean_a']:.2f}  mean_b={st['mean_b']:.2f}"
            )
        print()
        print("  Top 5 statements (Spearman)")
        for bid, st in sorted_stmts[-5:][::-1]:
            k_str = f"kappa={st['weighted_kappa']:.3f}" if st["weighted_kappa"] is not None else "kappa=n/a"
            print(
                f"    {bid:<40}  rho={st['spearman']:.4f}  {k_str}  n={st['n']}  "
                f"mean_a={st['mean_a']:.2f}  mean_b={st['mean_b']:.2f}"
            )
        print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Populate ~/judge_correlations/inputs/ from local + GCS")
    sub.add_parser("analyze", help="Compute per-statement Spearman + weighted-kappa, write outputs/")
    args = parser.parse_args()
    if args.command == "download":
        cmd_download()
    elif args.command == "analyze":
        cmd_analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
