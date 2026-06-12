# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data-only refresh for the Grug-MoE swarm dashboard v2.

Eval-metric filter: drops aggregates (``/macro_*``, ``/micro_*``, group-level
``eval/<group>/{bpb,loss}``, top-level ``eval/{bpb,loss,macro_*}``) and
infrastructure timing (``eval/loading_time``, ``eval/total_time``), keeping
only per-domain ``eval/<group>/<domain>/{bpb,loss}``. Macros are linear
combos of the kept metrics — they don't add info but inflate the apparent
task count; timing isn't data-mix dependent so it would inject noise into
the DSP fit.

Produces a single ``dashboard_data.json`` that the static
``dashboard_template.html`` renders client-side with Observable Plot + d3.
No plotly. No HTML in Python. Just:

    fetch -> JSON blob -> copy template -> serve.

The HTTP server and subprocess-build loop live in ``swarm_dashboard.py`` and
just call this script via its ``--build`` mode.
"""

import argparse
import csv
import http.server
import json
import os


def _is_data_mix_metric(key: str) -> bool:
    """Keep only per-domain bpb; drop loss (rank-identical to bpb per domain
    after rank-INT, so doubles up signal), macro/micro aggregates, and timing."""
    if not key.endswith("/bpb"):
        return False
    if "/macro_" in key or "/micro_" in key:
        return False
    # Require at least 3 path segments after the eval/ prefix so we drop
    # group-level rollups like eval/paloma/bpb and eval/uncheatable_eval/bpb.
    return key.count("/") >= 3


# Backward-compat shim: rebased main relocated `this_output_path` and `versioned`
# from `marin.execution.executor` to `marin.execution.types`, but our experiment
# tree still imports the old names. Re-export them on the `executor` module so
# transitive imports through `datakit_moe_mix` / `grug_moe_mix` keep working
# without touching every importer.
import marin.execution.executor as _marin_executor
from marin.execution.types import this_output_path as _this_output_path, versioned as _versioned
if not hasattr(_marin_executor, "this_output_path"):
    _marin_executor.this_output_path = _this_output_path
if not hasattr(_marin_executor, "versioned"):
    _marin_executor.versioned = _versioned
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import wandb

_ENTITY = "held"
_PROJECT = "marin_moe"
_GROUP = "swarm_fisher_dsp_tau20_lam0p25_uscentral2"
_MIXTURES_CSV = "gs://marin-us-central2/grug-moe/swarm_data/production_swarm_168p_uscentral2_d_optimal_mixtures.csv"
_OUTPUT_PREFIX = "gs://marin-us-central2/grug/"
_CANDIDATE_RE = re.compile(r"swarm_fisher_dsp_d512_(\d{6})-[a-f0-9]+")
_BUDGET = 840
_TARGET_STEPS = 47759
_RUNNING_THRESHOLD_MIN = 60

# Human-readable names for the 40 lexical clusters (see bucket_samples.json).
_CLUSTER_NAMES = {
    0: "json-schema-and-config", 1: "software-bug-discussion", 2: "quantitative-explainer",
    3: "language-and-naming", 4: "sentiment-polarity", 5: "non-english-web",
    6: "tabular-numeric-dumps", 7: "performance-metrics", 8: "places-demographics",
    9: "wiki-tables-headers", 10: "ascii-banners-and-source", 11: "food-and-drink",
    12: "weather-and-seasons", 13: "age-and-aging", 14: "time-and-dates",
    15: "finance-and-law", 16: "wellbeing-and-self-help", 17: "civics-and-history",
    18: "health-and-medicine", 19: "household-care-howtos", 20: "automotive-mechanical",
    21: "aviation-and-space", 22: "animals-environment", 23: "ecology-and-disasters",
    24: "culture-events-lifestyle", 25: "community-programs", 26: "consumer-tech-howto",
    27: "visual-arts", 28: "geopolitics-and-news", 29: "religion-and-theology",
    30: "narrative-and-essay", 31: "music-and-performance", 32: "gaming-and-gambling",
    33: "sports", 34: "plants-and-wildlife", 35: "legal-cases-policy",
    36: "industrial-manufacturing", 37: "biomedical-research",
    38: "library-and-distance-problems", 39: "pledge-and-patriotism",
}


def _idx_from_name(name: str) -> int | None:
    if not name or "_d512_" not in name:
        return None
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return None


def _cluster_of(bucket: str) -> int | None:
    if bucket.startswith("c") and len(bucket) >= 4:
        try:
            return int(bucket[1:3])
        except ValueError:
            return None
    return None


def _quality_of(bucket: str) -> int | None:
    if bucket.startswith("c") and len(bucket) >= 4 and bucket[3].isdigit():
        return int(bucket[3])
    return None


def fetch_mixtures() -> dict:
    with fsspec.open(_MIXTURES_CSV, "rt") as f:
        reader = csv.DictReader(f)
        fns = reader.fieldnames or []
        rows = list(reader)
    p0, p1 = "phase_0/", "phase_1/"
    bucket_names = [c[len(p0):] for c in fns if c.startswith(p0)]
    assert [c[len(p1):] for c in fns if c.startswith(p1)] == bucket_names
    candidates: list[dict | None] = [None] * _BUDGET
    for r in rows:
        idx = int(r["experiment_index"])
        candidates[idx] = {
            "phase_0": [float(r[f"{p0}{b}"]) for b in bucket_names],
            "phase_1": [float(r[f"{p1}{b}"]) for b in bucket_names],
        }
    return {"buckets": bucket_names, "candidates": candidates}


def _extract_evals(summary) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in summary.items():
        if not isinstance(k, str) or not k.startswith("eval/"):
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


# Each (task, json_key) lands as `eval/<task>/bpb` in the candidate's evals dict;
# the dashboard's per-bucket metric filter (`_is_data_mix_metric`) accepts these
# because they end in `/bpb`, have ≥3 path components, and don't trip the
# macro_/micro_ exclusions.
_LOGPROB_EVAL_TASKS: tuple[tuple[str, str], ...] = (
    ("mmlu_0shot",   "mmlu_sl_verb_0shot"),
    ("mmlu_5shot",   "mmlu_sl_verb_5shot"),
    ("gsm8k_5shot",  "logprob_gsm8k_5shot"),
    ("humaneval_10shot", "logprob_humaneval_10shot"),
)
_LOGPROB_AGG_KEY = "mmlu_sl_verb"  # MMLU group-level aggregate in results.json
_LOGPROB_PREFIX = "gs://marin-us-central2/evaluation/grug_logprob/"


def fetch_logprob_evals() -> dict[int, dict[str, float]]:
    """For each finished swarm candidate, pull bpb from each task's results.json.

    Returns ``{candidate_idx: {"eval/<short>/bpb": value, ...}}`` with one entry
    per task that has a non-empty results.json on GCS. If a candidate is missing
    any task it just won't have that key — downstream code already skips NaNs
    via per-metric masks. Reads run in parallel via a thread pool because
    4 tasks × ~700 candidates × sequential GCS ≈ 14 min single-threaded.
    """
    fs = fsspec.filesystem("gs")
    by_idx: dict[int, dict[str, float]] = {}

    paths: list[tuple[str, str, str]] = []  # (short, task_alias, gs_path)
    for short, task_alias in _LOGPROB_EVAL_TASKS:
        for p in fs.glob(f"{_LOGPROB_PREFIX}swarm_fisher_dsp_d512_*/{task_alias}*/results.json"):
            paths.append((short, task_alias, p))

    def _load(item):
        short, task_alias, path = item
        m = re.search(r"swarm_fisher_dsp_d512_(\d{6})/", path)
        if not m:
            return None
        try:
            with fs.open(f"gs://{path}", "rt") as f:
                blob = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        results = blob.get("results") or {}
        entry = results.get(_LOGPROB_AGG_KEY) if task_alias.startswith("mmlu") else results.get(task_alias)
        if not isinstance(entry, dict):
            return None
        bpb = entry.get("bpb,none")
        if not isinstance(bpb, (int, float)):
            return None
        return int(m.group(1)), short, float(bpb)

    with ThreadPoolExecutor(max_workers=32) as ex:
        for result in ex.map(_load, paths):
            if result is None:
                continue
            idx, short, bpb = result
            # Use `eval/logprob/<short>/bpb` so `_is_data_mix_metric` accepts it
            # (filter requires ≥3 path separators).
            by_idx.setdefault(idx, {})[f"eval/logprob/{short}/bpb"] = bpb
    return by_idx


_PPL_PREFIX = "gs://marin-us-central2/evaluation/grug_ppl/"
# Datasets from these prefixes already arrive via the W&B `_extract_evals` path
# (`eval/paloma/<ds>/bpb` and `eval/uncheatable_eval/<ds>/bpb`); dropping them
# from the PPL fetch avoids a near-duplicate metric per dataset that would
# show up twice in the DSP fit and Pearson heatmap.
_PPL_DEDUP_PREFIXES = ("paloma/", "uncheatable_eval/")


def fetch_ppl_evals() -> dict[int, dict[str, float]]:
    """For each finished swarm candidate, aggregate one mean-bpb metric per
    PPL bundle out of all that bundle's results.json files.

    Each bundle gets a single key ``eval/ppl/<bundle_key>/bpb`` containing the
    byte-weighted mean bpb across every non-error dataset in the bundle that
    isn't already covered by the W&B paloma/uncheatable surface (see
    ``_PPL_DEDUP_PREFIXES``). Per-dataset granularity is intentionally dropped
    to keep the DSP fit tractable — each bundle adds one metric instead of a
    few dozen.
    """
    fs = fsspec.filesystem("gs")
    paths = fs.glob(f"{_PPL_PREFIX}swarm_fisher_dsp_d512_*/*/results.json")

    def _load(path: str):
        m = re.search(r"swarm_fisher_dsp_d512_(\d{6})/", path)
        if not m:
            return None
        try:
            with fs.open(f"gs://{path}", "rt") as f:
                blob = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        bundle = blob.get("bundle")
        if not bundle:
            return None
        sum_nll = 0.0
        sum_bytes = 0
        for ds_key, entry in (blob.get("results") or {}).items():
            if not isinstance(entry, dict) or "error" in entry:
                continue
            if any(ds_key.startswith(p) for p in _PPL_DEDUP_PREFIXES):
                continue
            nll = entry.get("nll")
            n_bytes = entry.get("n_bytes")
            if not isinstance(nll, (int, float)) or not isinstance(n_bytes, int) or n_bytes <= 0:
                continue
            sum_nll += float(nll)
            sum_bytes += int(n_bytes)
        if sum_bytes == 0:
            return None
        # nll is in nats over n_bytes; bpb = nll/bytes/ln(2). The eval_perplexity
        # script already reports per-dataset bpb under the same formula, so this
        # is equivalent to a byte-weighted mean of the per-dataset values.
        import math
        bpb = sum_nll / sum_bytes / math.log(2)
        return int(m.group(1)), bundle, float(bpb)

    by_idx: dict[int, dict[str, float]] = {}
    with ThreadPoolExecutor(max_workers=32) as ex:
        for result in ex.map(_load, paths):
            if result is None:
                continue
            idx, bundle, bpb = result
            by_idx.setdefault(idx, {})[f"eval/ppl/{bundle}/bpb"] = bpb
    return by_idx


def fetch_wandb_runs(api_key: str) -> dict[int, dict]:
    api = wandb.Api(api_key=api_key)
    runs = api.runs(f"{_ENTITY}/{_PROJECT}", filters={"group": _GROUP})
    by_idx: dict[int, dict] = {}
    for r in runs:
        idx = _idx_from_name(r.name)
        if idx is None:
            continue
        created_at = r.created_at or ""
        existing = by_idx.get(idx)
        if existing is not None and existing["created_at"] >= created_at:
            continue
        summary = r.summary
        loss = summary.get("train/loss") or 0
        by_idx[idx] = {
            "id": r.id,
            "name": r.name,
            "state": (r.state or "queued").lower(),
            "step": int(summary.get("_step", 0) or 0),
            "loss": float(loss) if loss else 0.0,
            "runtime_s": float(summary.get("_runtime", 0) or 0),
            "url": r.url,
            "created_at": created_at,
            "evals": _extract_evals(summary),
        }
    return by_idx


def fetch_gcs_state() -> dict[int, dict]:
    fs = fsspec.filesystem("gs")
    cand_dirs = [d for d in fs.ls(_OUTPUT_PREFIX, detail=False) if "swarm_fisher_dsp_d512_" in d]

    def probe(dir_path: str) -> tuple[str, str | None, int, str | None]:
        terminal = None
        try:
            with fs.open(f"gs://{dir_path}/.executor_status", "rt") as f:
                content = f.read().strip()
            terminal = content if content in ("SUCCESS", "FAILED") else None
        except FileNotFoundError:
            pass
        max_step = 0
        last_updated: str | None = None
        try:
            for item in fs.ls(f"gs://{dir_path}/checkpoints/", detail=False):
                if "step-" in item:
                    try:
                        max_step = max(max_step, int(item.rsplit("step-", 1)[-1]))
                    except ValueError:
                        pass
            if max_step > 0:
                for it in fs.ls(f"gs://{dir_path}/checkpoints/step-{max_step}/", detail=True):
                    if it["size"] > 0:
                        last_updated = it.get("updated")
                        break
        except FileNotFoundError:
            pass
        return dir_path, terminal, max_step, last_updated

    with ThreadPoolExecutor(max_workers=64) as ex:
        results = list(ex.map(probe, cand_dirs))

    def rank(terminal: str | None, step: int) -> tuple[int, int, int]:
        return (1 if terminal == "SUCCESS" else 0, 1 if terminal is None else 0, step)

    by_idx: dict[int, dict] = {}
    for dir_path, terminal, max_step, last_updated in results:
        m = _CANDIDATE_RE.search(dir_path)
        if not m:
            continue
        idx = int(m.group(1))
        new_rank = rank(terminal, max_step)
        existing = by_idx.get(idx)
        if existing is not None and rank(existing["terminal"], existing["step"]) >= new_rank:
            continue
        by_idx[idx] = {
            "dir": dir_path,
            "terminal": terminal,
            "step": max_step,
            "last_checkpoint_updated": last_updated,
        }

    now = datetime.now(timezone.utc)
    for info in by_idx.values():
        if info["terminal"] == "SUCCESS":
            info["state"] = "finished"
            info["minutes_since_ckpt"] = None
        elif info["terminal"] == "FAILED" and info["step"] > 0:
            info["state"] = "failed"
            info["minutes_since_ckpt"] = None
        elif info["step"] == 0:
            info["state"] = "queued"
            info["minutes_since_ckpt"] = None
        else:
            info["state"] = "in_progress"
            updated_iso = info.get("last_checkpoint_updated")
            try:
                updated = datetime.fromisoformat(updated_iso.replace("Z", "+00:00"))
                info["minutes_since_ckpt"] = (now - updated).total_seconds() / 60
            except (AttributeError, ValueError):
                info["minutes_since_ckpt"] = None
    return by_idx


def compute_tsne(mixtures: dict, cache_path: Path) -> list[list[float]]:
    import numpy as np
    from sklearn.manifold import TSNE

    if cache_path.exists():
        return json.loads(cache_path.read_text())
    buckets = mixtures["buckets"]
    rows = []
    for cand in mixtures["candidates"]:
        if cand is None:
            rows.append([0.0] * (2 * len(buckets)))
        else:
            rows.append(cand["phase_0"] + cand["phase_1"])
    X = np.asarray(rows, dtype=np.float32)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
    coords = emb.tolist()
    cache_path.write_text(json.dumps(coords))
    return coords


def compute_correlations(
    runs: dict[int, dict], gcs: dict[int, dict], mixtures: dict
) -> dict | None:
    """Per-bucket Pearson correlation between mixture weights and each eval metric.

    Sign-flips bpb/loss metrics so positive r = "up-weighting helps".
    """
    import numpy as np

    buckets = mixtures["buckets"]
    candidates = mixtures["candidates"]

    rows = []
    for idx, g in gcs.items():
        if g.get("state") != "finished":
            continue
        wr = runs.get(idx)
        if not wr or not wr.get("evals"):
            continue
        cand = candidates[idx] if idx < len(candidates) else None
        if cand is None:
            continue
        rows.append((idx, cand, wr["evals"]))
    if len(rows) < 5:
        return None

    def sort_key(b: str) -> tuple[int, int]:
        c = _cluster_of(b)
        if c is None:
            return (999, 0)
        return (c, _quality_of(b) or 0)

    bucket_order = sorted(range(len(buckets)), key=lambda i: sort_key(buckets[i]))
    ordered = [buckets[i] for i in bucket_order]

    n = len(rows)
    W0 = np.zeros((n, len(buckets)))
    W1 = np.zeros((n, len(buckets)))
    for i, (_, cand, _) in enumerate(rows):
        W0[i, :] = np.asarray(cand["phase_0"])
        W1[i, :] = np.asarray(cand["phase_1"])
    W0 = W0[:, bucket_order]
    W1 = W1[:, bucket_order]

    eval_keys = sorted({k for _, _, e in rows for k in e.keys() if _is_data_mix_metric(k)})
    Y = np.full((n, len(eval_keys)), np.nan)
    for i, (_, _, e) in enumerate(rows):
        for j, k in enumerate(eval_keys):
            v = e.get(k)
            if v is not None:
                Y[i, j] = v
    # Every kept metric ends in /bpb or /loss (lower is better), so flip them all.
    sign = -np.ones(len(eval_keys), dtype=np.float64)
    Y = Y * sign[None, :]

    def pearson(W: np.ndarray) -> list[list[float]]:
        Wc = (W - W.mean(0, keepdims=True)) / (W.std(0, keepdims=True) + 1e-12)
        R = np.zeros((Y.shape[1], W.shape[1]))
        for j in range(Y.shape[1]):
            mask = ~np.isnan(Y[:, j])
            if mask.sum() < 5:
                continue
            yc = Y[mask, j]
            yc = (yc - yc.mean()) / (yc.std() + 1e-12)
            R[j, :] = yc @ Wc[mask, :] / mask.sum()
        return R.tolist()

    return {
        "n_candidates": n,
        "buckets": ordered,
        "eval_keys": eval_keys,
        "phase_0": pearson(W0),
        "phase_1": pearson(W1),
        "sum": pearson(W0 + W1),
    }


def fit_dsp_predictors(runs: dict[int, dict], gcs: dict[int, dict], mixtures: dict) -> dict | None:
    """Port of Calvin's DSP fit from the 300M-swarm gist, adapted to our 168 buckets.

    For each eval metric:
      * sign-flip bpb/loss so higher = better
      * rank-INT to N(0,1) z scores across the finished candidates
      * fit theta_per_metric = {b0, log_gamma, log_rho[D], tau[D], log_a[D], log_p[D]}
        with L-BFGS-B against MSE on z
    Then optimise a Pareto-improver mixture vs the token-proportional baseline.

    Per-bucket c_d = target_budget / bucket_tokens (the natural "epochs at unit
    weight"; gives proportional baseline = token-share).
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    from scipy.stats import norm, rankdata

    from experiments.grug.moe.datakit_moe_mix import (
        _MIXABLE_BUCKETS, _TAIL_BUCKETS, TARGET_BUDGET, _TOKEN_COUNTS,
    )

    buckets = mixtures["buckets"]
    candidates = mixtures["candidates"]

    # Per-bucket effective epoch multiplier per phase: c_phase[d] = phase_budget / bucket_tokens.
    # The swarm runs an 80/20 step split (swarm_fisher_dsp._PHASE1_START_STEP),
    # so phase_0 sees 0.8 × TARGET_BUDGET tokens, phase_1 sees 0.2 ×. Without
    # this split, c0 == c1 collapses Pareto-improver gradients to a single
    # direction in (theta0, theta1) and the two phases come out identical.
    _PHASE0_FRAC, _PHASE1_FRAC = 0.8, 0.2
    c_arr = np.zeros(len(buckets), dtype=np.float64)
    for i, b in enumerate(buckets):
        n_tok = _TOKEN_COUNTS.get(b, 0)
        c_arr[i] = TARGET_BUDGET / max(n_tok, 1)
    c0 = (c_arr * _PHASE0_FRAC).tolist()
    c1 = (c_arr * _PHASE1_FRAC).tolist()

    # Proportional baseline = w ∝ 1/c (token-share), normalised per phase. With
    # c0 = α · c and c1 = β · c the normalised weights are still identical per
    # phase; what differs at optimisation time is the per-phase dose magnitude
    # (gamma absorbs the inter-phase scale during the DSP fit).
    inv_c = 1.0 / np.maximum(c_arr, 1e-12)
    w0_prop = (inv_c / inv_c.sum()).tolist()
    w1_prop = w0_prop

    # Collect (mixture, evals) for finished candidates.
    rows = []
    for idx, g in gcs.items():
        if g.get("state") != "finished":
            continue
        wr = runs.get(idx)
        if not wr or not wr.get("evals"):
            continue
        cand = candidates[idx] if idx < len(candidates) else None
        if cand is None:
            continue
        rows.append((idx, cand, wr["evals"]))
    n = len(rows)
    if n < 30:
        return None

    eval_keys = sorted({k for _, _, e in rows for k in e.keys() if _is_data_mix_metric(k)})
    sign = -np.ones(len(eval_keys), dtype=np.float64)

    # Build raw Y (n × T) sign-flipped so higher = better; drop NaNs by keeping
    # rows where every metric is present (we have very few NaNs after backfill).
    Y_full = np.full((n, len(eval_keys)), np.nan, dtype=np.float64)
    for i, (_, _, e) in enumerate(rows):
        for j, k in enumerate(eval_keys):
            v = e.get(k)
            if v is not None:
                Y_full[i, j] = v
    Y_full = Y_full * sign[None, :]
    keep_row = ~np.isnan(Y_full).any(axis=1)
    if keep_row.sum() < 30:
        # Fall back: keep all rows but later we'll mask per-metric.
        keep_row[:] = True
    Y = Y_full[keep_row, :]
    rows_kept = [r for r, k in zip(rows, keep_row) if k]
    n_k = len(rows_kept)

    # Per-task rank-INT to N(0,1). norm.ppf((rank - 0.5) / n).
    Z = np.zeros_like(Y)
    for j in range(Y.shape[1]):
        col = Y[:, j]
        mask = ~np.isnan(col)
        ranks = rankdata(col[mask], method="average")
        z = norm.ppf((ranks - 0.5) / mask.sum())
        Z[mask, j] = z
        Z[~mask, j] = 0.0

    W0 = np.stack([np.asarray(cand["phase_0"]) for _, cand, _ in rows_kept]).astype(np.float64)
    W1 = np.stack([np.asarray(cand["phase_1"]) for _, cand, _ in rows_kept]).astype(np.float64)
    D = W0.shape[1]
    print(f"  DSP fit: n={n_k} candidates, D={D} buckets, T={len(eval_keys)} metrics")

    # Treat phase_0 and phase_1 buckets as 2*D independent domains in the
    # regression: each (bucket, phase) has its own rho/tau/a/p so the fit can
    # learn distinct curvature per phase. Stack costs and weights into length-2D
    # vectors with phase_0 buckets first, then phase_1 buckets. gamma drops out
    # — phase-asymmetric effects are now expressed per-domain instead of as a
    # task-scalar weight on phase_1.
    _c_stack_jax = jnp.asarray(np.concatenate([c0, c1]))

    def _spinv(x):
        return float(np.log(np.exp(x) - 1.0))

    def _predict_dsp(theta, w_stack_b):
        b0 = theta["b0"]
        rho = jax.nn.softplus(theta["log_rho"])
        tau = theta["tau"]
        a = jax.nn.softplus(theta["log_a"])
        p = jax.nn.softplus(theta["log_p"])
        z = w_stack_b * _c_stack_jax[None, :]
        signal = a[None, :] * (1.0 - jnp.exp(-rho[None, :] * z))
        u = jnp.log1p(z) - tau[None, :]
        penalty = p[None, :] * jax.nn.softplus(u) ** 2
        return -(b0 - signal.sum(axis=1) + penalty.sum(axis=1))

    D2 = 2 * D
    _theta_init = {
        "b0":        jnp.array(0.0),
        "log_rho":   jnp.full(D2, _spinv(0.3)),
        "tau":       jnp.full(D2, 2.0),
        "log_a":     jnp.full(D2, _spinv(0.1)),
        "log_p":     jnp.full(D2, _spinv(0.01)),
    }
    _flat_init, _unravel = ravel_pytree(_theta_init)

    @jit
    def _loss_vg(theta_flat, w_stack_b, y_b):
        def loss(t):
            return jnp.mean((_predict_dsp(_unravel(t), w_stack_b) - y_b) ** 2)
        return value_and_grad(loss)(theta_flat)

    W_stack_jax = jnp.asarray(np.concatenate([W0, W1], axis=1))

    def _fit_one(y):
        def f(t):
            v, g = _loss_vg(jnp.asarray(t), W_stack_jax, jnp.asarray(y))
            return float(v), np.asarray(g, dtype=np.float64)
        res = minimize(f, np.asarray(_flat_init, dtype=np.float64),
                       method="L-BFGS-B", jac=True,
                       options={"maxiter": 200, "ftol": 1e-8})
        return _unravel(jnp.asarray(res.x))

    dsp_params = []
    for t_idx in range(Z.shape[1]):
        th = _fit_one(Z[:, t_idx])
        pred = np.asarray(_predict_dsp(th, W_stack_jax))
        sigma = float(np.sqrt(np.mean((pred - Z[:, t_idx]) ** 2)))
        # Split the 2D-length params back into phase_0 (first D) + phase_1 (last D)
        # slices so the JS predictor can stay in (w0, w1) coordinates.
        dsp_params.append({
            "b0":         round(float(th["b0"]), 5),
            "log_rho_0": [round(float(x), 5) for x in th["log_rho"][:D]],
            "log_rho_1": [round(float(x), 5) for x in th["log_rho"][D:]],
            "tau_0":     [round(float(x), 5) for x in th["tau"][:D]],
            "tau_1":     [round(float(x), 5) for x in th["tau"][D:]],
            "log_a_0":   [round(float(x), 5) for x in th["log_a"][:D]],
            "log_a_1":   [round(float(x), 5) for x in th["log_a"][D:]],
            "log_p_0":   [round(float(x), 5) for x in th["log_p"][:D]],
            "log_p_1":   [round(float(x), 5) for x in th["log_p"][D:]],
            "sigma":      round(sigma, 4),
        })
        if (t_idx + 1) % 10 == 0 or t_idx == Z.shape[1] - 1:
            print(f"    [{t_idx+1:>2d}/{Z.shape[1]}] {eval_keys[t_idx][:50]:50s} sigma={sigma:.3f}")

    # Pareto-improver vs proportional baseline. With per-phase domains the
    # DSP params arrays stack phase_0 then phase_1 along the bucket axis.
    _w0_prop_jax = jnp.asarray(w0_prop)
    _w1_prop_jax = jnp.asarray(w1_prop)
    _b0_arr        = jnp.asarray([p["b0"]        for p in dsp_params])

    def _stack_per_phase_param(key0: str, key1: str) -> jnp.ndarray:
        return jnp.asarray([list(p[key0]) + list(p[key1]) for p in dsp_params])

    _log_rho_arr = _stack_per_phase_param("log_rho_0", "log_rho_1")
    _tau_arr     = _stack_per_phase_param("tau_0",     "tau_1")
    _log_a_arr   = _stack_per_phase_param("log_a_0",   "log_a_1")
    _log_p_arr   = _stack_per_phase_param("log_p_0",   "log_p_1")

    @jit
    def _predict_all(w0v, w1v):
        rho = jax.nn.softplus(_log_rho_arr)
        a_t = jax.nn.softplus(_log_a_arr)
        p_t = jax.nn.softplus(_log_p_arr)
        w_stack = jnp.concatenate([w0v, w1v])
        z = (w_stack * _c_stack_jax)[None, :]
        signal = a_t * (1.0 - jnp.exp(-rho * z))
        u = jnp.log1p(z) - _tau_arr
        penalty = p_t * jax.nn.softplus(u) ** 2
        return -(_b0_arr - signal.sum(axis=1) + penalty.sum(axis=1))

    _z_baseline = _predict_all(_w0_prop_jax, _w1_prop_jax)
    _eps_margin = 0.05

    @jit
    def _pareto_loss(theta0, theta1):
        w0v = jax.nn.softmax(theta0)
        w1v = jax.nn.softmax(theta1)
        zs = _predict_all(w0v, w1v)
        return jnp.sum(jnp.maximum(0.0, _z_baseline + _eps_margin - zs) ** 2)

    @jit
    def _adam_step(theta0, theta1, m0, m1, v0, v1, t):
        g0, g1 = jax.grad(_pareto_loss, argnums=(0, 1))(theta0, theta1)
        lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
        m0 = b1 * m0 + (1 - b1) * g0
        m1 = b1 * m1 + (1 - b1) * g1
        v0 = b2 * v0 + (1 - b2) * g0 ** 2
        v1 = b2 * v1 + (1 - b2) * g1 ** 2
        bc = 1 - b1 ** (t + 1)
        bv = 1 - b2 ** (t + 1)
        theta0 = theta0 - lr * (m0 / bc) / (jnp.sqrt(v0 / bv) + eps)
        theta1 = theta1 - lr * (m1 / bc) / (jnp.sqrt(v1 / bv) + eps)
        return theta0, theta1, m0, m1, v0, v1

    _theta0 = jnp.log(_w0_prop_jax + 1e-12)
    _theta1 = jnp.log(_w1_prop_jax + 1e-12)
    _m0 = jnp.zeros_like(_theta0); _m1 = jnp.zeros_like(_theta1)
    _v0 = jnp.zeros_like(_theta0); _v1 = jnp.zeros_like(_theta1)
    for step in range(1000):
        _theta0, _theta1, _m0, _m1, _v0, _v1 = _adam_step(_theta0, _theta1, _m0, _m1, _v0, _v1, step)
    w_pareto0 = np.asarray(jax.nn.softmax(_theta0))
    w_pareto1 = np.asarray(jax.nn.softmax(_theta1))
    _z_pareto = np.asarray(_predict_all(jax.nn.softmax(_theta0), jax.nn.softmax(_theta1)))
    _z_base_np = np.asarray(_z_baseline)
    delta = _z_pareto - _z_base_np
    print(f"  Pareto: improved on {(delta > 0).sum()}/{len(delta)} tasks "
          f"(strict >+{_eps_margin}: {(delta > _eps_margin).sum()})")

    return {
        "task_names": eval_keys,
        "bucket_names": buckets,
        "c0": [round(float(x), 6) for x in c0],
        "c1": [round(float(x), 6) for x in c1],
        "w0_prop": [round(x, 6) for x in w0_prop],
        "w1_prop": [round(x, 6) for x in w1_prop],
        "w0_pareto": [round(float(x), 6) for x in w_pareto0],
        "w1_pareto": [round(float(x), 6) for x in w_pareto1],
        "dsp_params": dsp_params,
        "n_candidates_fit": n_k,
    }


def fit_predictors(runs: dict[int, dict], gcs: dict[int, dict], mixtures: dict) -> dict | None:
    """Per-metric ridge regression on cluster-aggregated weights (phase_0+phase_1).

    Sign-flipped so the predicted target is "higher = better". JS evaluates
    ``y_hat = intercept + coef · w`` live as the user drags sliders.
    """
    import numpy as np
    from sklearn.linear_model import Ridge

    buckets = mixtures["buckets"]
    bucket_clusters = [_cluster_of(b) for b in buckets]

    cluster_ids = sorted({c for c in bucket_clusters if c is not None})
    n_clusters = len(cluster_ids)
    cluster_index = {c: i for i, c in enumerate(cluster_ids)}

    rows = []
    for idx, g in gcs.items():
        if g.get("state") != "finished":
            continue
        wr = runs.get(idx)
        if not wr or not wr.get("evals"):
            continue
        cand = mixtures["candidates"][idx] if idx < len(mixtures["candidates"]) else None
        if cand is None:
            continue
        w_vec = np.zeros(n_clusters)
        for w0, w1, c in zip(cand["phase_0"], cand["phase_1"], bucket_clusters):
            if c is None:
                continue
            w_vec[cluster_index[c]] += w0 + w1
        # Normalize to a distribution (so coefficients are interpretable as
        # "moving 1 unit of mixture weight from baseline").
        s = w_vec.sum()
        if s > 0:
            w_vec /= s
        rows.append((w_vec, wr["evals"]))
    if len(rows) < 10:
        return None

    X = np.stack([w for w, _ in rows])
    eval_keys = sorted({k for _, e in rows for k in e.keys() if _is_data_mix_metric(k)})

    metrics: dict[str, dict] = {}
    for k in eval_keys:
        y = np.array([e.get(k, np.nan) for _, e in rows])
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            continue
        ys = y[mask] * -1.0  # all kept metrics end in /bpb or /loss (lower = better)
        Xm = X[mask]
        m = Ridge(alpha=1.0).fit(Xm, ys)
        # Baseline = mean weights across candidates; predicted at baseline = intercept + coef · mean.
        baseline = float(m.predict(Xm.mean(axis=0, keepdims=True))[0])
        # Per-candidate residuals -> sigma for an uncertainty band.
        resid = ys - m.predict(Xm)
        metrics[k] = {
            "intercept": float(m.intercept_),
            "coef": m.coef_.tolist(),
            "sign": sign,
            "y_mean": float(ys.mean()),
            "y_std": float(ys.std()),
            "y_min": float(ys.min()),
            "y_max": float(ys.max()),
            "baseline_pred": baseline,
            "sigma": float(resid.std()),
        }

    # Proportional baseline mixture (token-share per cluster, normalized).
    from experiments.grug.moe.datakit_moe_mix import _MIXABLE_BUCKETS, _TAIL_BUCKETS
    cluster_tokens: dict[int, int] = defaultdict(int)
    for c, q, t in _MIXABLE_BUCKETS + _TAIL_BUCKETS:
        cluster_tokens[c] += t
    total = sum(cluster_tokens.values())
    baseline_weights = [cluster_tokens.get(c, 0) / total for c in cluster_ids]

    return {
        "cluster_ids": cluster_ids,
        "cluster_names": {c: _CLUSTER_NAMES.get(c, f"c{c:02d}") for c in cluster_ids},
        "baseline_weights": baseline_weights,
        "metrics": metrics,
    }


def build_data(api_key: str, mixtures_cache: Path, tsne_cache: Path) -> dict:
    if mixtures_cache.exists():
        mixtures = json.loads(mixtures_cache.read_text())
    else:
        mixtures = fetch_mixtures()
        mixtures_cache.write_text(json.dumps(mixtures))

    runs = fetch_wandb_runs(api_key)
    # Merge logprob bpb scores into each run's evals dict alongside the W&B
    # paloma/uncheatable keys. Both surfaces survive `_is_data_mix_metric` and
    # get fit together by the DSP + pearson stages.
    logprob_evals = fetch_logprob_evals()
    print(f"  logprob evals: {len(logprob_evals)} candidates have at least one task")
    ppl_evals = fetch_ppl_evals()
    print(f"  ppl evals: {len(ppl_evals)} candidates have at least one bundle")
    for store in (logprob_evals, ppl_evals):
        for idx, ev in store.items():
            if idx in runs:
                runs[idx].setdefault("evals", {}).update(ev)
            else:
                runs[idx] = {"id": "", "name": "", "state": "finished", "step": 0,
                             "loss": 0.0, "runtime_s": 0.0, "url": "", "created_at": "", "evals": ev}
    gcs = fetch_gcs_state()
    tsne = compute_tsne(mixtures, tsne_cache)
    corr = compute_correlations(runs, gcs, mixtures)
    dsp = fit_dsp_predictors(runs, gcs, mixtures)

    # Dominant cluster (by amplification ratio = weight / token-share) per candidate.
    from experiments.grug.moe.datakit_moe_mix import _MIXABLE_BUCKETS, _TAIL_BUCKETS
    cluster_tokens: dict[int, int] = defaultdict(int)
    for c, q, t in _MIXABLE_BUCKETS + _TAIL_BUCKETS:
        cluster_tokens[c] += t
    total_tokens = sum(cluster_tokens.values())
    cluster_share = {c: t / total_tokens for c, t in cluster_tokens.items()}

    buckets = mixtures["buckets"]
    bucket_clusters = [_cluster_of(b) for b in buckets]

    candidates_out = []
    for idx in range(_BUDGET):
        cand = mixtures["candidates"][idx] if idx < len(mixtures["candidates"]) else None
        g = gcs.get(idx, {})
        wr = runs.get(idx, {})

        dominant_cluster = None
        cluster_amp_top = None
        if cand is not None:
            weights = [a + b for a, b in zip(cand["phase_0"], cand["phase_1"])]
            cluster_sum: dict[int, float] = defaultdict(float)
            for w, c in zip(weights, bucket_clusters):
                if c is not None:
                    cluster_sum[c] += w
            total_w = sum(cluster_sum.values()) or 1.0
            amps = {
                c: (w / total_w) / cluster_share[c] if cluster_share.get(c, 0) > 0 else 0.0
                for c, w in cluster_sum.items()
            }
            if amps:
                dominant_cluster, cluster_amp_top = max(amps.items(), key=lambda kv: kv[1])

        candidates_out.append({
            "idx": idx,
            "state": g.get("state", "queued"),
            "step": g.get("step", 0),
            "minutes_since_ckpt": g.get("minutes_since_ckpt"),
            "loss": wr.get("loss", 0.0),
            "wandb_url": wr.get("url"),
            "wandb_state": wr.get("state"),
            "evals": wr.get("evals", {}),
            "phase_0": cand["phase_0"] if cand else None,
            "phase_1": cand["phase_1"] if cand else None,
            "tsne": tsne[idx] if idx < len(tsne) else None,
            "dominant_cluster": dominant_cluster,
            "dominant_amp": cluster_amp_top,
        })

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "budget": _BUDGET,
        "target_steps": _TARGET_STEPS,
        "running_threshold_min": _RUNNING_THRESHOLD_MIN,
        "candidates": candidates_out,
        "buckets": buckets,
        "cluster_names": _CLUSTER_NAMES,
        "cluster_token_share": cluster_share,
        "correlation": corr,
        "dsp": dsp,
    }


def serve(out_path: Path, data_path: Path, port: int, interval_s: int, build_args: list[str]) -> None:
    serve_dir = out_path.parent.resolve()
    out_name = out_path.name
    data_name = data_path.name
    build_cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *build_args]

    def trigger_build() -> None:
        try:
            r = subprocess.run(
                build_cmd, cwd=serve_dir, capture_output=True, text=True, timeout=600,
                env=os.environ.copy(),
            )
            stamp = datetime.now().isoformat(timespec="seconds")
            if r.returncode == 0:
                tail = (r.stdout.strip().splitlines() or [""])[-1]
                print(f"[{stamp}] build ok: {tail}")
            else:
                err = r.stderr.strip()[-500:] or r.stdout.strip()[-500:]
                print(f"[{stamp}] build FAILED (exit {r.returncode}): {err}")
        except subprocess.TimeoutExpired:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] build TIMEOUT")
        except Exception as e:  # noqa: BLE001
            print(f"[{datetime.now().isoformat(timespec='seconds')}] build EXC: {e}")

    trigger_build()

    def loop():
        while True:
            time.sleep(interval_s)
            trigger_build()

    threading.Thread(target=loop, daemon=True).start()

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(serve_dir), **kw)

        def do_GET(self):  # noqa: N802
            if self.path in ("/", ""):
                self.path = f"/{out_name}"
            return super().do_GET()

        def log_message(self, fmt, *args):
            return

    httpd = http.server.ThreadingHTTPServer(("", port), _Handler)
    print(f"serving http://localhost:{port}/  (build every {interval_s}s as subprocess)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixtures", default="dashboard_mixtures.json")
    ap.add_argument("--tsne", default="dashboard_tsne.json")
    ap.add_argument("--data", default="dashboard_data.json")
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--template", default=str(Path(__file__).parent / "dashboard_template.html"))
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8375)
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()

    if args.serve:
        build_args = [
            "--mixtures", args.mixtures,
            "--tsne", args.tsne,
            "--data", args.data,
            "--out", args.out,
            "--template", args.template,
        ]
        serve(Path(args.out), Path(args.data), port=args.port, interval_s=args.interval,
              build_args=build_args)
        return

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY env var required.")

    data = build_data(api_key, Path(args.mixtures), Path(args.tsne))
    Path(args.data).write_text(json.dumps(data, separators=(",", ":"), default=lambda v: None))
    shutil.copyfile(args.template, args.out)
    print(
        f"Wrote {args.out} + {args.data} "
        f"({sum(1 for c in data['candidates'] if c['state']=='finished')} finished, "
        f"{sum(1 for c in data['candidates'] if c['state']=='in_progress')} in_progress)"
    )


if __name__ == "__main__":
    main()
