# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static-HTML progress dashboard for the Grug-MoE swarm.

Eval-metric filter: drops aggregates (``/macro_*``, ``/micro_*``, group-level
``eval/<group>/{bpb,loss}``, top-level ``eval/{bpb,loss,macro_*}``) and
infrastructure timing (``eval/loading_time``, ``eval/total_time``), keeping
only per-domain ``eval/<group>/<domain>/{bpb,loss}``. Macros are linear
combos of the kept metrics — they don't add info but inflate the apparent
task count; timing isn't data-mix dependent so it would inject noise into
the DSP fit.

Produces a single ``dashboard_data.json`` that the static ``template.html``
renders client-side with Observable Plot + d3.
No plotly. No HTML in Python. Just:

    fetch -> JSON blob -> copy template -> serve.

``--serve`` runs an HTTP server and periodically re-invokes this script in
``--build`` mode to refresh the JSON in place.
"""

import argparse
import csv
import http.server
import json
import math
import os
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
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import norm, rankdata
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from tqdm import tqdm

from experiments.grug.moe.datakit_moe_mix import _MIXABLE_BUCKETS, _TAIL_BUCKETS, _TOKEN_COUNTS, TARGET_BUDGET


def _is_data_mix_metric(key: str) -> bool:
    """Keep only per-domain bpb; drop loss, aggregates, and timing."""
    if not key.endswith("/bpb"):
        return False
    if "/macro_" in key or "/micro_" in key:
        return False
    return key.count("/") >= 3


_ENTITY = "held"
_PROJECT = "marin_moe"
_GROUP = "swarm_fisher_dsp_tau20_lam0p25_uscentral2"
# A task enters the DSP fit and the rank-correlation matrix only if it is present
# for >= _FIT_MIN_FRAC of finished candidates (absolute floor _FIT_MIN_ABS for
# numerical stability). Partial tasks give unreliable dose-response / correlations.
_FIT_MIN_ABS = 30
_FIT_MIN_FRAC = 0.90
_MIXTURES_CSV = "gs://marin-us-central2/grug-moe/swarm_data/production_swarm_168p_uscentral2_d_optimal_mixtures.csv"
_OUTPUT_PREFIX = "gs://marin-us-central2/grug/"
_CANDIDATE_RE = re.compile(r"swarm_fisher_dsp_d512_(\d{6})-[a-f0-9]+")
_BUDGET = 840
_TARGET_STEPS = 47759
_RUNNING_THRESHOLD_MIN = 60

# Human-readable names for the 40 lexical clusters (see bucket_samples.json).
_CLUSTER_NAMES = {
    0: "json-schema-and-config",
    1: "software-bug-discussion",
    2: "quantitative-explainer",
    3: "language-and-naming",
    4: "sentiment-polarity",
    5: "non-english-web",
    6: "tabular-numeric-dumps",
    7: "performance-metrics",
    8: "places-demographics",
    9: "wiki-tables-headers",
    10: "ascii-banners-and-source",
    11: "food-and-drink",
    12: "weather-and-seasons",
    13: "age-and-aging",
    14: "time-and-dates",
    15: "finance-and-law",
    16: "wellbeing-and-self-help",
    17: "civics-and-history",
    18: "health-and-medicine",
    19: "household-care-howtos",
    20: "automotive-mechanical",
    21: "aviation-and-space",
    22: "animals-environment",
    23: "ecology-and-disasters",
    24: "culture-events-lifestyle",
    25: "community-programs",
    26: "consumer-tech-howto",
    27: "visual-arts",
    28: "geopolitics-and-news",
    29: "religion-and-theology",
    30: "narrative-and-essay",
    31: "music-and-performance",
    32: "gaming-and-gambling",
    33: "sports",
    34: "plants-and-wildlife",
    35: "legal-cases-policy",
    36: "industrial-manufacturing",
    37: "biomedical-research",
    38: "library-and-distance-problems",
    39: "pledge-and-patriotism",
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
    bucket_names = [c[len(p0) :] for c in fns if c.startswith(p0)]
    assert [c[len(p1) :] for c in fns if c.startswith(p1)] == bucket_names
    candidates: list[dict | None] = [None] * _BUDGET
    for r in rows:
        idx = int(r["experiment_index"])
        candidates[idx] = {
            "phase_0": [float(r[f"{p0}{b}"]) for b in bucket_names],
            "phase_1": [float(r[f"{p1}{b}"]) for b in bucket_names],
        }
    return {"buckets": bucket_names, "candidates": candidates}


# Display-name overrides for a few task aliases whose underlying lm-eval
# group name is verbose ("mmlu_sl_verb") or carries a "logprob_" prefix.
# Auto-discovered aliases not in this map use the alias verbatim as their
# short name (so adding e.g. ``arc_easy_0shot`` Just Works).
_LOGPROB_SHORT_OVERRIDES: dict[str, str] = {
    "mmlu_sl_verb_0shot": "mmlu_0shot",
    "mmlu_sl_verb_5shot": "mmlu_5shot",
    "logprob_gsm8k_5shot": "gsm8k_5shot",
    "logprob_humaneval_10shot": "humaneval_10shot",
}
# JSON key inside ``results`` is normally the dir alias verbatim, BUT lm-eval
# group tasks publish their rolled-up score under the group name without the
# few-shot suffix (e.g. dir ``mmlu_sl_verb_0shot`` → results key
# ``mmlu_sl_verb``). Override only the group-task aliases.
_LOGPROB_RESULT_KEY_OVERRIDES: dict[str, str] = {
    "mmlu_sl_verb_0shot": "mmlu_sl_verb",
    "mmlu_sl_verb_5shot": "mmlu_sl_verb",
}
_LOGPROB_PREFIX = "gs://marin-us-central2/evaluation/grug_logprob/"
_LOGPROB_DIR_RE = re.compile(r"swarm_fisher_dsp_d512_(\d{6})/([^/]+?)-[a-f0-9]+/results\.json$")


def _candidate_result_keys(task_alias: str) -> tuple[str, ...]:
    """Possible keys under ``results`` for this dir alias, in priority order.

    Different lm-eval tasks publish under different conventions:
      - private logprob tasks (``logprob_gsm8k_5shot``) use the full alias
      - bare-name tasks (``arc_easy_0shot``) use the lm-eval task name only
      - group tasks (``mmlu_sl_verb_0shot``) roll up under the group name
    We try them in order and take the first one that's a populated dict.
    """
    if task_alias in _LOGPROB_RESULT_KEY_OVERRIDES:
        return (_LOGPROB_RESULT_KEY_OVERRIDES[task_alias],)
    keys: list[str] = [task_alias]
    m = re.search(r"^(.*?)_\d+shot$", task_alias)
    if m and m.group(1) != task_alias:
        keys.append(m.group(1))
    return tuple(keys)


def _incremental_load(prefix: str, cache_path: Path, parse_blob, desc: str) -> list:
    """Load every ``results.json`` under ``prefix``, reusing a path+mtime cache so
    only new/changed files are re-fetched from GCS.

    ``parse_blob(path, blob)`` returns a JSON-serializable value (or ``None`` to
    skip). The cache maps ``gcs_path -> [mtime, value]``; an entry whose mtime is
    unchanged is reused without re-opening the file (so a steady-state refresh
    only reads the handful of new results), and entries for paths that no longer
    exist are dropped. Returns the list of non-``None`` values.
    """
    fs = fsspec.filesystem("gs")
    info = fs.find(prefix, detail=True)
    paths = {
        p: str(meta.get("updated") or meta.get("mtime") or "") for p, meta in info.items() if p.endswith("results.json")
    }
    cache: dict[str, list] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}
    stale = [p for p, mt in paths.items() if (cache.get(p) or [None])[0] != mt]

    def _open(p: str):
        try:
            with fs.open("gs://" + p, "rt") as f:
                blob = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return p, None
        return p, parse_blob(p, blob)

    if stale:
        with ThreadPoolExecutor(max_workers=32) as ex:
            for p, val in tqdm(ex.map(_open, stale), total=len(stale), desc=desc):
                cache[p] = [paths[p], val]
    cache = {p: cache[p] for p in paths if p in cache}
    cache_path.write_text(json.dumps(cache))
    return [v[1] for v in cache.values() if v[1] is not None]


def fetch_logprob_evals(cache_path: Path = Path("dashboard_logprob_cache.json")) -> dict[int, dict[str, float]]:
    """For each finished swarm candidate, pull bpb from each task's results.json.

    Auto-discovers task aliases by globbing every result.json under
    ``grug_logprob/`` and extracts the bare lm-eval task name from the dir alias.
    New aliases added to ``launch_swarm_evals.py`` flow to the dashboard
    automatically once their first result lands — no edit needed here. Uses the
    path+mtime cache (see ``_incremental_load``) so a refresh only re-reads the
    results that changed since the last build.
    """

    def _parse(path, blob):
        m = _LOGPROB_DIR_RE.search(path)
        if not m:
            return None
        idx, task_alias = int(m.group(1)), m.group(2)
        results = blob.get("results") or {}
        short = _LOGPROB_SHORT_OVERRIDES.get(task_alias, task_alias)
        # `eval/logprob/<short>/bpb` clears the dashboard filter
        # (`_is_data_mix_metric` needs >=3 path separators).
        for k in _candidate_result_keys(task_alias):
            entry = results.get(k)
            if isinstance(entry, dict) and isinstance(entry.get("bpb,none"), (int, float)):
                return [idx, f"eval/logprob/{short}/bpb", float(entry["bpb,none"])]
        # Fallback: lm-eval wrote the score under its own task/group name rather
        # than the dir alias. Pick the entry whose key is a prefix of every other
        # bpb-bearing key — i.e. the group rollup (dir `gpqa_0shot` -> key
        # `leaderboard_gpqa`, with `leaderboard_gpqa_{main,diamond,...}` leaves),
        # or the lone entry of a single task (dir `belebele_spanish` -> key
        # `belebele_spa_Latn`).
        bpb_keys = [k for k, v in results.items() if isinstance(v, dict) and isinstance(v.get("bpb,none"), (int, float))]
        roots = [k for k in bpb_keys if all(o == k or o.startswith(k + "_") for o in bpb_keys)]
        if len(roots) == 1:
            return [idx, f"eval/logprob/{short}/bpb", float(results[roots[0]]["bpb,none"])]
        return None

    by_idx: dict[int, dict[str, float]] = {}
    for idx, key, bpb in _incremental_load(_LOGPROB_PREFIX, cache_path, _parse, "  logprob evals"):
        by_idx.setdefault(idx, {})[key] = bpb
    return by_idx


_PPL_PREFIX = "gs://marin-us-central2/evaluation/grug_ppl/"
# paloma/uncheatable datasets ride inside the multilingual_raw bundle (a
# superset of base_raw). Emit them per-dataset as `eval/<ds>/bpb` — the same
# keys the W&B surface used to publish — so the DSP fit keeps per-domain
# granularity for them instead of folding them into the bundle aggregate.
_PPL_PERDATASET_PREFIXES = ("paloma/", "uncheatable_eval/")


def fetch_ppl_evals(cache_path: Path = Path("dashboard_ppl_cache.json")) -> dict[int, dict[str, float]]:
    """For each finished swarm candidate, pull per-bundle and per-dataset bpb from
    the PPL bundle results.json files.

    Most bundles collapse to a single byte-weighted ``eval/ppl/<bundle>/bpb`` to
    keep the DSP fit tractable. The paloma/uncheatable datasets (carried inside
    ``multilingual_raw``) are instead emitted per-dataset as ``eval/<ds>/bpb`` —
    the same keys the W&B surface used to publish — plus a per-prefix rollup for
    the grid. Uses the path+mtime cache so a refresh only re-reads changed files.
    """

    def _parse(path, blob):
        m = re.search(r"swarm_fisher_dsp_d512_(\d{6})/", path)
        if not m:
            return None
        bundle = blob.get("bundle")
        if not bundle:
            return None
        metrics: dict[str, float] = {}
        sum_nll = 0.0
        sum_bytes = 0
        pref_nll: dict[str, float] = {}
        pref_bytes: dict[str, int] = {}
        for ds_key, entry in (blob.get("results") or {}).items():
            if not isinstance(entry, dict) or "error" in entry:
                continue
            nll = entry.get("nll")
            n_bytes = entry.get("n_bytes")
            if not isinstance(nll, (int, float)) or not isinstance(n_bytes, int) or n_bytes <= 0:
                continue
            # nll is in nats over n_bytes; bpb = nll/bytes/ln(2).
            matched = next((p for p in _PPL_PERDATASET_PREFIXES if ds_key.startswith(p)), None)
            if matched:
                metrics[f"eval/{ds_key}/bpb"] = float(nll) / int(n_bytes) / math.log(2)
                pref_nll[matched] = pref_nll.get(matched, 0.0) + float(nll)
                pref_bytes[matched] = pref_bytes.get(matched, 0) + int(n_bytes)
            else:
                sum_nll += float(nll)
                sum_bytes += int(n_bytes)
        # Per-prefix rollup (e.g. `eval/paloma/bpb`): only 2 path segments, so
        # `_is_data_mix_metric` drops it from the DSP fit — kept for the grid's
        # per-candidate eval coloring (formerly the W&B macro).
        for p, b in pref_bytes.items():
            metrics[f"eval/{p.rstrip('/')}/bpb"] = pref_nll[p] / b / math.log(2)
        if sum_bytes > 0:
            metrics[f"eval/ppl/{bundle}/bpb"] = sum_nll / sum_bytes / math.log(2)
        if not metrics:
            return None
        return [int(m.group(1)), metrics]

    by_idx: dict[int, dict[str, float]] = {}
    for idx, metrics in _incremental_load(_PPL_PREFIX, cache_path, _parse, "  ppl evals"):
        by_idx.setdefault(idx, {}).update(metrics)
    return by_idx


def fetch_wandb_runs(api_key: str, cache_path: Path = Path("dashboard_wandb_cache.json")) -> dict[int, dict]:
    """Pull per-candidate run metadata (step/loss/url/state) from W&B.

    Cached by run id: a finished run's summary is stable, so on refresh only
    new/still-running runs have their (network-expensive, lazily-fetched)
    ``r.summary`` read again — the bulk of finished runs are served from cache.
    Also pulls paloma/uncheatable per-dataset bpb from each run's summary
    (full 840 coverage); logprob tasks and multilingual/fineweb PPL come from
    fetch_logprob_evals + fetch_ppl_evals.
    """
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}
    api = wandb.Api(api_key=api_key)
    runs = api.runs(f"{_ENTITY}/{_PROJECT}", filters={"group": _GROUP})
    fresh: dict[str, dict] = {}
    for r in tqdm(runs, desc="  wandb runs"):
        idx = _idx_from_name(r.name)
        if idx is None:
            continue
        cached = cache.get(r.id)
        if cached is not None and cached.get("state") == "finished":
            fresh[r.id] = cached
            continue
        summary = r.summary  # network-expensive lazy read
        loss = summary.get("train/loss")
        # paloma/uncheatable per-dataset bpb are logged during training, so W&B
        # has full 840-candidate coverage — far ahead of the PPL eval jobs, which
        # carry these same datasets but lag. Same `eval/<ds>/bpb` keys, so they
        # merge into one column (PPL overwrites the few it has with equal values).
        sd = dict(summary)
        evals = {
            k: float(v)
            for k, v in sd.items()
            if isinstance(v, (int, float))
            and k.endswith("/bpb")
            and (k.startswith("eval/paloma/") or k.startswith("eval/uncheatable_eval/"))
        }
        fresh[r.id] = {
            "idx": idx,
            "id": r.id,
            "name": r.name,
            "state": (r.state or "queued").lower(),
            "step": int(summary.get("_step", 0) or 0),
            "loss": float(loss) if loss else None,
            "runtime_s": float(summary.get("_runtime", 0) or 0),
            "url": r.url,
            "created_at": r.created_at or "",
            "evals": evals,
        }
    cache_path.write_text(json.dumps(fresh))
    # Keep the latest run per candidate idx (by created_at).
    by_idx: dict[int, dict] = {}
    for meta in fresh.values():
        idx = meta["idx"]
        existing = by_idx.get(idx)
        if existing is not None and existing["created_at"] >= meta["created_at"]:
            continue
        by_idx[idx] = meta
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
        results = list(tqdm(ex.map(probe, cand_dirs), total=len(cand_dirs), desc="  gcs state"))

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


def compute_correlations(runs: dict[int, dict], gcs: dict[int, dict], mixtures: dict) -> dict | None:
    """Per-bucket Pearson correlation between mixture weights and each eval metric.

    Sign-flips bpb/loss metrics so positive r = "up-weighting helps".
    """
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

    # Same >=90%-coverage filter as the DSP fit: a partially-run task gives noisy
    # correlations, so keep it out of the rank-correlation matrix too.
    _cov_thresh = max(_FIT_MIN_ABS, int(np.ceil(_FIT_MIN_FRAC * n)))
    _cand_keys = {k for _, _, e in rows for k in e.keys() if _is_data_mix_metric(k)}
    eval_keys = sorted(k for k in _cand_keys if sum(1 for _, _, e in rows if e.get(k) is not None) >= _cov_thresh)
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

    sum_R = pearson(W0 + W1)

    def _cluster_order(R: list[list[float]]) -> list[int]:
        """Seriate tasks by hierarchical clustering (optimal leaf ordering) of
        their per-bucket correlation profiles, so tasks that correlate with the
        same data subsets sit adjacent.

        Rows are z-scored before a Euclidean linkage, which makes the distance
        monotonic in 1 - corr(profile_a, profile_b) — i.e. tasks are grouped by
        how similarly they respond to the buckets, not by overall magnitude.
        """
        M = np.nan_to_num(np.asarray(R, dtype=np.float64), nan=0.0)
        if M.shape[0] < 3:
            return list(range(M.shape[0]))
        sd = M.std(axis=1, keepdims=True)
        Mz = np.where(sd > 1e-9, (M - M.mean(axis=1, keepdims=True)) / (sd + 1e-12), 0.0)
        d = pdist(Mz, metric="euclidean")
        if d.size == 0 or not np.all(np.isfinite(d)):
            return list(range(M.shape[0]))
        Z = optimal_leaf_ordering(linkage(d, method="average"), d)
        return [int(i) for i in leaves_list(Z)]

    return {
        "n_candidates": n,
        "buckets": ordered,
        "eval_keys": eval_keys,
        "phase_0": pearson(W0),
        "phase_1": pearson(W1),
        "sum": sum_R,
        "task_order": _cluster_order(sum_R),
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
    buckets = mixtures["buckets"]
    candidates = mixtures["candidates"]

    # Per-bucket effective epoch multiplier per phase: c_phase[d] = phase_budget / bucket_tokens.
    # The swarm runs an 80/20 step split (launch_swarm._PHASE1_START_STEP),
    # so phase_0 sees 0.8 * TARGET_BUDGET tokens, phase_1 sees 0.2 *. Without
    # this split, c0 == c1 collapses Pareto-improver gradients to a single
    # direction in (theta0, theta1) and the two phases come out identical.
    _PHASE0_FRAC, _PHASE1_FRAC = 0.8, 0.2
    c_arr = np.zeros(len(buckets), dtype=np.float64)
    for i, b in enumerate(buckets):
        n_tok = _TOKEN_COUNTS.get(b, 0)
        c_arr[i] = TARGET_BUDGET / max(n_tok, 1)
    c0 = (c_arr * _PHASE0_FRAC).tolist()
    c1 = (c_arr * _PHASE1_FRAC).tolist()

    # Proportional baseline = w proportional to 1/c (token-share), normalised per phase. With
    # c0 = alpha * c and c1 = beta * c the normalised weights are still identical per
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

    # Build raw Y (n x T) sign-flipped so higher = better; drop NaNs by keeping
    # rows where every metric is present (we have very few NaNs after backfill).
    Y_full = np.full((n, len(eval_keys)), np.nan, dtype=np.float64)
    for i, (_, _, e) in enumerate(rows):
        for j, k in enumerate(eval_keys):
            v = e.get(k)
            if v is not None:
                Y_full[i, j] = v
    Y_full = Y_full * sign[None, :]
    # Fit each task on every candidate that has that task — NOT the all-tasks
    # intersection. A sparse task (just-started belebele, a no-bpb task) no
    # longer shrinks the usable candidate set for well-covered tasks, and absent
    # candidates are dropped per-task at fit time rather than imputed to the mean
    # (Z=0), which would bias the regression.
    task_present = (~np.isnan(Y_full)).sum(axis=0)
    n_k = n

    # Per-task rank-INT to N(0,1) over the candidates present for that task.
    Z = np.full_like(Y_full, np.nan)
    for j in range(Y_full.shape[1]):
        col = Y_full[:, j]
        mask = ~np.isnan(col)
        if not mask.any():
            continue
        ranks = rankdata(col[mask], method="average")
        Z[mask, j] = norm.ppf((ranks - 0.5) / mask.sum())

    W0 = np.stack([np.asarray(cand["phase_0"]) for _, cand, _ in rows]).astype(np.float64)
    W1 = np.stack([np.asarray(cand["phase_1"]) for _, cand, _ in rows]).astype(np.float64)
    D = W0.shape[1]
    # Only fit tasks with >=90% coverage of the finished candidates: a partially
    # run task gives an unreliable dose-response and pollutes the Pareto/math-code
    # optimisation. Sparser tasks still appear in `task_coverage` for visibility.
    # Keep a small absolute floor for numerical stability when n is small.
    _fit_threshold = max(_FIT_MIN_ABS, int(np.ceil(_FIT_MIN_FRAC * n)))
    fit_cols = [j for j in range(len(eval_keys)) if int(task_present[j]) >= _fit_threshold]
    fit_keys = [eval_keys[j] for j in fit_cols]
    print(
        f"  DSP fit: {n} candidates available, D={D} buckets, "
        f"{len(fit_keys)}/{len(eval_keys)} tasks fit (>={_fit_threshold} present = {_FIT_MIN_FRAC:.0%} coverage)"
    )

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
        "b0": jnp.array(0.0),
        "log_rho": jnp.full(D2, _spinv(0.3)),
        "tau": jnp.full(D2, 2.0),
        "log_a": jnp.full(D2, _spinv(0.1)),
        "log_p": jnp.full(D2, _spinv(0.01)),
    }
    _flat_init, _unravel = ravel_pytree(_theta_init)

    @jit
    def _loss_vg(theta_flat, w_stack_b, y_b, mask_b):
        def loss(t):
            err = (_predict_dsp(_unravel(t), w_stack_b) - y_b) ** 2
            return jnp.sum(mask_b * err) / jnp.sum(mask_b)

        return value_and_grad(loss)(theta_flat)

    _W_stack_np = np.concatenate([W0, W1], axis=1)
    W_stack_jax = jnp.asarray(_W_stack_np)

    def _fit_one(y_b, mask_b):
        def f(t):
            v, g = _loss_vg(jnp.asarray(t), W_stack_jax, y_b, mask_b)
            return float(v), np.asarray(g, dtype=np.float64)

        res = minimize(
            f,
            np.asarray(_flat_init, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 200, "ftol": 1e-8},
        )
        return _unravel(jnp.asarray(res.x))

    # Keep the full row matrix and pass a per-task 0/1 weight mask instead of
    # subsetting rows: the JIT batch shape stays constant (one compile) while each
    # task is still fit only on the candidates that have it.
    dsp_params = []
    for _done, t_idx in enumerate(fit_cols, 1):
        col = Z[:, t_idx]
        m = ~np.isnan(col)
        y_b = jnp.asarray(np.where(m, col, 0.0))
        mask_b = jnp.asarray(m.astype(np.float64))
        th = _fit_one(y_b, mask_b)
        pred = np.asarray(_predict_dsp(th, W_stack_jax))
        sigma = float(np.sqrt(np.mean((pred[m] - col[m]) ** 2)))
        # Split the 2D-length params back into phase_0 (first D) + phase_1 (last D)
        # slices so the JS predictor can stay in (w0, w1) coordinates.
        dsp_params.append(
            {
                "b0": round(float(th["b0"]), 5),
                "log_rho_0": [round(float(x), 5) for x in th["log_rho"][:D]],
                "log_rho_1": [round(float(x), 5) for x in th["log_rho"][D:]],
                "tau_0": [round(float(x), 5) for x in th["tau"][:D]],
                "tau_1": [round(float(x), 5) for x in th["tau"][D:]],
                "log_a_0": [round(float(x), 5) for x in th["log_a"][:D]],
                "log_a_1": [round(float(x), 5) for x in th["log_a"][D:]],
                "log_p_0": [round(float(x), 5) for x in th["log_p"][:D]],
                "log_p_1": [round(float(x), 5) for x in th["log_p"][D:]],
                "sigma": round(sigma, 4),
                "n_fit": int(m.sum()),
            }
        )
        if _done % 10 == 0 or _done == len(fit_cols):
            print(f"    [{_done:>2d}/{len(fit_cols)}] {eval_keys[t_idx][:50]:50s} sigma={sigma:.3f} n={int(m.sum())}")

    # Pareto-improver vs proportional baseline. With per-phase domains the
    # DSP params arrays stack phase_0 then phase_1 along the bucket axis.
    _w0_prop_jax = jnp.asarray(w0_prop)
    _w1_prop_jax = jnp.asarray(w1_prop)
    _b0_arr = jnp.asarray([p["b0"] for p in dsp_params])

    def _stack_per_phase_param(key0: str, key1: str) -> jnp.ndarray:
        return jnp.asarray([list(p[key0]) + list(p[key1]) for p in dsp_params])

    _log_rho_arr = _stack_per_phase_param("log_rho_0", "log_rho_1")
    _tau_arr = _stack_per_phase_param("tau_0", "tau_1")
    _log_a_arr = _stack_per_phase_param("log_a_0", "log_a_1")
    _log_p_arr = _stack_per_phase_param("log_p_0", "log_p_1")

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
        v0 = b2 * v0 + (1 - b2) * g0**2
        v1 = b2 * v1 + (1 - b2) * g1**2
        bc = 1 - b1 ** (t + 1)
        bv = 1 - b2 ** (t + 1)
        theta0 = theta0 - lr * (m0 / bc) / (jnp.sqrt(v0 / bv) + eps)
        theta1 = theta1 - lr * (m1 / bc) / (jnp.sqrt(v1 / bv) + eps)
        return theta0, theta1, m0, m1, v0, v1

    _theta0 = jnp.log(_w0_prop_jax + 1e-12)
    _theta1 = jnp.log(_w1_prop_jax + 1e-12)
    _m0 = jnp.zeros_like(_theta0)
    _m1 = jnp.zeros_like(_theta1)
    _v0 = jnp.zeros_like(_theta0)
    _v1 = jnp.zeros_like(_theta1)
    for step in range(1000):
        _theta0, _theta1, _m0, _m1, _v0, _v1 = _adam_step(_theta0, _theta1, _m0, _m1, _v0, _v1, step)
    w_pareto0 = np.asarray(jax.nn.softmax(_theta0))
    w_pareto1 = np.asarray(jax.nn.softmax(_theta1))
    _z_pareto = np.asarray(_predict_all(jax.nn.softmax(_theta0), jax.nn.softmax(_theta1)))
    _z_base_np = np.asarray(_z_baseline)
    delta = _z_pareto - _z_base_np
    print(
        f"  Pareto: improved on {(delta > 0).sum()}/{len(delta)} tasks "
        f"(strict >+{_eps_margin}: {(delta > _eps_margin).sum()})"
    )

    # Math/code-prioritized mixture: the same one-sided Pareto floor (keep every
    # task >= baseline + eps) PLUS a reward for the mean math/code z. The floor
    # protects every metric where there's no tradeoff; KAPPA buys math/code gains
    # against that floor once tradeoffs appear. FLOOR_WEIGHT vs KAPPA sets how
    # hard the floor resists being spent for math/code.
    _MATHCODE_KEYS = (
        "gsm8k",
        "humaneval",
        "mbpp",
        "math",
        "arithmetic",
        "asdiv",
        "code",
        "github",
        "formal_hardware",
        "arxiv",
        "wikipedia_english",
        "programing_languages",
        "mmlu_5shot",
    )
    _mc_mask = jnp.asarray([1.0 if any(k in name.lower() for k in _MATHCODE_KEYS) else 0.0 for name in fit_keys])
    _mc_n = float(np.asarray(_mc_mask).sum()) or 1.0
    _theta0_prop = jnp.log(_w0_prop_jax + 1e-12)
    _theta1_prop = jnp.log(_w1_prop_jax + 1e-12)
    # FLOOR_WEIGHT: how hard to hold every task >= baseline. KAPPA: math/code pull.
    # REG: trust-region L2 toward proportional (in logit space) so the optimum stays
    # in the DSP's fitted range instead of running to an extreme-mixture corner.
    _FLOOR_WEIGHT, _KAPPA, _REG = 10.0, 1.0, 0.005

    @jit
    def _mathcode_loss(theta0, theta1):
        zs = _predict_all(jax.nn.softmax(theta0), jax.nn.softmax(theta1))
        floor = jnp.sum(jnp.maximum(0.0, _z_baseline + _eps_margin - zs) ** 2)
        mc_mean = jnp.sum(zs * _mc_mask) / _mc_n
        reg = jnp.sum((theta0 - _theta0_prop) ** 2) + jnp.sum((theta1 - _theta1_prop) ** 2)
        return _FLOOR_WEIGHT * floor - _KAPPA * mc_mean + _REG * reg

    @jit
    def _adam_step_mc(theta0, theta1, m0, m1, v0, v1, t):
        g0, g1 = jax.grad(_mathcode_loss, argnums=(0, 1))(theta0, theta1)
        lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
        m0 = b1 * m0 + (1 - b1) * g0
        m1 = b1 * m1 + (1 - b1) * g1
        v0 = b2 * v0 + (1 - b2) * g0**2
        v1 = b2 * v1 + (1 - b2) * g1**2
        bc = 1 - b1 ** (t + 1)
        bv = 1 - b2 ** (t + 1)
        theta0 = theta0 - lr * (m0 / bc) / (jnp.sqrt(v0 / bv) + eps)
        theta1 = theta1 - lr * (m1 / bc) / (jnp.sqrt(v1 / bv) + eps)
        return theta0, theta1, m0, m1, v0, v1

    _t0, _t1 = _theta0_prop, _theta1_prop
    _mm0 = jnp.zeros_like(_t0)
    _mm1 = jnp.zeros_like(_t1)
    _vv0 = jnp.zeros_like(_t0)
    _vv1 = jnp.zeros_like(_t1)
    for step in range(1000):
        _t0, _t1, _mm0, _mm1, _vv0, _vv1 = _adam_step_mc(_t0, _t1, _mm0, _mm1, _vv0, _vv1, step)
    w_mathcode0 = np.asarray(jax.nn.softmax(_t0))
    w_mathcode1 = np.asarray(jax.nn.softmax(_t1))
    _z_mc = np.asarray(_predict_all(jax.nn.softmax(_t0), jax.nn.softmax(_t1)))
    _mc_np = np.asarray(_mc_mask).astype(bool)
    _delta_mc = _z_mc - _z_base_np
    print(
        f"  Math/code mix: {(_delta_mc >= -1e-9).sum()}/{len(_delta_mc)} tasks held >= baseline, "
        f"min delta {_delta_mc.min():.3f}; math/code mean z "
        f"{_z_base_np[_mc_np].mean():.3f} -> {_z_mc[_mc_np].mean():.3f}"
    )

    return {
        "task_names": fit_keys,
        "bucket_names": buckets,
        "c0": [round(float(x), 6) for x in c0],
        "c1": [round(float(x), 6) for x in c1],
        "w0_prop": [round(x, 6) for x in w0_prop],
        "w1_prop": [round(x, 6) for x in w1_prop],
        "w0_pareto": [round(float(x), 6) for x in w_pareto0],
        "w1_pareto": [round(float(x), 6) for x in w_pareto1],
        "w0_mathcode": [round(float(x), 6) for x in w_mathcode0],
        "w1_mathcode": [round(float(x), 6) for x in w_mathcode1],
        "dsp_params": dsp_params,
        "n_candidates_fit": n_k,
        "n_finished_with_evals": n,
        "task_coverage": [
            {
                "task": k,
                "n": int(task_present[j]),
                "pct": round(float(task_present[j]) / n, 4),
                "fitted": bool(int(task_present[j]) >= _fit_threshold),
            }
            for j, k in enumerate(eval_keys)
        ],
    }


def fit_predictors(runs: dict[int, dict], gcs: dict[int, dict], mixtures: dict) -> dict | None:
    """Per-metric ridge regression on cluster-aggregated weights (phase_0+phase_1).

    Sign-flipped so the predicted target is "higher = better". JS evaluates
    ``y_hat = intercept + coef · w`` live as the user drags sliders.
    """
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
        for w0, w1, c in zip(cand["phase_0"], cand["phase_1"], bucket_clusters, strict=True):
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
            "sign": -1.0,
            "y_mean": float(ys.mean()),
            "y_std": float(ys.std()),
            "y_min": float(ys.min()),
            "y_max": float(ys.max()),
            "baseline_pred": baseline,
            "sigma": float(resid.std()),
        }

    cluster_tokens: dict[int, int] = defaultdict(int)
    for c, _q, t in _MIXABLE_BUCKETS + _TAIL_BUCKETS:
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
    # Eval surfaces: logprob tasks + PPL bundles (incl. per-dataset
    # paloma/uncheatable). Both survive `_is_data_mix_metric` and get fit
    # together by the DSP + pearson stages.
    logprob_evals = fetch_logprob_evals()
    print(f"  logprob evals: {len(logprob_evals)} candidates have at least one task")
    ppl_evals = fetch_ppl_evals()
    print(f"  ppl evals: {len(ppl_evals)} candidates have at least one bundle")
    for store in (logprob_evals, ppl_evals):
        for idx, ev in store.items():
            if idx in runs:
                runs[idx].setdefault("evals", {}).update(ev)
            else:
                runs[idx] = {
                    "id": "",
                    "name": "",
                    "state": "finished",
                    "step": 0,
                    "loss": None,
                    "runtime_s": 0.0,
                    "url": "",
                    "created_at": "",
                    "evals": ev,
                }
    gcs = fetch_gcs_state()
    tsne = compute_tsne(mixtures, tsne_cache)
    corr = compute_correlations(runs, gcs, mixtures)
    dsp = fit_dsp_predictors(runs, gcs, mixtures)

    cluster_tokens: dict[int, int] = defaultdict(int)
    for c, _q, t in _MIXABLE_BUCKETS + _TAIL_BUCKETS:
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
            weights = [a + b for a, b in zip(cand["phase_0"], cand["phase_1"], strict=True)]
            cluster_sum: dict[int, float] = defaultdict(float)
            for w, c in zip(weights, bucket_clusters, strict=True):
                if c is not None:
                    cluster_sum[c] += w
            total_w = sum(cluster_sum.values()) or 1.0
            amps = {
                c: (w / total_w) / cluster_share[c] if cluster_share.get(c, 0) > 0 else 0.0
                for c, w in cluster_sum.items()
            }
            if amps:
                dominant_cluster, cluster_amp_top = max(amps.items(), key=lambda kv: kv[1])

        candidates_out.append(
            {
                "idx": idx,
                "state": g.get("state", "queued"),
                "step": g.get("step", 0),
                "minutes_since_ckpt": g.get("minutes_since_ckpt"),
                "loss": wr.get("loss"),
                "wandb_url": wr.get("url"),
                "wandb_state": wr.get("state"),
                "evals": wr.get("evals", {}),
                "phase_0": cand["phase_0"] if cand else None,
                "phase_1": cand["phase_1"] if cand else None,
                "tsne": tsne[idx] if idx < len(tsne) else None,
                "dominant_cluster": dominant_cluster,
                "dominant_amp": cluster_amp_top,
            }
        )

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
    build_cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *build_args]

    def trigger_build() -> None:
        try:
            r = subprocess.run(
                build_cmd,
                cwd=serve_dir,
                capture_output=True,
                text=True,
                timeout=600,
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
        except Exception as e:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] build EXC: {e}")

    def loop():
        # First build runs in the background so the server binds and serves the
        # existing dashboard_data.json immediately (instant render) instead of
        # blocking startup for a full GCS gather; it refreshes in place when done.
        trigger_build()
        while True:
            time.sleep(interval_s)
            trigger_build()

    threading.Thread(target=loop, daemon=True).start()

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(serve_dir), **kw)

        def do_GET(self):
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
    dashboard_dir = Path(__file__).parent
    ap.add_argument("--mixtures", default=str(dashboard_dir / "dashboard_mixtures.json"))
    ap.add_argument("--tsne", default=str(dashboard_dir / "dashboard_tsne.json"))
    ap.add_argument("--data", default=str(dashboard_dir / "dashboard_data.json"))
    ap.add_argument("--out", default=str(dashboard_dir / "dashboard.html"))
    ap.add_argument("--template", default=str(dashboard_dir / "template.html"))
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8375)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument(
        "--render-only",
        action="store_true",
        help="Copy --template to --out using the existing --data JSON; skip all "
        "GCS/W&B data gathering. For frontend-only (template) edits.",
    )
    args = ap.parse_args()

    if args.serve:
        build_args = [
            "--mixtures",
            args.mixtures,
            "--tsne",
            args.tsne,
            "--data",
            args.data,
            "--out",
            args.out,
            "--template",
            args.template,
        ]
        serve(Path(args.out), Path(args.data), port=args.port, interval_s=args.interval, build_args=build_args)
        return

    if args.render_only:
        shutil.copyfile(args.template, args.out)
        print(f"Rendered {args.out} from template (reused existing {args.data}; no data refresh).")
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
