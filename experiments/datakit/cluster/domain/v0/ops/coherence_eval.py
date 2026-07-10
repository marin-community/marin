# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster-coherence eval via Claude oracle.

Reads a ``cluster_stats_<K>.json`` produced by ``summarize.py`` and asks Claude
to judge each cluster: propose a short label, decide whether the top terms +
representative excerpts share a coherent topic, and flag outlier reps that
don't fit. Writes one JSON report alongside the input plus per-cluster
progress files so a kill/restart resumes where it left off.

Usage::

    uv run python -m experiments.datakit.cluster.domain.v0.ops.coherence_eval \\
        --input gs://marin-eu-west4/datakit/cluster/summarize_k40_<hash>/cluster_stats_40.json \\
        --output gs://marin-eu-west4/datakit/cluster/coherence/coherence_k40_<hash>.json

Anthropic API key is read from ``ANTHROPIC_API_KEY`` (env var). The repo's
``.marin.yaml`` carries the key for iris workers; if running locally, source
it manually -- never echo or commit it.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import anthropic
from pydantic import BaseModel, Field
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_WORKERS = 8
DEFAULT_MAX_TOKENS = 4096
_PROGRESS_DIRNAME = "_coherence_progress"


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class ClusterCoherence(BaseModel):
    """Claude's judgement of one cluster."""

    label: str = Field(description="Short (2-5 word) label naming what the cluster is about.")
    verdict: Literal["coherent", "borderline", "incoherent"] = Field(
        description=(
            "coherent: clear shared topic/style across terms and reps. "
            "borderline: mixed but a recognizable through-line. "
            "incoherent: no shared theme worth labeling."
        )
    )
    outlier_indices: list[int] = Field(
        default_factory=list,
        description="Indices (1-based) of representative excerpts that don't fit the proposed label. Empty if all fit.",
    )
    reasoning: str = Field(description="One sentence justification.")


SYSTEM_PROMPT = (
    "You evaluate document-cluster coherence for a Marin datakit pretraining pipeline. "
    "A cluster is described by (1) its top distinctive terms (c-TF-IDF unigrams/bigrams "
    "that separate this cluster from the rest of the corpus) and (2) 5 representative "
    "excerpts -- the documents closest to the cluster's embedding centroid. "
    "Decide whether the cluster has a coherent shared topic or style. Be honest: "
    "many real clusters are messy and 'borderline' or 'incoherent' is a useful signal. "
    "Use the proposed label as a sanity check -- if you can't write a short label that "
    "covers most reps, the verdict is probably 'incoherent'."
)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _format_cluster_prompt(cluster: dict) -> str:
    """Build the per-cluster user message body."""
    term_line = ", ".join(cluster.get("top_terms", [])) or "(no terms reported)"
    parts = [
        f"Cluster id: {cluster['cluster_id']}",
        f"Top terms: {term_line}",
        "",
        "Representative excerpts (closest to centroid):",
    ]
    for i, rep in enumerate(cluster.get("representatives", []), 1):
        src = rep.get("source", "?")
        excerpt = (rep.get("excerpt") or "").strip().replace("\n", " ")
        # Keep prompt short -- the excerpts are already truncated to ~500 chars upstream.
        parts.append(f"[{i}] source={src}: {excerpt}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# IO helpers (GCS + local both supported via fsspec / StoragePath)
# ---------------------------------------------------------------------------


def _read_json(uri: str) -> dict:
    return json.loads(StoragePath(uri).read_text())


def _write_json(uri: str, payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    StoragePath(uri).write_text(text)


def _progress_uri(output_uri: str, cluster_id: int) -> str:
    base = output_uri.rsplit("/", 1)[0]
    return f"{base}/{_PROGRESS_DIRNAME}/cluster_{cluster_id:06d}.json"


def _try_load_progress(output_uri: str, cluster_id: int) -> dict | None:
    try:
        return _read_json(_progress_uri(output_uri, cluster_id))
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("Ignoring unreadable progress for cluster %d: %r", cluster_id, e)
        return None


# ---------------------------------------------------------------------------
# Single-cluster call
# ---------------------------------------------------------------------------


def _judge_cluster(
    client: anthropic.Anthropic,
    model: str,
    cluster: dict,
    max_tokens: int,
) -> ClusterCoherence:
    """One Anthropic call: parse the structured response into ``ClusterCoherence``."""
    body = _format_cluster_prompt(cluster)
    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=[
            # Stable system prefix -- mark cacheable so repeated calls in the same
            # 5-minute window read from cache (cluster body is the volatile suffix).
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": body}],
        output_format=ClusterCoherence,
    )
    return response.parsed_output


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run(
    input_uri: str,
    output_uri: str,
    model: str,
    workers: int,
    max_tokens: int,
    limit: int | None,
) -> None:
    stats = _read_json(input_uri)
    clusters = stats.get("clusters", [])
    if limit is not None:
        clusters = clusters[:limit]
    if not clusters:
        raise RuntimeError(f"No clusters found in {input_uri}")

    k_view = stats.get("k_view")
    k_train = stats.get("k_train")
    logger.info(
        "Loaded %d clusters from %s (k_train=%s, k_view=%s); model=%s, workers=%d",
        len(clusters),
        input_uri,
        k_train,
        k_view,
        model,
        workers,
    )

    # Resume: any per-cluster progress file already saved is reused.
    cached: dict[int, dict] = {}
    pending: list[dict] = []
    for c in clusters:
        cid = int(c["cluster_id"])
        prior = _try_load_progress(output_uri, cid)
        if prior is not None:
            cached[cid] = prior
        else:
            pending.append(c)
    logger.info("Resuming with %d cached, %d to judge", len(cached), len(pending))

    client = anthropic.Anthropic()
    t0 = time.monotonic()
    completed = len(cached)
    total = len(clusters)

    def _one(c: dict) -> tuple[int, dict | None, Exception | None]:
        cid = int(c["cluster_id"])
        try:
            verdict = _judge_cluster(client, model, c, max_tokens)
        except Exception as e:
            return cid, None, e
        record = {
            "cluster_id": cid,
            "n_sampled": c.get("n_sampled"),
            "top_terms": c.get("top_terms", []),
            "representatives": c.get("representatives", []),
            **verdict.model_dump(),
        }
        # Persist per-cluster so a kill mid-run doesn't lose progress.
        _write_json(_progress_uri(output_uri, cid), record)
        return cid, record, None

    failures: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, c): c for c in pending}
        for fut in as_completed(futures):
            cid, record, err = fut.result()
            completed += 1
            if err is not None:
                failures[cid] = f"{type(err).__name__}: {err}"
                logger.warning("Cluster %d failed: %s", cid, failures[cid])
                continue
            cached[cid] = record  # type: ignore[assignment]
            if completed % max(1, total // 20) == 0:
                elapsed = time.monotonic() - t0
                logger.info("%d/%d clusters judged (%.1fs elapsed)", completed, total, elapsed)

    # Single-shot retry for transient failures.
    if failures:
        logger.info("Retrying %d failed clusters once", len(failures))
        retry_ids = list(failures.keys())
        retry_clusters = [c for c in pending if int(c["cluster_id"]) in retry_ids]
        for c in retry_clusters:
            cid, record, err = _one(c)
            if err is None:
                cached[cid] = record  # type: ignore[assignment]
                failures.pop(cid, None)

    # Aggregate.
    verdict_counts = Counter(r.get("verdict") for r in cached.values() if r is not None)
    coherent_ids = sorted(cid for cid, r in cached.items() if r and r.get("verdict") == "coherent")
    borderline_ids = sorted(cid for cid, r in cached.items() if r and r.get("verdict") == "borderline")
    incoherent_ids = sorted(cid for cid, r in cached.items() if r and r.get("verdict") == "incoherent")

    summary = {
        "input": input_uri,
        "model": model,
        "k_train": k_train,
        "k_view": k_view,
        "n_clusters_total": total,
        "n_clusters_judged": len(cached),
        "n_failed": len(failures),
        "verdict_counts": dict(verdict_counts),
        "coherent_fraction": verdict_counts.get("coherent", 0) / max(1, len(cached)),
        "coherent_ids": coherent_ids,
        "borderline_ids": borderline_ids,
        "incoherent_ids": incoherent_ids,
        "failures": failures,
        "clusters": [cached[cid] for cid in sorted(cached.keys())],
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    _write_json(output_uri, summary)
    logger.info(
        "Wrote %s: %d coherent / %d borderline / %d incoherent / %d failed",
        output_uri,
        verdict_counts.get("coherent", 0),
        verdict_counts.get("borderline", 0),
        verdict_counts.get("incoherent", 0),
        len(failures),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="cluster_stats_<K>.json URI (gs:// or local)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Coherence report URI to write (gs:// or local)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model id (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent API calls (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"max_tokens per call (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N clusters (smoke test)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set. Source it from .marin.yaml or export it manually.")
        sys.exit(2)

    _run(
        input_uri=args.input,
        output_uri=args.output,
        model=args.model,
        workers=args.workers,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
