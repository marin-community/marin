# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow"]
# ///
"""Swarm-run loading for the mixing-via-embeddings experiment (spec: `SwarmRun`, loaders).

Two sources, one row per (mixture run, scale):

- 60M / 1.2B: the vendored canonical run table `data/two_phase_many.csv.gz` (see `data/README.md`
  for provenance). This CSV is the label source of truth for the 60M scale.
- 300M / 6B: the public W&B project `marin-community/marin`, queried via raw GraphQL (no API
  key). The replay reuses the 60M run names and mixtures; weights read back from run configs are
  asserted consistent with the CSV.

Run as a script to write `runs.parquet` and `domains.parquet` under `scratch/mixture_features/`.

Ground constants below were verified against the swarm branch (tag `swarm-branch`, commit
bf26b666a) and live W&B run configs; see the inline notes on each.
"""

import argparse
import csv
import gzip
import json
import logging
import math
import urllib.request
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
RUN_TABLE_CSV = DATA_DIR / "two_phase_many.csv.gz"
DOMAIN_TOKEN_COUNTS_JSON = DATA_DIR / "domain_token_counts.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "scratch" / "mixture_features"

WANDB_GRAPHQL_URL = "https://api.wandb.ai/graphql"
WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "marin"
USER_AGENT = "marin-mixture-features/0.1"

EXPERIMENT_60M = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240"
EXPERIMENT_300M = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b"

SCALE_60M = "60m_1p2b"
SCALE_300M = "300m_6b"
MODEL_SIZE = {SCALE_60M: 60_000_000, SCALE_300M: 300_000_000}

# Both scales train with batch 128 x seq 2048 (verified from live W&B run configs and
# two_phase_dolma3_dolmino_top_level.py / launch_two_phase_many_qsplit240_300m_6b.py on the
# swarm branch: EXPERIMENT_BUDGET 1_200_000_000 and 6_000_000_000, PHASE_BOUNDARIES = [0.8]).
TOKENS_PER_STEP = 128 * 2048
# Exact per-scale (phase-1 start step, total steps), read from run configs' train_weights
# blocks; the 0.8 boundary is block-size aligned, hence not exactly 0.8 * total.
PHASE_STEPS = {SCALE_60M: (3648, 4577), SCALE_300M: (18304, 22888)}
PHASE_TOKENS = {
    scale: (boundary * TOKENS_PER_STEP, (total - boundary) * TOKENS_PER_STEP)
    for scale, (boundary, total) in PHASE_STEPS.items()
}

# weight_sampler.py `DirichletSamplingParams.min_dominant_weight`: vertex-biased samples give
# one domain >= 0.7 of a phase's mass. We identify vertex runs from the weights themselves.
VERTEX_MIN_DOMINANT_WEIGHT = 0.7

PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
METRIC_COLUMNS = (
    PRIMARY_METRIC,
    "eval/uncheatable_eval/macro_bpb",
    "eval/paloma/macro_bpb",
)

WEIGHT_CONSISTENCY_ATOL = 1e-6
MIN_CONSISTENCY_RUNS = 5

RUNS_QUERY = """
query Runs($entity: String!, $project: String!, $filters: JSONString, $after: String) {
  project(name: $project, entityName: $entity) {
    runs(filters: $filters, first: 50, after: $after) {
      edges { node { name displayName state createdAt config summaryMetrics } }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""


@dataclass(frozen=True)
class SwarmRun:
    """One proxy run of the qsplit240 swarm at one scale."""

    run_id: str  # mixture identity, shared across scales (CSV run_id)
    run_name: str  # e.g. "run_00127"; the replay reuses 60M run names
    scale: str  # SCALE_60M | SCALE_300M
    model_size: int  # nominal parameters
    wandb_run_id: str
    source_experiment: str
    phase_weights: tuple[Mapping[str, float], ...]  # per-phase mixture, normalized per phase
    phase_tokens: tuple[int, ...]  # tokens trained in each phase
    domain_tokens: tuple[Mapping[str, float], ...]  # weights x phase_tokens
    is_vertex: bool  # max phase-reduced weight >= VERTEX_MIN_DOMINANT_WEIGHT
    vertex_domain: str | None
    metrics: Mapping[str, float]


def swarm_domains() -> list[str]:
    """The 39 swarm domain names, sorted, from the vendored run table header."""
    with gzip.open(RUN_TABLE_CSV, "rt") as f:
        header = next(csv.reader(f))
    domains = sorted(c[len("phase_0_") :] for c in header if c.startswith("phase_0_"))
    if len(domains) != 39:
        raise ValueError(f"Expected 39 domains in {RUN_TABLE_CSV}, got {len(domains)}")
    return domains


def domain_token_counts() -> dict[str, int]:
    """Per-domain available token counts (TOP_LEVEL_DOMAIN_TOKEN_COUNTS from the swarm branch)."""
    payload = json.loads(DOMAIN_TOKEN_COUNTS_JSON.read_text())
    return {d: int(n) for d, n in payload["available_tokens"].items()}


def _normalize_phase(weights: Mapping[str, float], context: str) -> dict[str, float]:
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-3):
        raise ValueError(f"{context}: phase weights sum to {total}, expected ~1")
    return {d: w / total for d, w in weights.items()}


def _vertex_of(phase_weights: tuple[Mapping[str, float], ...]) -> tuple[bool, str | None]:
    """Vertex identification from the weights: max-over-phases dose >= 0.7 for some domain."""
    reduced = {d: max(pw[d] for pw in phase_weights) for d in phase_weights[0]}
    dominant = max(reduced, key=reduced.__getitem__)
    if reduced[dominant] >= VERTEX_MIN_DOMINANT_WEIGHT:
        return True, dominant
    return False, None


def _make_run(
    *,
    run_id: str,
    run_name: str,
    scale: str,
    wandb_run_id: str,
    source_experiment: str,
    phase_weights: tuple[dict[str, float], ...],
    metrics: dict[str, float],
) -> SwarmRun:
    is_vertex, vertex_domain = _vertex_of(phase_weights)
    tokens = PHASE_TOKENS[scale]
    domain_tokens = tuple({d: w * t for d, w in pw.items()} for pw, t in zip(phase_weights, tokens, strict=True))
    return SwarmRun(
        run_id=run_id,
        run_name=run_name,
        scale=scale,
        model_size=MODEL_SIZE[scale],
        wandb_run_id=wandb_run_id,
        source_experiment=source_experiment,
        phase_weights=phase_weights,
        phase_tokens=tokens,
        domain_tokens=domain_tokens,
        is_vertex=is_vertex,
        vertex_domain=vertex_domain,
        metrics=metrics,
    )


def load_qsplit240_60m(expected_missing: Collection[str] = ()) -> list[SwarmRun]:
    """Load the 60M/1.2B swarm from the vendored canonical run table.

    Raises listing offending run names if any run is incomplete (not `completed`, or missing the
    primary metric) unless it appears in `expected_missing`.
    """
    domains = swarm_domains()
    df = pd.read_csv(RUN_TABLE_CSV)

    bad = df[(df["status"] != "completed") | df[PRIMARY_METRIC].isna()]["run_name"].tolist()
    unexpected = [r for r in bad if r not in set(expected_missing)]
    if unexpected:
        raise ValueError(f"Incomplete 60M runs not in expected_missing: {unexpected}")
    df = df[~df["run_name"].isin(bad)]

    runs = []
    for row in df.itertuples(index=False):
        # zip with df.columns: itertuples mangles names containing "/".
        record = dict(zip(df.columns, row, strict=True))
        phase_weights = tuple(
            _normalize_phase(
                {d: float(record[f"phase_{p}_{d}"]) for d in domains},
                context=f"60m {record['run_name']} phase {p}",
            )
            for p in (0, 1)
        )
        metrics = {m: float(record[m]) for m in METRIC_COLUMNS if not pd.isna(record[m])}
        runs.append(
            _make_run(
                run_id=str(record["run_id"]),
                run_name=str(record["run_name"]),
                scale=SCALE_60M,
                wandb_run_id=str(record["wandb_run_id"]),
                source_experiment=str(record["source_experiment"]),
                phase_weights=phase_weights,
                metrics=metrics,
            )
        )
    return runs


def _graphql(query: str, variables: dict) -> dict:
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        WANDB_GRAPHQL_URL,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    if payload.get("errors"):
        raise RuntimeError(f"W&B GraphQL errors: {payload['errors']}")
    return payload["data"]


def _iter_wandb_runs(display_name_regex: str):
    after = None
    while True:
        data = _graphql(
            RUNS_QUERY,
            {
                "entity": WANDB_ENTITY,
                "project": WANDB_PROJECT,
                "filters": json.dumps({"display_name": {"$regex": display_name_regex}}),
                "after": after,
            },
        )
        runs = data["project"]["runs"]
        for edge in runs["edges"]:
            yield edge["node"]
        if not runs["pageInfo"]["hasNextPage"]:
            return
        after = runs["pageInfo"]["endCursor"]


def _config_value(node_config: dict, key: str):
    entry = node_config.get(key)
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def _parse_300m_node(node: dict, domains: list[str]) -> dict | None:
    """Extract identity, weights, and metrics from one W&B node; None if not a 300M replay run.

    Display names of the replay are truncated (`ngd3dm2_qsplit2~<hash>/run_NNNNN`) and most runs
    carry no tags, so identification goes through the checkpoint path in the config.
    """
    config = json.loads(node["config"])
    trainer = _config_value(config, "trainer") or {}
    base_path = (trainer.get("checkpointer") or {}).get("base_path", "")
    if f"/{EXPERIMENT_300M}/" not in base_path:
        return None

    run_name = node["displayName"].rsplit("/", 1)[-1]
    data_cfg = _config_value(config, "data")
    train_weights = data_cfg["train_weights"]
    if [b[0] for b in train_weights] != [0, PHASE_STEPS[SCALE_300M][0]]:
        raise ValueError(f"300m {run_name}: unexpected phase boundaries {[b[0] for b in train_weights]}")
    if trainer["num_train_steps"] != PHASE_STEPS[SCALE_300M][1]:
        raise ValueError(f"300m {run_name}: unexpected num_train_steps {trainer['num_train_steps']}")

    phase_weights = []
    for _, weights in train_weights:
        extras = {d: w for d, w in weights.items() if d not in set(domains)}
        nonzero_extras = {d: w for d, w in extras.items() if abs(w) > 1e-12}
        if nonzero_extras:
            raise ValueError(f"300m {run_name}: nonzero weights outside the 39 swarm domains: {nonzero_extras}")
        phase_weights.append({d: float(weights[d]) for d in domains})

    summary = json.loads(node["summaryMetrics"] or "{}")
    metrics = {m: float(summary[m]) for m in METRIC_COLUMNS if m in summary}
    return {
        "run_name": run_name,
        "wandb_run_id": node["name"],
        "state": node["state"],
        "created_at": node["createdAt"],
        "raw_phase_weights": tuple(phase_weights),
        "metrics": metrics,
    }


def pull_300m_metrics(runs_60m: list[SwarmRun], expected_missing: Collection[str] = ()) -> list[SwarmRun]:
    """Pull the 300M/6B replay from public W&B and match it to 60M mixtures by run_name.

    Weight consistency with the 60M CSV is asserted (atol 1e-6) for every matched run; raises if
    fewer than MIN_CONSISTENCY_RUNS could be checked. Matched runs missing the primary metric
    raise unless listed in `expected_missing`.
    """
    domains = swarm_domains()
    by_name_60m = {r.run_name: r for r in runs_60m}

    candidates: dict[str, dict] = {}
    n_wandb = 0
    for node in _iter_wandb_runs("ngd3dm2_qsplit2"):
        parsed = _parse_300m_node(node, domains)
        if parsed is None:
            continue
        n_wandb += 1
        name = parsed["run_name"]
        incumbent = candidates.get(name)
        if incumbent is None or _dedup_rank(parsed) > _dedup_rank(incumbent):
            candidates[name] = parsed
    logger.info("W&B: %d runs under %s, %d distinct run names", n_wandb, EXPERIMENT_300M, len(candidates))

    unmatched = sorted(set(candidates) - set(by_name_60m))
    if unmatched:
        raise ValueError(f"300M W&B runs with no 60M counterpart in the CSV: {unmatched}")

    runs, n_consistent = [], 0
    missing_metric = []
    for name, parsed in sorted(candidates.items()):
        run_60m = by_name_60m[name]
        phase_weights = tuple(
            _normalize_phase(pw, context=f"300m {name} phase {p}") for p, pw in enumerate(parsed["raw_phase_weights"])
        )
        for p in (0, 1):
            got = np.array([phase_weights[p][d] for d in domains])
            want = np.array([run_60m.phase_weights[p][d] for d in domains])
            if not np.allclose(got, want, atol=WEIGHT_CONSISTENCY_ATOL):
                raise ValueError(
                    f"300m {name} phase {p}: weights disagree with 60M CSV "
                    f"(max abs diff {np.max(np.abs(got - want)):.3e} > atol {WEIGHT_CONSISTENCY_ATOL})"
                )
        n_consistent += 1

        if PRIMARY_METRIC not in parsed["metrics"]:
            missing_metric.append(name)
            continue
        runs.append(
            _make_run(
                run_id=run_60m.run_id,
                run_name=name,
                scale=SCALE_300M,
                wandb_run_id=parsed["wandb_run_id"],
                source_experiment=EXPERIMENT_300M,
                phase_weights=phase_weights,
                metrics=parsed["metrics"],
            )
        )

    unexpected = [n for n in missing_metric if n not in set(expected_missing)]
    if unexpected:
        raise ValueError(f"300M runs missing {PRIMARY_METRIC} and not in expected_missing: {unexpected}")
    if n_consistent < MIN_CONSISTENCY_RUNS:
        raise ValueError(f"Only {n_consistent} 300M runs weight-checked against the CSV (< {MIN_CONSISTENCY_RUNS})")
    logger.info(
        "300M: %d matched of %d on W&B; weight consistency verified for %d runs at atol %g",
        len(runs),
        len(candidates),
        n_consistent,
        WEIGHT_CONSISTENCY_ATOL,
    )
    return runs


def _dedup_rank(parsed: dict) -> tuple:
    """Preference order for duplicate wandb runs of one run_name: finished + metric, then newest."""
    return (parsed["state"] == "finished", PRIMARY_METRIC in parsed["metrics"], parsed["created_at"])


def runs_dataframe(runs: list[SwarmRun]) -> pd.DataFrame:
    """Flatten SwarmRuns to one row per (run, scale) with per-phase weight and metric columns."""
    domains = swarm_domains()
    records = []
    for r in runs:
        rec: dict = {
            "run_id": r.run_id,
            "run_name": r.run_name,
            "scale": r.scale,
            "model_size": r.model_size,
            "wandb_run_id": r.wandb_run_id,
            "source_experiment": r.source_experiment,
            "is_vertex": r.is_vertex,
            "vertex_domain": r.vertex_domain,
            "phase_0_tokens": r.phase_tokens[0],
            "phase_1_tokens": r.phase_tokens[1],
        }
        for p in (0, 1):
            for d in domains:
                rec[f"phase_{p}_{d}"] = r.phase_weights[p][d]
        for m in METRIC_COLUMNS:
            rec[m] = r.metrics.get(m)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def domains_dataframe() -> pd.DataFrame:
    payload = json.loads(DOMAIN_TOKEN_COUNTS_JSON.read_text())
    counts, n_parts = payload["available_tokens"], payload["n_partitions"]
    return pd.DataFrame(
        {
            "domain": sorted(counts),
            "available_tokens": [int(counts[d]) for d in sorted(counts)],
            "n_partitions": [int(n_parts[d]) for d in sorted(counts)],
        }
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs_60m = load_qsplit240_60m()
    runs_300m = pull_300m_metrics(runs_60m)
    df = runs_dataframe(runs_60m + runs_300m)

    runs_path = args.output_dir / "runs.parquet"
    df.to_parquet(runs_path, index=False)
    domains_path = args.output_dir / "domains.parquet"
    domains_dataframe().to_parquet(domains_path, index=False)

    for scale, group in df.groupby("scale"):
        logger.info(
            "scale %s: %d runs, %d with non-null %s, %d vertex runs",
            scale,
            len(group),
            int(group[PRIMARY_METRIC].notna().sum()),
            PRIMARY_METRIC,
            int(group["is_vertex"].sum()),
        )
    logger.info("wrote %s and %s", runs_path, domains_path)


if __name__ == "__main__":
    main()
