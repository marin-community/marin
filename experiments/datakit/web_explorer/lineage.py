# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a datakit store to the dataset paths of every stage that fed it.

The dashboard is pointed at one clustered-store output and needs to reach its
upstream stages (normalize / tokenize / quality / decontam / cluster_assign /
dedup) to query them. Two artifact formats are supported:

* **New** (post lazy-``ArtifactStep`` refactor): ``artifact.json`` is a full
  :class:`~marin.execution.artifact.ArtifactRecord` carrying ``deps`` +
  ``config`` — the lineage is recorded, so we read it straight off.

* **Legacy** (e.g. ``store_8ac06c74``): ``.artifact.json`` is a bare
  :class:`~experiments.datakit.store.datakit_store.ClusteredStoreData` payload
  with no dependency paths. We rebuild the reference DAG in-process
  (:func:`reference_datakit_steps`) — every step's ``output_path`` is a
  content-addressed hash, so the reconstructed relative paths match GCS exactly
  (verified: ``tokenize/nsf_awards_d9eefb77`` == the real object). The stages
  that don't depend on the domain-centroids / quality-model
  (normalize/tokenize/decontam/dedup) resolve with no extra input; ``quality``
  and ``cluster_assign`` need those two paths, which the store hash verifies.

Physical location: reconstructed paths are prefix-independent (only the hash
matters), so we re-root the relative path onto whichever candidate bucket
actually holds the data.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field

from ducky.client import DuckyClient
from marin.execution.artifact import read_artifact, read_record
from rigging.filesystem import marin_prefix, open_url

from experiments.datakit.reference_pipeline import DEFAULT_SCALE, reference_datakit_steps, select_sources
from experiments.datakit.store.datakit_store import ClusteredStoreData

logger = logging.getLogger(__name__)

# Placeholder inputs let us reconstruct the model-independent stages
# (normalize/tokenize/decontam/dedup) even when the caller has no centroids /
# quality model — those stages' hashes don't depend on either.
_PLACEHOLDER_CENTROIDS = "gs://placeholder/centroids"
_PLACEHOLDER_QUALITY_MODEL = "gs://placeholder/model.bin"

# Buckets to probe for the physical copy of a reconstructed dataset, tried in
# order. The store's own bucket first, then known replicas.
_FALLBACK_PREFIXES = ("gs://marin-us-central2", "gs://marin-us-east5")


@dataclass(frozen=True)
class StoreLineage:
    """Resolved dataset paths for a store and its upstream stages.

    Per-source dicts map ``source_name -> dataset path``. ``quality`` and
    ``cluster_assign`` are empty when no domain-centroids / quality-model were
    supplied (they can't be resolved without them). ``verified`` is True only
    when a reconstruction reproduced the store's own hash.
    """

    store_path: str
    data_prefix: str
    cluster_view: int
    quality_thresholds: list[float]
    tokenizer: str
    source_names: list[str]
    normalize: dict[str, str] = field(default_factory=dict)
    tokenize: dict[str, str] = field(default_factory=dict)
    quality: dict[str, str] = field(default_factory=dict)
    decontam: dict[str, str] = field(default_factory=dict)
    cluster_assign: dict[str, str] = field(default_factory=dict)
    dedup: str | None = None
    verified: bool = False


def save_lineage(lineage: StoreLineage, path: str) -> None:
    """Cache a resolved lineage as JSON (resolution costs ~2 min of ducky globs)."""
    with open_url(path, "w") as f:
        json.dump(dataclasses.asdict(lineage), f, indent=2)


def load_lineage(path: str) -> StoreLineage:
    """Load a cached lineage written by :func:`save_lineage`."""
    with open_url(path, "r") as f:
        return StoreLineage(**json.load(f))


def _hash_suffix(path: str) -> str:
    """The trailing ``_<hash>`` of a step output dir (``store_8ac06c74`` -> ``8ac06c74``)."""
    return path.rstrip("/").rsplit("_", 1)[-1]


def _relativize(path: str, ambient_prefix: str) -> str:
    """Strip the ambient marin prefix, leaving the bucket-relative object path."""
    ambient_prefix = ambient_prefix.rstrip("/")
    if path.startswith(ambient_prefix + "/"):
        return path[len(ambient_prefix) + 1 :]
    # Fallback: strip any gs://bucket/ prefix.
    without_scheme = path.split("://", 1)[-1]
    return without_scheme.split("/", 1)[1] if "/" in without_scheme else without_scheme


def _stem(relative_path: str) -> str:
    """Drop the trailing ``_<hash>`` segment: ``datakit/decontam/nsf_awards_25c0`` -> ``datakit/decontam/nsf_awards``."""
    return relative_path.rstrip("/").rsplit("_", 1)[0]


def _discover_hashes(ducky: DuckyClient, data_prefix: str, stage_root: str) -> dict[str, list[str]]:
    """Map ``stem -> [relative dataset paths]`` for one stage subtree, via ducky glob.

    Datasets sit at ``<stage_root>/<name>_<hash>`` (flat source) or
    ``<stage_root>/<group>/<name>_<hash>`` (grouped source), so we glob both
    depths and index by the hash-stripped stem to disambiguate later.
    """
    out: dict[str, list[str]] = {}
    # Source names have 1-3 slash-separated parts (nsf_awards / cp/foodista /
    # safety_pt/moral_education/score_5_morals), so datasets sit 1-3 levels under
    # the stage root — glob all three depths.
    for depth in ("*", "*/*", "*/*/*"):
        sql = f"SELECT file FROM glob('{data_prefix}/{stage_root}/{depth}/.artifact.json')"
        for row in ducky.run(sql).rows:
            rel = _relativize(row[0].rsplit("/.artifact.json", 1)[0], data_prefix)
            out.setdefault(_stem(rel), []).append(rel)
    return out


def _pick(reconstructed_rel: str, discovered: dict[str, list[str]]) -> str | None:
    """Resolve the actual dataset path for a reconstructed one.

    Prefer the exact reconstructed hash (correct when code/config/models are
    unchanged); else the sole on-GCS candidate; else ``None`` (ambiguous —
    multiple hashes and none match the reconstruction).
    """
    candidates = discovered.get(_stem(reconstructed_rel), [])
    if reconstructed_rel in candidates:
        return reconstructed_rel
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    logger.warning("ambiguous dataset for %s: %d candidates %s", _stem(reconstructed_rel), len(candidates), candidates)
    return None


def _discovery_yield(ducky: DuckyClient, data_prefix: str, stage_root: str) -> int:
    """How many datasets a broad glob discovers under a stage root (depth 1 + 2)."""
    total = 0
    for depth in ("*", "*/*"):
        total += ducky.run(f"SELECT count(*) FROM glob('{data_prefix}/{stage_root}/{depth}/.artifact.json')").scalar()
    return total


def _pick_data_prefix(store_path: str, ducky: DuckyClient) -> str:
    """Choose the bucket where discovery actually yields the stage datasets.

    Data is replicated across regions, but broad ``*`` globs only enumerate
    fully in the bucket that holds the complete stage tree (others list sparsely
    or not at all). Pick the candidate with the highest tokenize-discovery yield;
    ties favor the store's own bucket.
    """
    store_prefix = "gs://" + store_path.split("://", 1)[-1].split("/", 1)[0]
    candidates = [store_prefix, *(p for p in _FALLBACK_PREFIXES if p != store_prefix)]
    best, best_yield = store_prefix, -1
    for prefix in candidates:
        y = _discovery_yield(ducky, prefix, _STAGE_ROOTS["tokenize"])
        logger.info("data-prefix probe %s: tokenize discovery yield=%d", prefix, y)
        if y > best_yield:
            best, best_yield = prefix, y
    if best_yield <= 0:
        logger.warning("no bucket discovered tokenize datasets; defaulting to %s", store_prefix)
    return best


def read_store_payload(store_path: str) -> ClusteredStoreData:
    """Load the store's :class:`ClusteredStoreData`, tolerating both artifact formats.

    New stores wrap the payload in an ``ArtifactRecord.result`` (``read_artifact``).
    Legacy stores (e.g. ``store_8ac06c74``) wrote the raw payload directly to
    ``.artifact.json`` — which the current ``read_artifact`` no longer reads (it
    now expects a record), so fall back to parsing that file as the payload.
    """
    try:
        return read_artifact(store_path, ClusteredStoreData)
    except (FileNotFoundError, ValueError):
        pass
    base = store_path.rstrip("/")
    for name in (".artifact.json", "artifact.json"):
        try:
            with open_url(f"{base}/{name}", "r") as f:
                return ClusteredStoreData.model_validate_json(f.read())
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"no ClusteredStoreData payload at {store_path}")


# Stage -> subtree root under the data prefix (normalize lives at ``normalized/``,
# not ``datakit/normalize/``).
_STAGE_ROOTS = {
    "normalize": "normalized",
    "tokenize": "datakit/tokenize",
    "quality": "datakit/quality",
    "decontam": "datakit/decontam",
    "cluster_assign": "datakit/cluster_assign",
}


def resolve_lineage(
    store_path: str,
    ducky: DuckyClient,
    *,
    domain_centroids: str | None = None,
    quality_model: str | None = None,
) -> StoreLineage:
    """Resolve ``store_path`` to its upstream stage datasets. See module docstring."""
    store_path = store_path.rstrip("/")
    record = read_record(store_path)
    if record is not None and record.deps:
        return _resolve_from_record(store_path, record)
    return _resolve_legacy(store_path, ducky, domain_centroids=domain_centroids, quality_model=quality_model)


def _resolve_legacy(
    store_path: str,
    ducky: DuckyClient,
    *,
    domain_centroids: str | None,
    quality_model: str | None,
) -> StoreLineage:
    payload = read_store_payload(store_path)
    known = set(select_sources(None))
    sources = [s for s in payload.source_names if s in known]
    dropped = sorted(set(payload.source_names) - known)
    if dropped:
        logger.warning("%d store sources not in current registry, skipping: %s", len(dropped), dropped[:8])

    have_models = domain_centroids is not None and quality_model is not None
    steps = reference_datakit_steps(
        select_sources(sources),
        domain_centroids=domain_centroids or _PLACEHOLDER_CENTROIDS,
        quality_model=quality_model or _PLACEHOLDER_QUALITY_MODEL,
        scale=DEFAULT_SCALE,
    )
    ambient = marin_prefix()

    # Reconstructed relative paths per stage (exact stem/naming; hash exact only
    # where code/config/models are unchanged).
    recon: dict[str, dict[str, str]] = {s: {} for s in _STAGE_ROOTS}
    recon["normalize"] = {name: _relativize(step.output_path, ambient) for name, step in steps.sources.items()}
    dedup_recon: str | None = None
    for step in steps.all_steps:
        for stage in ("tokenize", "quality", "decontam", "cluster_assign"):
            prefix = f"datakit/{stage}/"
            if step.name.startswith(prefix):
                recon[stage][step.name[len(prefix) :]] = _relativize(step.output_path, ambient)
        if step.name == "datakit/dedup":
            dedup_recon = _relativize(step.output_path, ambient)

    verified = have_models and _hash_suffix(steps.output_buckets.output_path) == _hash_suffix(store_path)
    if have_models and not verified:
        logger.warning(
            "reconstructed store hash %s != %s — centroids/quality-model likely wrong",
            _hash_suffix(steps.output_buckets.output_path),
            _hash_suffix(store_path),
        )

    data_prefix = _pick_data_prefix(store_path, ducky)

    # Discover actual on-GCS hashes per stage and resolve each reconstructed path
    # to its real dataset (reconstructed hash if present, else sole candidate).
    resolved: dict[str, dict[str, str]] = {}
    for stage, root in _STAGE_ROOTS.items():
        discovered = _discover_hashes(ducky, data_prefix, root)
        picked = {name: _pick(rel, discovered) for name, rel in recon[stage].items()}
        missing = sorted(n for n, p in picked.items() if p is None)
        if missing:
            logger.warning("%s: unresolved for %d/%d sources (e.g. %s)", stage, len(missing), len(picked), missing[:5])
        resolved[stage] = {name: f"{data_prefix}/{rel}" for name, rel in picked.items() if rel}

    # dedup sits directly at datakit/dedup_<hash> (no per-source nesting).
    dedup_dirs = [
        _relativize(row[0].rsplit("/.artifact.json", 1)[0], data_prefix)
        for row in ducky.run(f"SELECT file FROM glob('{data_prefix}/datakit/dedup_*/.artifact.json')").rows
    ]
    dedup_path = None
    if dedup_recon:
        dedup_rel = _pick(dedup_recon, {"datakit/dedup": dedup_dirs})
        dedup_path = f"{data_prefix}/{dedup_rel}" if dedup_rel else None

    return StoreLineage(
        store_path=store_path,
        data_prefix=data_prefix,
        cluster_view=payload.cluster_view,
        quality_thresholds=payload.quality_thresholds,
        tokenizer=payload.tokenizer,
        source_names=sources,
        normalize=resolved["normalize"],
        tokenize=resolved["tokenize"],
        quality=resolved["quality"],
        decontam=resolved["decontam"],
        cluster_assign=resolved["cluster_assign"],
        dedup=dedup_path,
        verified=verified,
    )


def _resolve_from_record(store_path: str, record) -> StoreLineage:
    """Resolve lineage from a new-format ``ArtifactRecord`` (walks ``deps``).

    ``deps`` are ``name@version`` refs whose recorded steps live at sibling
    output paths; we classify each by its step name. Kept minimal until a
    new-format store exists to exercise it end-to-end.
    """
    raise NotImplementedError(
        "new-format ArtifactRecord lineage walking is not wired yet; "
        "store_8ac06c74 and current datakit runs use the legacy payload path"
    )
