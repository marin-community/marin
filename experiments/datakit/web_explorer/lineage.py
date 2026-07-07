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

import json
import logging
from dataclasses import dataclass, field

from ducky.client import DuckyClient, DuckyError
from rigging.filesystem import marin_prefix

from experiments.datakit.reference_pipeline import DEFAULT_SCALE, reference_datakit_steps, select_sources
from experiments.datakit.store.datakit_store import ClusteredStoreData

logger = logging.getLogger(__name__)


def read_text(ducky: DuckyClient, path: str) -> str | None:
    """Fetch a file's raw content via ducky (DuckDB ``read_text``), or None if absent.

    Keeps the explorer a pure ducky client: every read — store artifact, lineage
    cache, dedup record — goes through ducky rather than the object store directly,
    so the dashboard needs no storage credentials of its own.
    """
    try:
        rows = ducky.run(f"SELECT content FROM read_text('{path}')").dicts()
    except DuckyError:
        return None
    return rows[0]["content"] if rows else None


# Placeholder inputs let us reconstruct the model-independent stages
# (normalize/tokenize/decontam/dedup) even when the caller has no centroids /
# quality model — those stages' hashes don't depend on either.
_PLACEHOLDER_CENTROIDS = "gs://placeholder/centroids"
_PLACEHOLDER_QUALITY_MODEL = "gs://placeholder/model.bin"

# Known replicas of the stage tree, probed in order when the store's own prefix
# doesn't hold it. The R2 (``marin-na``) mirror is preferred — zero-egress and
# co-located with the CoreWeave ducky — and nests under ``/marin``; the GCS mirrors
# sit at the bucket root and come after.
_FALLBACK_PREFIXES = ("s3://marin-na/marin", "gs://marin-us-central2", "gs://marin-us-east5")


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


def load_lineage(ducky: DuckyClient, path: str) -> StoreLineage:
    """Load a pre-resolved lineage cache JSON, read via ducky (no direct storage).

    The explorer never *writes* the cache (ducky is read-only); a cache is an
    optional pre-baked optimization — absent one, resolution runs at startup.
    """
    content = read_text(ducky, path)
    if content is None:
        raise FileNotFoundError(f"no lineage cache at {path}")
    return StoreLineage(**json.loads(content))


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


def _discover_hashes(ducky: DuckyClient, data_prefix: str, stage_root: str, sources: list[str]) -> dict[str, list[str]]:
    """Map ``stem -> [relative dataset paths]`` for one stage subtree, via ducky glob.

    Datasets sit at ``<stage_root>/<name>_<hash>`` (flat source) or
    ``<stage_root>/<group>/<name>_<hash>`` (grouped source), 1-3 levels deep, indexed
    by the hash-stripped stem to disambiguate later. Depths 1-2 are cheap broad globs;
    a broad depth-3 ``*/*/*`` glob is pathologically slow on R2 (~100s to enumerate the
    tree), so target only the known 3-level source prefixes (a handful, each instant).
    """
    out: dict[str, list[str]] = {}

    def _add(sql: str) -> None:
        for row in ducky.run(sql).rows:
            rel = _relativize(row[0].rsplit("/.artifact.json", 1)[0], data_prefix)
            out.setdefault(_stem(rel), []).append(rel)

    for depth in ("*", "*/*"):
        _add(f"SELECT file FROM glob('{data_prefix}/{stage_root}/{depth}/.artifact.json')")
    # 3-level sources (e.g. ``safety_pt/moral_education/score_5_morals``): glob each
    # distinct 2-level prefix + leaf rather than a broad, slow ``*/*/*``.
    for prefix in sorted({s.rsplit("/", 1)[0] for s in sources if s.count("/") == 2}):
        _add(f"SELECT file FROM glob('{data_prefix}/{stage_root}/{prefix}/*/.artifact.json')")
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
    # The stage tree sits beside the store under the same prefix: strip the trailing
    # "/datakit/store_<hash>" to get "<scheme>://<bucket>[/<path>]" holding ``datakit/``
    # + ``normalized/``. Only probe same-scheme replicas — a gs:// fallback from an
    # s3:// (CoreWeave/R2) store would be a wrong-scheme, cross-region read.
    store_prefix = store_path.rsplit("/datakit/", 1)[0]
    scheme = f"{store_path.split('://', 1)[0]}://"
    candidates = [store_prefix, *(p for p in _FALLBACK_PREFIXES if p.startswith(scheme) and p != store_prefix)]
    best, best_yield = store_prefix, -1
    for prefix in candidates:
        y = _discovery_yield(ducky, prefix, _STAGE_ROOTS["tokenize"])
        logger.info("data-prefix probe %s: tokenize discovery yield=%d", prefix, y)
        if y > best_yield:
            best, best_yield = prefix, y
    if best_yield <= 0:
        logger.warning("no bucket discovered tokenize datasets; defaulting to %s", store_prefix)
    return best


def read_store_payload(ducky: DuckyClient, store_path: str) -> ClusteredStoreData:
    """Load the store's :class:`ClusteredStoreData` via ducky, tolerating both formats.

    New stores wrap the payload in an ``ArtifactRecord`` (``result`` holds the
    payload); legacy stores (e.g. ``store_8ac06c74``) wrote the raw payload directly
    to ``.artifact.json``. Read the file's text via ducky and parse either shape.
    """
    base = store_path.rstrip("/")
    for name in ("artifact.json", ".artifact.json"):
        content = read_text(ducky, f"{base}/{name}")
        if content is None:
            continue
        doc = json.loads(content)
        # new-format ArtifactRecord nests the payload under "result"; legacy is bare.
        payload = doc["result"] if isinstance(doc, dict) and "result" in doc else doc
        return ClusteredStoreData.model_validate(payload)
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
    # New-format stores write ``artifact.json`` as an ``ArtifactRecord`` carrying
    # ``deps``; legacy stores write a bare payload to ``.artifact.json``.
    content = read_text(ducky, f"{store_path}/artifact.json")
    if content is not None:
        doc = json.loads(content)
        if isinstance(doc, dict) and doc.get("deps"):
            return _resolve_from_record(store_path, doc)
    return _resolve_legacy(store_path, ducky, domain_centroids=domain_centroids, quality_model=quality_model)


def _resolve_legacy(
    store_path: str,
    ducky: DuckyClient,
    *,
    domain_centroids: str | None,
    quality_model: str | None,
) -> StoreLineage:
    payload = read_store_payload(ducky, store_path)
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
        discovered = _discover_hashes(ducky, data_prefix, root, sources)
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


def _resolve_from_record(store_path: str, record: dict) -> StoreLineage:
    """Resolve lineage from a new-format ``ArtifactRecord`` dict (walks ``deps``).

    ``deps`` are ``name@version`` refs whose recorded steps live at sibling
    output paths; we classify each by its step name. Kept minimal until a
    new-format store exists to exercise it end-to-end.
    """
    raise NotImplementedError(
        "new-format ArtifactRecord lineage walking is not wired yet; "
        "store_8ac06c74 and current datakit runs use the legacy payload path"
    )
