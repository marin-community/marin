# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute verified fuzzy-duplicate markers from MinHash candidates.

MinHash LSH only retrieves candidates. A document is marked ``dup_doc=True``
after an exact, full-text comparison to the connected component's canonical,
and only when it shared an LSH bucket directly with that retained canonical.
Rejected candidates and their exact scores are persisted for audit.
"""

import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import MAX_WORKERS_PER_JOB, ZephyrContext
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.connected_components import connected_components
from marin.processing.classification.deduplication.dedup_commons import _load_batches
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData, MinHashParams
from marin.processing.classification.deduplication.fuzzy_verification import (
    FuzzyVerificationParams,
    VerificationResult,
    verify_candidate,
)

logger = logging.getLogger(__name__)

FUZZY_DUPS_CANDIDATE_SCOPE = "direct_canonical_exact_v1"
_VERIFICATION_COUNTER = "dedup/fuzzy/verification"
_SCORE_HISTOGRAM_MAX_PERCENT = 100
_UNIQUE_NGRAM_HISTOGRAM_OVERFLOW_BIN = 33


class FuzzyDupsPerSource(BaseModel):
    """Per-source output entry inside :class:`FuzzyDupsAttrData`.

    Attributes:
        attr_dir: Directory containing per-shard duplicate marker Parquet
            files. Filenames mirror the source's MinHash attr (and thus its
            normalized) shards.
    """

    attr_dir: str


class FuzzyDupsAttrData(BaseModel):
    """Co-partitioned verified-duplicate markers for one or more sources.

    Persisted as the step's ``.artifact``. Load via
    ``Artifact.from_path(step, FuzzyDupsAttrData)``.

    Attributes:
        version: Schema version of this artifact.
        params: MinHash params; equal to every input's params.
        verification: Exact verification thresholds.
        decisions_dir: Parquet evidence for accepted and rejected candidates.
        sources: Mapping from each input's ``MinHashAttrData.source_main_dir``
            to its per-source attr output entry.
        counters: Aggregated zephyr counters across all sources.
    """

    version: str = "v2"
    params: MinHashParams
    verification: FuzzyVerificationParams
    decisions_dir: str
    sources: dict[str, FuzzyDupsPerSource]
    counters: dict[str, int | float]


def _validate_inputs(inputs: list[MinHashAttrData]) -> MinHashParams:
    """Ensure every input shares the same MinHash params and a unique source; raise otherwise."""
    if not inputs:
        raise ValueError("compute_fuzzy_dups_attrs requires at least one input")

    head = inputs[0].params
    mismatched = [(i, m.params) for i, m in enumerate(inputs) if m.params != head]
    if mismatched:
        details = "; ".join(f"inputs[{i}]={p}" for i, p in mismatched)
        raise ValueError(
            f"All MinHashAttrData inputs must share identical MinHash params. "
            f"inputs[0]={head} but mismatches: {details}"
        )

    seen: dict[str, int] = {}
    for i, m in enumerate(inputs):
        if m.source_main_dir in seen:
            raise ValueError(
                f"Duplicate source_main_dir in inputs: inputs[{seen[m.source_main_dir]}] and "
                f"inputs[{i}] both point to {m.source_main_dir!r}. Each source must be "
                "represented at most once so its output attr tree is unambiguous."
            )
        seen[m.source_main_dir] = i

    return head


def _build_shard_index(inputs: list[MinHashAttrData]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Enumerate every source shard across *inputs* and assign a global file_idx.

    Returns:
        (entries, source_tag_for_input) where ``entries[file_idx]`` holds
        ``{attr_path, source_main_dir, source_tag, basename}`` and
        ``source_tag_for_input[source_main_dir] = "source_NNN"``.
    """
    # Sort inputs by source_main_dir so source_tags are deterministic regardless
    # of the order callers happen to pass them in.
    ordered = sorted(enumerate(inputs), key=lambda iv: iv[1].source_main_dir)
    source_tag: dict[str, str] = {}
    for new_idx, (_, m) in enumerate(ordered):
        source_tag[m.source_main_dir] = f"source_{new_idx:03d}"

    entries: list[dict[str, Any]] = []
    for m in inputs:
        attr_shards = sorted(str(shard) for shard in StoragePath(f"{m.attr_dir.rstrip('/')}/*.parquet").glob())
        if not attr_shards:
            raise FileNotFoundError(f"No attr parquet shards under {m.attr_dir}")
        for attr_path in attr_shards:
            basename = os.path.basename(attr_path)
            entries.append(
                {
                    "file_idx": len(entries),
                    "attr_path": attr_path,
                    "source_path": f"{m.source_main_dir.rstrip('/')}/{basename}",
                    "source_main_dir": m.source_main_dir,
                    "source_tag": source_tag[m.source_main_dir],
                    "basename": basename,
                }
            )
    return entries, source_tag


# Separator between the per-source CC tag and the original content-hash id.
# "|" can't appear in the hex-digit content hashes produced by normalize's
# generate_id, so splitting on the first "|" is unambiguous.
_CC_ID_SEP = "|"


def _cc_id(source_tag: str, doc_id: str) -> str:
    """Prefix *doc_id* with *source_tag* so CC treats cross-source collisions as distinct nodes.

    ``connected_components`` keys nodes by a hash of the record id. Two
    inputs can carry byte-identical normalized ids (e.g. exact text overlap
    across datasets), and without this prefix they collapse to a single
    node — under-reporting dups and potentially clobbering co-partitioned
    attr files. The prefix is stripped in :func:`_strip_cc_prefix` before
    the final attr parquet is written.
    """
    return f"{source_tag}{_CC_ID_SEP}{doc_id}"


def _strip_cc_prefix(record_id: str) -> str:
    """Reverse :func:`_cc_id`, returning the original ``doc_id``."""
    return record_id.split(_CC_ID_SEP, 1)[1]


def _emit_bucket_records(entries: list[dict[str, Any]]) -> Iterator[dict]:
    """For each (bucket, id) pair across all attr shards in *entries*, emit a routing record."""
    for entry in entries:
        for batch in _load_batches(entry["attr_path"], columns=["id", "buckets"]):
            ids = batch["id"]
            buckets_col = batch["buckets"]
            for doc_id, doc_buckets in zip(ids, buckets_col, strict=True):
                if not doc_buckets.is_valid:
                    continue
                cc_id = _cc_id(entry["source_tag"], doc_id.as_py())
                for b in doc_buckets.as_py():
                    yield {"bucket": str(b), "id": cc_id, "file_idx": entry["file_idx"]}


# Key under which ``entries`` is staged via ``ZephyrContext.put`` for the
# stage1 reducer. Holding the list in the closure would serialize it into
# every ``pull_task`` RPC pickle (one per dispatched reduce task) — at
# ~100k entries this OOMs the coord under the high-fan-out dispatch that
# kicks in when ``max_parallelism`` is large. Pulling it from shared data
# instead means each worker fetches and caches the list once.
_SHARED_ENTRIES_KEY = "fuzzy_dups_entries"


def _candidate_node(record: dict[str, Any]) -> dict[str, Any]:
    """Reduce one CC node to the fields required for direct-canonical matching."""
    id_norm = record["id_norm"]
    component_id = record["component_id"]
    adjacency = record["adjacency_list"]
    is_canonical = component_id == id_norm
    is_singleton = len(adjacency) == 1 and adjacency[0] == id_norm
    is_direct_candidate = is_canonical or component_id in adjacency
    if is_singleton:
        counters.pipeline.update_counter("dedup/fuzzy/document/singletons_skipped", 1)
    elif not is_direct_candidate:
        counters.pipeline.update_counter("dedup/fuzzy/document/transitive_members_kept", 1)
    return {
        "doc_id": _strip_cc_prefix(record["record_id"]),
        "id_norm": id_norm,
        "component_id": component_id,
        "file_idx": record["file_idx"],
        "is_canonical": is_canonical,
        "canonical_order": 0 if is_canonical else 1,
        "emit": not is_singleton and is_direct_candidate,
    }


def _candidate_pairs(component_id: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Pair every direct member with the component canonical."""
    canonical = next(records)
    if not canonical["is_canonical"] or canonical["id_norm"] != component_id:
        raise AssertionError(f"Component {component_id} did not start with its canonical")
    for member in records:
        if member["is_canonical"]:
            raise AssertionError(f"Component {component_id} has multiple canonicals")
        yield {
            "pair_id": member["id_norm"],
            "component_id": component_id,
            "member_id": member["doc_id"],
            "member_file_idx": member["file_idx"],
            "canonical_id": canonical["doc_id"],
            "canonical_file_idx": canonical["file_idx"],
        }


def _candidate_text_requests(pair: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for role, role_order in (("canonical", 0), ("member", 1)):
        yield {
            **pair,
            "role": role,
            "role_order": role_order,
            "file_idx": pair[f"{role}_file_idx"],
            "document_id": pair[f"{role}_id"],
        }


def _rows(path: str, columns: list[str]) -> Iterator[dict[str, Any]]:
    for batch in _load_batches(path, columns=columns):
        yield from batch.to_pylist()


def _requested_texts(file_idx: int, requests: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Resolve requests against the MinHash shard's order-preserving source subsequence."""
    entry = zephyr_worker_ctx().get_shared(_SHARED_ENTRIES_KEY)[file_idx]
    requests_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        requests_by_id[request["document_id"]].append(request)

    minhash_rows = iter(_rows(entry["attr_path"], ["id", "buckets"]))
    minhash_row = next(minhash_rows, None)
    for source_row in _rows(entry["source_path"], ["id", "text"]):
        if minhash_row is None or source_row["id"] != minhash_row["id"]:
            if source_row["id"] in requests_by_id:
                raise AssertionError(f"Missing MinHash buckets for {source_row['id']} in {entry['attr_path']}")
            continue

        matching_requests = requests_by_id.pop(source_row["id"], ())
        for request in matching_requests:
            counters.pipeline.update_counter(f"{_VERIFICATION_COUNTER}/text_reads", 1)
            counters.pipeline.update_counter(f"{_VERIFICATION_COUNTER}/text_chars", len(source_row["text"]))
            yield {
                **request,
                "text": source_row["text"],
                "buckets": minhash_row["buckets"],
            }
        minhash_row = next(minhash_rows, None)

    if minhash_row is not None:
        raise AssertionError(f"MinHash row {minhash_row['id']} is absent from {entry['source_path']}")

    if requests_by_id:
        missing_id = min(requests_by_id)
        raise AssertionError(f"Missing source text and MinHash buckets for {missing_id} in {entry['source_path']}")


def _result_fields(result: VerificationResult) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "rejection": result.rejection.value if result.rejection is not None else None,
        "member_chars": result.member_chars,
        "canonical_chars": result.canonical_chars,
        "member_tokens": result.member_tokens,
        "canonical_tokens": result.canonical_tokens,
        "member_ngrams": result.member_ngrams,
        "canonical_ngrams": result.canonical_ngrams,
        "shared_ngrams": result.shared_ngrams,
        "member_unique_ngrams": result.member_unique_ngrams,
        "member_containment": result.member_containment,
        "jaccard": result.jaccard,
        "under_tokenized": result.under_tokenized,
        "char_jaccard": result.char_jaccard,
    }


def _score_bin(score: float) -> str:
    return f"{min(int(score * 100), _SCORE_HISTOGRAM_MAX_PERCENT):03d}"


def _make_candidate_verifier(params: FuzzyVerificationParams):
    def verify(pair_id: str, pieces: Iterator[dict[str, Any]]) -> dict[str, Any]:
        canonical = next(pieces)
        member = next(pieces)
        if canonical["role"] != "canonical" or member["role"] != "member":
            raise AssertionError(f"Candidate {pair_id} text pieces are out of order")
        if next(pieces, None) is not None:
            raise AssertionError(f"Candidate {pair_id} has more than two text pieces")

        result = verify_candidate(member["text"], canonical["text"], params)
        shared_buckets = len(set(member["buckets"]) & set(canonical["buckets"]))
        if not shared_buckets:
            raise AssertionError(f"Direct candidate {pair_id} has no shared LSH bucket")

        entries = zephyr_worker_ctx().get_shared(_SHARED_ENTRIES_KEY)
        member_entry = entries[member["member_file_idx"]]
        canonical_entry = entries[canonical["canonical_file_idx"]]
        rejection = result.rejection.value if result.rejection is not None else "accepted"
        counters.pipeline.update_counter(f"{_VERIFICATION_COUNTER}/candidates", 1)
        counters.pipeline.update_counter(f"{_VERIFICATION_COUNTER}/decision/{rejection}", 1)
        counters.pipeline.update_counter(
            f"{_VERIFICATION_COUNTER}/source/{member_entry['source_tag']}/decision/{rejection}",
            1,
        )
        counters.pipeline.update_counter(
            f"{_VERIFICATION_COUNTER}/histogram/member_containment/{_score_bin(result.member_containment)}",
            1,
        )
        counters.pipeline.update_counter(
            f"{_VERIFICATION_COUNTER}/histogram/jaccard/{_score_bin(result.jaccard)}",
            1,
        )
        counters.pipeline.update_counter(
            f"{_VERIFICATION_COUNTER}/histogram/member_unique/"
            f"{min(result.member_unique_ngrams, _UNIQUE_NGRAM_HISTOGRAM_OVERFLOW_BIN)}",
            1,
        )
        counters.pipeline.update_counter(
            f"{_VERIFICATION_COUNTER}/histogram/shared_buckets/{shared_buckets}",
            1,
        )
        return {
            "pair_id": pair_id,
            "component_id": member["component_id"],
            "member_id": member["member_id"],
            "member_file_idx": member["member_file_idx"],
            "member_source_main_dir": member_entry["source_main_dir"],
            "canonical_id": canonical["canonical_id"],
            "canonical_file_idx": canonical["canonical_file_idx"],
            "canonical_source_main_dir": canonical_entry["source_main_dir"],
            "shared_buckets": shared_buckets,
            "verification_rule": params.rule_version,
            **_result_fields(result),
        }

    return verify


def _marker_records(decision: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield {
        "marker_id": decision["member_id"],
        "marker_file_idx": decision["member_file_idx"],
        "attributes": {
            "dup_doc": True,
            "dup_cluster_id": decision["component_id"],
            "dup_representative_id": decision["canonical_id"],
            "dup_verifier_version": decision["verification_rule"],
            "dup_shared_buckets": decision["shared_buckets"],
            "dup_member_containment": decision["member_containment"],
            "dup_jaccard": decision["jaccard"],
            "dup_member_unique_ngrams": decision["member_unique_ngrams"],
            "dup_under_tokenized": decision["under_tokenized"],
            "dup_char_jaccard": decision["char_jaccard"],
        },
    }


def _make_marker_writer(output_path: str):
    def aggregate(file_idx: int, markers: Iterator[dict[str, Any]]) -> dict[str, Any]:
        entry = zephyr_worker_ctx().get_shared(_SHARED_ENTRIES_KEY)[file_idx]
        out_path = f"{output_path}/outputs/{entry['source_tag']}/{entry['basename']}"
        count = 0

        def marker_rows() -> Iterator[dict[str, Any]]:
            nonlocal count
            previous_id = None
            previous_attributes = None
            for marker in markers:
                if marker["marker_id"] == previous_id:
                    if marker["attributes"] != previous_attributes:
                        raise AssertionError(f"Conflicting verified markers for {previous_id}")
                    continue
                previous_id = marker["marker_id"]
                previous_attributes = marker["attributes"]
                if marker["attributes"]["dup_doc"]:
                    count += 1
                    counters.pipeline.update_counter("dedup/fuzzy/document/verified_duplicates", 1)
                yield {"id": marker["marker_id"], "attributes": marker["attributes"]}

        result = write_parquet_file(marker_rows(), out_path)
        return {**result, "file_idx": file_idx, "verified_duplicates": count}

    return aggregate


_MARKER_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field(
            "attributes",
            pa.struct(
                [
                    pa.field("dup_doc", pa.bool_()),
                    pa.field("dup_cluster_id", pa.string()),
                    pa.field("dup_representative_id", pa.string()),
                    pa.field("dup_verifier_version", pa.string()),
                    pa.field("dup_shared_buckets", pa.int64()),
                    pa.field("dup_member_containment", pa.float64()),
                    pa.field("dup_jaccard", pa.float64()),
                    pa.field("dup_member_unique_ngrams", pa.int64()),
                    pa.field("dup_under_tokenized", pa.bool_()),
                    pa.field("dup_char_jaccard", pa.float64()),
                ]
            ),
        ),
    ]
)


def _make_empty_marker_writer(output_path: str):
    def ensure_outputs(entries: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for entry in entries:
            out_path = f"{output_path}/outputs/{entry['source_tag']}/{entry['basename']}"
            if StoragePath(out_path).exists():
                yield {"file_idx": entry["file_idx"], "created": False}
                continue
            StoragePath(os.path.dirname(out_path)).mkdirs(exist_ok=True)
            with StoragePath(out_path).open("wb") as stream:
                pq.write_table(pa.Table.from_pylist([], schema=_MARKER_SCHEMA), stream)
            counters.pipeline.update_counter("dedup/fuzzy/document/empty_attr_files", 1)
            yield {"file_idx": entry["file_idx"], "created": True}

    return ensure_outputs


def compute_fuzzy_dups_attrs(
    *,
    inputs: list[MinHashAttrData],
    output_path: str,
    verification_params: FuzzyVerificationParams | None = None,
    cc_max_iterations: int = 10,
    cc_resume: bool = False,
    max_parallelism: int = MAX_WORKERS_PER_JOB,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
) -> FuzzyDupsAttrData:
    """Mark only direct candidates that pass exact full-text verification.

    Connected components identify a deterministic retained canonical, but
    transitive members are never deleted. Each direct canonical neighbor is
    joined back to full normalized text and scored exactly. Accepted members
    receive a sparse ``dup_doc=True`` marker with the representative ID and
    scores; rejected decisions are retained under ``metadata/decisions``.

    Args:
        inputs: ``MinHashAttrData`` artifacts to fuzzy-dedup together.
        output_path: Output root. Per-source attr trees land under
            ``<output_path>/outputs/source_NNN/``.
        verification_params: Exact verifier thresholds. Defaults to the
            precision-first exact token-subset rule.
        cc_max_iterations: Max iterations for connected components.
        max_parallelism: Worker count for the ZephyrContext.
        worker_resources: Per-worker resource request. Required when
            ``map_task_resources`` is set.
        coordinator_resources: Coordinator resource request.
        map_task_resources: ResourceConfig for map-stage tasks.
        reduce_task_resources: ResourceConfig for reduce-stage tasks (e.g.
            the per-shard ``group_by`` writer).

    Raises:
        ValueError: If inputs is empty or input params disagree.
        FileNotFoundError: If any input ``attr_dir`` is missing parquet shards.
    """
    params = _validate_inputs(inputs)
    verification = verification_params or FuzzyVerificationParams()
    entries, source_tag = _build_shard_index(inputs)

    logger.info(
        "Computing fuzzy dups for %d inputs (%d total shards) → %s, params=%s",
        len(inputs),
        len(entries),
        output_path,
        params,
    )

    ctx_kwargs: dict = {
        "name": "fuzzy-dups",
        "max_workers": max_parallelism,
        "resources": worker_resources or ResourceConfig(cpu=1, ram="32g", disk="5g"),
    }
    if coordinator_resources is not None:
        ctx_kwargs["coordinator_resources"] = coordinator_resources
    if map_task_resources is not None:
        ctx_kwargs["map_task_resources"] = map_task_resources
    if reduce_task_resources is not None:
        ctx_kwargs["reduce_task_resources"] = reduce_task_resources
    ctx = ZephyrContext(**ctx_kwargs)

    # Cap shard count at max_parallelism. Each group reads its attr files
    # sequentially and emits bucket records; file_idx is preserved on the entry
    # itself, not by enumeration order, so grouping is safe.
    n_groups = min(max_parallelism, len(entries))
    entry_groups: list[list[dict[str, Any]]] = [[] for _ in range(n_groups)]
    for i, entry in enumerate(entries):
        entry_groups[i % n_groups].append(entry)

    bucket_ds = Dataset.from_list(entry_groups).flat_map(_emit_bucket_records)
    converged, cc_files = connected_components(
        bucket_ds,
        ctx,
        output_dir=f"{output_path}/metadata/cc",
        max_iterations=cc_max_iterations,
        resume=cc_resume,
    )
    if not converged:
        # A non-converged CC is still deterministic and reproducible across
        # runs/executor counts (the bucket group_by sorts by id_norm, pinning the
        # graph topology -- see connected_components), but it is *incomplete*: some
        # true duplicate-clusters remain split across several component_ids, each
        # keeping its own local-min canonical, so a few extra near-dups survive.
        # Warn rather than fail -- callers that cap iterations get a stable,
        # under-deduped result; raise cc_max_iterations for complete dedup (see
        # marin#6798).
        logger.warning(
            "Connected components did not converge within cc_max_iterations=%d; dedup is deterministic but "
            "incomplete (some near-dup clusters remain split). Raise cc_max_iterations for complete dedup.",
            cc_max_iterations,
        )

    ctx.put(_SHARED_ENTRIES_KEY, entries)
    decisions_dir = f"{output_path}/metadata/decisions"
    verification_pipeline = (
        Dataset.from_list(cc_files)
        .load_parquet()
        .map(_candidate_node)
        .filter(lambda record: record["emit"])
        .group_by(
            lambda record: record["component_id"],
            sort_by=lambda record: (record["canonical_order"], record["id_norm"]),
            reducer=_candidate_pairs,
        )
        .flat_map(_candidate_text_requests)
        .group_by(
            lambda request: request["file_idx"],
            sort_by=lambda request: (request["document_id"], request["pair_id"], request["role_order"]),
            reducer=_requested_texts,
        )
        .group_by(
            lambda piece: piece["pair_id"],
            sort_by=lambda piece: piece["role_order"],
            reducer=_make_candidate_verifier(verification),
        )
        .write_parquet(f"{decisions_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    verification_outcome = ctx.execute(verification_pipeline, verbose=True)
    decision_files = sorted(str(path) for path in StoragePath(f"{decisions_dir}/*.parquet").glob())

    marker_pipeline = (
        Dataset.from_list(decision_files)
        .load_parquet()
        .filter(lambda decision: decision["accepted"])
        .flat_map(_marker_records)
        .group_by(
            lambda marker: marker["marker_file_idx"],
            sort_by=lambda marker: marker["marker_id"],
            reducer=_make_marker_writer(output_path),
        )
    )
    marker_outcome = ctx.execute(marker_pipeline, verbose=True)
    empty_outcome = ctx.execute(
        Dataset.from_list(entry_groups).flat_map(_make_empty_marker_writer(output_path)),
        verbose=True,
    )

    sources: dict[str, FuzzyDupsPerSource] = {
        src_dir: FuzzyDupsPerSource(attr_dir=f"{output_path}/outputs/{tag}") for src_dir, tag in source_tag.items()
    }
    combined_counters: Counter[str] = Counter()
    for outcome in (verification_outcome, marker_outcome, empty_outcome):
        combined_counters.update(outcome.counters)
    candidates = int(combined_counters[f"{_VERIFICATION_COUNTER}/candidates"])
    verified = int(combined_counters[f"{_VERIFICATION_COUNTER}/decision/accepted"])
    logger.info(
        "Fuzzy dups: verified %d/%d direct-canonical candidates; retained %d rejected candidates",
        verified,
        candidates,
        candidates - verified,
    )

    return FuzzyDupsAttrData(
        params=params,
        verification=verification,
        decisions_dir=decisions_dir,
        sources=sources,
        counters=dict(combined_counters),
    )


def compute_fuzzy_dups_attrs_step(
    *,
    name: str,
    minhash_steps: list[StepSpec],
    verification_params: FuzzyVerificationParams | None = None,
    cc_max_iterations: int = 10,
    max_parallelism: int,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that computes fuzzy duplicate attrs from ``MinHashAttrData`` step outputs."""
    verification = verification_params or FuzzyVerificationParams()
    return StepSpec(
        name=name,
        deps=list(minhash_steps),
        fn=lambda output_path: compute_fuzzy_dups_attrs(
            inputs=[read_artifact(s.output_path, MinHashAttrData) for s in minhash_steps],
            output_path=output_path,
            verification_params=verification,
            cc_max_iterations=cc_max_iterations,
            max_parallelism=max_parallelism,
            worker_resources=worker_resources,
            coordinator_resources=coordinator_resources,
        ),
        hash_attrs={
            "candidate_scope": FUZZY_DUPS_CANDIDATE_SCOPE,
            "cc_max_iterations": cc_max_iterations,
            "verification": verification.model_dump(),
        },
        override_output_path=override_output_path,
    )
