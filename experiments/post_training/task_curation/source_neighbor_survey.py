# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remeasure an immutable overlap survey from its complete pinned TaskSpec inputs."""

import argparse
import hashlib
import json
import logging
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.source_fingerprints import (
    DirectionalOverlap,
    PairSimilarity,
    SourceFingerprint,
    SourceIdentitySurvey,
)

from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey, identity_survey_payload
from experiments.post_training.task_curation.source_fingerprints import duplicate_groups, pair_similarity
from experiments.post_training.task_curation.task_neighbors import (
    BAND_ROWS,
    PERMUTATIONS,
    SAMPLE_SIZE,
    SIMILARITY_THRESHOLD,
    PopulationIndex,
    TaskContent,
    content_rows,
    directional_overlap,
    exact_reference_integrity,
    measured_review,
    shingles,
    signatures,
)
from infra.marina.apps.rl_data_catalog.server.source_identity import original_source_id

logger = logging.getLogger(__name__)


def write_json(uri: str, document: dict) -> str:
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = StoragePath(uri)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Immutable output already differs: {uri}")
    else:
        with path.open("wb", auto_mkdir=True) as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def cache_population(source_id: str, fingerprint: SourceFingerprint, cache_dir: Path) -> Path:
    """Verify input bytes before extracting complete task content into a local cache."""
    path = cache_dir / f"{canonical_sha256({'source': source_id, 'inputs': fingerprint.inputs})}.parquet"
    if path.exists():
        return path
    configure_coreweave_s3()
    schema = pa.schema(
        [
            ("task_id", pa.string()),
            ("instruction_hash", pa.string()),
            ("text", pa.large_string()),
            ("reference_hash", pa.string()),
            ("tests_hash", pa.string()),
        ]
    )
    count = 0
    with pq.ParquetWriter(path.with_suffix(".pending"), schema, compression="zstd") as writer:
        for uri, digest in sorted(fingerprint.inputs.items()):
            if not uri.startswith("s3://marin-us-east-02a/"):
                raise ValueError(f"Rerun requires co-located us-east-02a input: {uri}")
            fs, remote_path = filesystem_for(uri)
            with tempfile.TemporaryFile() as local:
                with fs.open(remote_path, "rb") as remote:
                    sha = hashlib.sha256()
                    while chunk := remote.read(8 * 1024 * 1024):
                        sha.update(chunk)
                        local.write(chunk)
                if sha.hexdigest() != digest:
                    raise ValueError(f"Input digest differs: {uri}")
                local.seek(0)
                parquet = pq.ParquetFile(local)
                column = "lowered_task_spec" if "lowered_task_spec" in parquet.schema_arrow.names else "task_spec"
                for batch in parquet.iter_batches(batch_size=256, columns=[column]):
                    rows = [
                        asdict(row)
                        for row in content_rows(json.loads(encoded) for encoded in batch.column(column).to_pylist())
                    ]
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    count += len(rows)
    if count != fingerprint.task_count:
        raise ValueError(f"Population count differs for {source_id}: {count} != {fingerprint.task_count}")
    path.with_suffix(".pending").rename(path)
    logger.info("Cached %s: %d tasks", original_source_id(source_id), count)
    return path


def read_population(path: Path) -> PopulationIndex:
    return PopulationIndex([TaskContent(**row) for row in pq.read_table(path).to_pylist()])


def widened_pairs(survey: SourceIdentitySurvey) -> list[PairSimilarity]:
    """Keep every prior shortlist pair and add instruction-hash evidence without top-k truncation."""
    selected = {(row.similarity.left, row.similarity.right): row.similarity for row in survey.reviews}
    fingerprints = {key: value for key, value in survey.fingerprints.items() if value is not None}
    sources = sorted(fingerprints, key=original_source_id)
    for index, left in enumerate(sources):
        for right in sources[index + 1 :]:
            pair = pair_similarity(left, fingerprints[left], right, fingerprints[right])
            if pair.instruction_overlap >= 0.01:
                selected.setdefault((left, right), pair)
    return list(selected.values())


def discover_pairs(
    paths: dict[str, Path], fingerprints: dict[str, SourceFingerprint], pairs: list[PairSimilarity]
) -> list[PairSimilarity]:
    """Widen the shortlist using cross-source content neighbors in fixed-seed panels."""
    selected = {(pair.left, pair.right): pair for pair in pairs}
    buckets = defaultdict(list)
    rows = []
    for source_id, path in paths.items():
        population = read_population(path)
        panel = population.sample(256)
        del population
        for row, signature in zip(panel, signatures(panel), strict=True):
            candidates = set()
            for start in range(0, PERMUTATIONS, BAND_ROWS):
                candidates.update(buckets[start, tuple(signature[start : start + BAND_ROWS])])
            query = shingles(row.text)
            for index in candidates:
                other_id, other = rows[index]
                if source_id == other_id:
                    continue
                left, right = sorted((source_id, other_id), key=original_source_id)
                if (left, right) in selected:
                    continue
                reference = shingles(other.text)
                union = query | reference
                similarity = len(query & reference) / len(union) if union else 0
                if similarity >= SIMILARITY_THRESHOLD:
                    selected[left, right] = pair_similarity(left, fingerprints[left], right, fingerprints[right])
            index = len(rows)
            rows.append((source_id, row))
            for start in range(0, PERMUTATIONS, BAND_ROWS):
                buckets[start, tuple(signature[start : start + BAND_ROWS])].append(index)
    logger.info("Content-NN discovery widened %d pairs to %d", len(pairs), len(selected))
    return list(selected.values())


def rerun(
    survey: SourceIdentitySurvey,
    output_root: str,
    work_dir: Path,
    workers: int,
    sample_size: int,
    only_openqa: bool,
    allocated_cpus: int,
) -> dict:
    started = time.monotonic()
    fingerprints = {key: value for key, value in survey.fingerprints.items() if value is not None}
    pairs = widened_pairs(survey)
    if only_openqa:
        pairs = [
            pair
            for pair in pairs
            if pair.instruction_overlap >= 0.99
            and all("openq" in original_source_id(key) for key in (pair.left, pair.right))
        ]
    needed = {key for pair in pairs for key in (pair.left, pair.right)} if only_openqa else set(fingerprints)
    work_dir.mkdir(parents=True, exist_ok=True)
    # Cache the integrity-critical populations first; no population content leaves the region.
    sources = sorted(needed, key=lambda key: ("openq" not in original_source_id(key), original_source_id(key)))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        pending = {
            source: executor.submit(cache_population, source, fingerprints[source], work_dir) for source in sources
        }
        paths = {source: future.result() for source, future in pending.items()}
    if not only_openqa:
        pairs = discover_pairs(paths, fingerprints, pairs)
    directions: dict[tuple[str, str], DirectionalOverlap] = {}
    integrity = []
    for target_id in sources:
        target = read_population(paths[target_id])
        partners = sorted(
            {
                pair.left if pair.right == target_id else pair.right
                for pair in pairs
                if target_id in (pair.left, pair.right)
            },
            key=original_source_id,
        )
        for source_id in partners:
            source = read_population(paths[source_id])
            direction_key = canonical_sha256(
                {
                    "source": original_source_id(source_id),
                    "target": original_source_id(target_id),
                    "sample_size": sample_size,
                    "source_inputs": fingerprints[source_id].inputs,
                    "target_inputs": fingerprints[target_id].inputs,
                    "method": "content-nn-v2",
                }
            )
            receipt_uri = f"{output_root}/neighbors/{direction_key}.json"
            if StoragePath(receipt_uri).exists():
                direction = DirectionalOverlap.model_validate_json(StoragePath(receipt_uri).read_bytes())
            else:
                direction = directional_overlap(source, target, sample_size)
                write_json(receipt_uri, direction.model_dump(mode="json"))
            directions[source_id, target_id] = direction.model_copy(
                update={"neighbors": direction.neighbors[:20], "neighbor_receipts_uri": receipt_uri}
            )
            logger.info(
                "NN %s -> %s: %.4f, exact %.4f",
                original_source_id(source_id),
                original_source_id(target_id),
                direction.fraction,
                direction.exact_fraction,
            )
            if "openq" in original_source_id(source_id) and "openq" in original_source_id(target_id):
                result = {
                    "source": original_source_id(source_id),
                    "target": original_source_id(target_id),
                    **exact_reference_integrity(source, target),
                }
                integrity.append(result)
                write_json(f"{output_root}/integrity/{direction_key}.json", result)
                logger.warning("OPENQA INTEGRITY %s", json.dumps(result))
            del source
        del target
    prior = {(row.similarity.left, row.similarity.right): row for row in survey.reviews}
    reviews, diff = [], []
    for pair in pairs:
        review = measured_review(pair, directions[pair.left, pair.right], directions[pair.right, pair.left], sample_size)
        old = prior.get((pair.left, pair.right))
        # Historical ACKs are request-bound. Keep them in the prior artifact and
        # the diff, never attach an old approval to a newly measured request.
        reviews.append(review)
        diff.append(
            {
                "left": original_source_id(pair.left),
                "right": original_source_id(pair.right),
                "before": old.verdict.label if old else None,
                "after": review.verdict.label,
                "changed": old is None or old.verdict.label != review.verdict.label,
                "prior_prompt_sha256": old.prompt_sha256 if old else None,
                "prior_acknowledgement": old.acknowledgement if old else None,
                "left_to_right": directions[pair.left, pair.right].model_dump(mode="json", exclude={"neighbors"}),
                "right_to_left": directions[pair.right, pair.left].model_dump(mode="json", exclude={"neighbors"}),
            }
        )
    result = survey.model_copy(
        update={
            "generated_at": datetime.now(UTC).isoformat(),
            "reviews": reviews,
            "groups": duplicate_groups(reviews),
            "judge": None,
        }
    )
    uri = f"{output_root}/source-identities-content-nn-v2.json"
    sha = write_json(uri, identity_survey_payload(result))
    diff_sha = write_json(f"{output_root}/label-diff.json", {"pairs": diff})
    integrity_sha = write_json(
        f"{output_root}/openqa-integrity.json",
        {
            "directions": integrity,
            "limitation": (
                "Cross-source exact-question answer agreement; cannot exclude a shared upstream answer defect."
            ),
        },
    )
    elapsed = time.monotonic() - started
    summary = {
        "survey_uri": uri,
        "survey_sha256": sha,
        "label_diff_uri": f"{output_root}/label-diff.json",
        "label_diff_sha256": diff_sha,
        "integrity_uri": f"{output_root}/openqa-integrity.json",
        "integrity_sha256": integrity_sha,
        "source_snapshot": survey.snapshot_id,
        "population": "exact archived fingerprint.inputs; exclusions are not newly applied",
        "pairs": len(pairs),
        "prior_pairs": len(survey.reviews),
        "changed_pairs": sum(row["changed"] for row in diff),
        "labels": dict(Counter(row.verdict.label for row in reviews)),
        "sample_size": sample_size,
        "wall_seconds": elapsed,
        "requested_cpus": allocated_cpus,
        "cpu_hours_upper_bound": allocated_cpus * elapsed / 3600,
        "gpu_hours": 0,
    }
    write_json(f"{output_root}/result.json", summary)
    logger.info("RESULT %s", json.dumps(summary))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", required=True)
    parser.add_argument("--survey-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/content-nn-v2"))
    parser.add_argument("--allocated-cpus", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--only-openqa", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    payload = StoragePath(args.survey).read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.survey_sha256:
        raise ValueError("Survey input SHA-256 differs")
    rerun(
        canonical_identity_survey(json.loads(payload)),
        args.output_root,
        args.work_dir,
        args.workers,
        args.sample_size,
        args.only_openqa,
        args.allocated_cpus,
    )


if __name__ == "__main__":
    main()
