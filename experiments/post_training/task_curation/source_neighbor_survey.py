# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remeasure an immutable overlap survey from its complete pinned TaskSpec inputs."""

import argparse
import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.source_fingerprints import (
    CURRENT_OVERLAP_VERSION,
    DirectionalOverlap,
    PairSimilarity,
    SourceFingerprint,
    SourceIdentitySurvey,
)

from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey, identity_survey_payload
from experiments.post_training.task_curation.source_fingerprints import duplicate_groups, pair_similarity
from experiments.post_training.task_curation.source_population_cache import (
    PopulationLocations,
    cache_population,
    population_preflight,
    read_hash_population,
    read_locator_manifest,
    read_population,
)
from experiments.post_training.task_curation.task_neighbors import (
    SAMPLE_SIZE,
    SIMILARITY_THRESHOLD,
    directional_overlap,
    exact_reference_integrity,
    jaccard_similarity,
    lsh_band_keys,
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
    for source_id in sorted(paths, key=original_source_id):
        path = paths[source_id]
        population = read_population(path, fingerprints[source_id].inputs)
        panel = population.sample(256)
        del population
        for row, signature in zip(panel, signatures(panel), strict=True):
            candidates = set()
            for key in lsh_band_keys(signature):
                candidates.update(buckets[key])
            query = shingles(row.text)
            for index in candidates:
                other_id, other = rows[index]
                if source_id == other_id:
                    continue
                left, right = sorted((source_id, other_id), key=original_source_id)
                if (left, right) in selected:
                    continue
                reference = shingles(other.text)
                similarity = jaccard_similarity(query, reference)
                if similarity >= SIMILARITY_THRESHOLD:
                    selected[left, right] = pair_similarity(left, fingerprints[left], right, fingerprints[right])
            index = len(rows)
            rows.append((source_id, row))
            for key in lsh_band_keys(signature):
                buckets[key].append(index)
    logger.info("Content-NN discovery widened %d pairs to %d", len(pairs), len(selected))
    return list(selected.values())


def measure_target(
    target_id: str,
    partners: list[str],
    paths: dict[str, Path],
    inputs: dict[str, dict[str, str]],
    sample_size: int,
    output_root: str,
    reuse_neighbor_root: str | None,
) -> tuple[dict[tuple[str, str], DirectionalOverlap], list[dict]]:
    """Measure one target shard, reusing its index and immutable prior receipts."""
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    output = StoragePath(output_root)
    target = read_hash_population(paths[target_id], inputs[target_id])
    directions = {}
    integrity = []
    for source_id in partners:
        direction_key = canonical_sha256(
            {
                "source": original_source_id(source_id),
                "target": original_source_id(target_id),
                "sample_size": sample_size,
                "source_inputs": inputs[source_id],
                "target_inputs": inputs[target_id],
                "method": CURRENT_OVERLAP_VERSION,
            }
        )
        receipt_uri = str(output / "neighbors" / f"{direction_key}.json")
        candidates = [receipt_uri]
        if reuse_neighbor_root is not None:
            candidates.append(str(StoragePath(reuse_neighbor_root) / "neighbors" / f"{direction_key}.json"))
        prior_uri = next((uri for uri in candidates if StoragePath(uri).exists()), None)
        if prior_uri is not None:
            direction = DirectionalOverlap.model_validate_json(StoragePath(prior_uri).read_bytes())
            if direction.neighbors_sha256 != canonical_sha256(
                [row.model_dump(mode="json") for row in direction.neighbors]
            ):
                raise ValueError(f"Incomplete neighbor receipt: {prior_uri}")
            write_json(receipt_uri, direction.model_dump(mode="json"))
        else:
            source = read_hash_population(paths[source_id], inputs[source_id])
            direction = directional_overlap(source, target, sample_size)
            write_json(receipt_uri, direction.model_dump(mode="json"))
            del source
        directions[source_id, target_id] = direction.model_copy(
            update={"neighbors": direction.neighbors[:20], "neighbor_receipts_uri": receipt_uri}
        )
        logger.info(
            "NN %s -> %s: %.4f, exact %.4f%s",
            original_source_id(source_id),
            original_source_id(target_id),
            direction.fraction,
            direction.exact_fraction,
            " (reused)" if prior_uri else "",
        )
        if "openq" in original_source_id(source_id) and "openq" in original_source_id(target_id):
            source = read_hash_population(paths[source_id], inputs[source_id])
            result = {
                "source": original_source_id(source_id),
                "target": original_source_id(target_id),
                **exact_reference_integrity(source, target),
            }
            integrity.append(result)
            write_json(str(output / "integrity" / f"{direction_key}.json"), result)
            logger.warning("OPENQA INTEGRITY %s", json.dumps(result))
            del source
    return directions, integrity


def rerun(
    survey: SourceIdentitySurvey,
    output_root: str,
    work_dir: Path,
    workers: int,
    sample_size: int,
    only_openqa: bool,
    allocated_cpus: int,
    locations: PopulationLocations,
    reuse_neighbor_root: str | None = None,
) -> dict:
    output = StoragePath(output_root)
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
    population_preflight(survey, locations, "marin-us-east-02a")
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        pending = {
            executor.submit(
                cache_population,
                source,
                fingerprints[source],
                work_dir,
                locations,
                {},
                source_bucket="marin-us-east-02a",
            ): source
            for source in sources
        }
        paths = {}
        for future in as_completed(pending):
            source = pending[future]
            paths[source] = future.result()
            logger.info("Prepared %d/%d populations: %s", len(paths), len(sources), original_source_id(source))
    if not only_openqa:
        pairs = discover_pairs(paths, fingerprints, pairs)
    directions: dict[tuple[str, str], DirectionalOverlap] = {}
    integrity = []
    inputs = {source: fingerprint.inputs for source, fingerprint in fingerprints.items()}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        pending = {}
        for target_id in sources:
            partners = sorted(
                {
                    pair.left if pair.right == target_id else pair.right
                    for pair in pairs
                    if target_id in (pair.left, pair.right)
                },
                key=original_source_id,
            )
            if partners:
                future = executor.submit(
                    measure_target, target_id, partners, paths, inputs, sample_size, output_root, reuse_neighbor_root
                )
                pending[future] = target_id
        for future in as_completed(pending):
            measured, checked = future.result()
            directions.update(measured)
            integrity.extend(checked)
            logger.info("Measured %d/%d directions", len(directions), 2 * len(pairs))
    integrity.sort(key=lambda row: (row["source"], row["target"]))
    prior = {(row.similarity.left, row.similarity.right): row for row in survey.reviews}
    reviews, diff = [], []
    for pair in pairs:
        review = measured_review(
            pair,
            directions[pair.left, pair.right],
            directions[pair.right, pair.left],
            sample_size,
            canonical_sha256(fingerprints[pair.left].inputs),
            canonical_sha256(fingerprints[pair.right].inputs),
        )
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
    uri = str(output / f"source-identities-{CURRENT_OVERLAP_VERSION}.json")
    sha = write_json(uri, identity_survey_payload(result))
    diff_sha = write_json(str(output / "label-diff.json"), {"pairs": diff})
    integrity_sha = write_json(
        str(output / "openqa-integrity.json"),
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
        "label_diff_uri": str(output / "label-diff.json"),
        "label_diff_sha256": diff_sha,
        "integrity_uri": str(output / "openqa-integrity.json"),
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
    write_json(str(output / "result.json"), summary)
    logger.info("RESULT %s", json.dumps(summary))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", required=True)
    parser.add_argument("--survey-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--population-locators",
        type=Path,
        default=Path("experiments/post_training/task_curation/source_population_locators.json"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/content-nn-v2"))
    parser.add_argument("--allocated-cpus", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-neighbor-root")
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
        read_locator_manifest(args.population_locators),
        args.reuse_neighbor_root,
    )


if __name__ == "__main__":
    main()
