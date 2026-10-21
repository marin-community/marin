# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select an admitted Snowball v0 pool with an explicit source-extension recipe."""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import partial

from finestore.rl.curation import CurationCatalog
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256

from experiments.post_training.snowball_curriculum.bundle import write_manifest
from experiments.post_training.snowball_curriculum.selection import (
    Population,
    SourceSpec,
    allocate_quotas,
    load_sources,
    read_candidates,
    select_candidates,
    source_weights,
    write_selection,
)

BEN_SOURCE_FAMILIES = {
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_calendar_v2": "calendar",
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_comp_coding": "competitive_coding",
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_reasoning_gym": "reasoning_gym",
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa": "stem",
}
BEN_FAMILY_WEIGHTS = {
    "calendar": 1 / 6,
    "competitive_coding": 1 / 6,
    "reasoning_gym": 1 / 6,
    "stem": 1 / 6,
    "structural": 1 / 3,
}


def migrated_final_paths(final: str, receipt: dict) -> tuple[str, ...]:
    """Resolve admitted files through a verified migration without reading the old store."""
    final_path = StoragePath(final)
    matches = []
    for prefix in receipt["prefixes"]:
        if not prefix["ready"]:
            continue
        try:
            final_path.relative_to(StoragePath(prefix["source"]))
        except ValueError:
            continue
        matches.append(prefix)
    if not matches:
        raise ValueError(f"No completed migration covers admitted final data: {final}")
    migration = max(matches, key=lambda prefix: len(StoragePath(prefix["source"]).segments))
    raw_manifest = StoragePath(migration["url"]).read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != migration["sha256"]:
        raise ValueError(f"Migration manifest changed: {migration['url']}")
    manifest = json.loads(raw_manifest)
    if manifest["source"] != migration["source"] or manifest["destination"] != migration["destination"]:
        raise ValueError(f"Migration manifest roots differ from the receipt: {migration['url']}")
    suffix = final_path.relative_to(StoragePath(migration["source"]))
    destination = StoragePath(migration["destination"])
    final_destination = destination / suffix
    selected = []
    for file in manifest["files"]:
        path = destination / file["path"]
        try:
            relative = StoragePath(path.relative_to(final_destination))
        except ValueError:
            continue
        if len(relative.segments) == 1 and relative.name.startswith("part-") and relative.name.endswith(".parquet"):
            selected.append(str(path))
    paths = tuple(sorted(selected))
    live_paths = tuple(sorted(str(path) for path in (final_destination / "part-*.parquet").glob()))
    if not paths or paths != live_paths:
        raise ValueError(f"Migrated admitted files differ from the verified manifest: {final}")
    return paths


def catalog_sources(
    catalog_uri: str,
    snapshot_id: str | None = None,
    *,
    source_ids: set[str] | None = None,
    migration_receipt: str | None = None,
) -> tuple[str, tuple[SourceSpec, ...]]:
    """Resolve a production catalog snapshot to its admitted final files."""
    configure_coreweave_s3()
    catalog = CurationCatalog(catalog_uri)
    snapshot_id = snapshot_id or catalog.current().snapshot_id
    receipt = json.loads(StoragePath(migration_receipt).read_bytes()) if migration_receipt else None
    sources = []
    for source in catalog.sources(snapshot_id):
        if source_ids is not None and source.id not in source_ids:
            continue
        if source.assessment_key is None:
            continue
        assessment = catalog.assessment(source.assessment_key)
        manifest = catalog.document(assessment.curation_uri)
        final = manifest["datasets"]["final"]
        paths = (
            migrated_final_paths(final, receipt)
            if receipt is not None
            else tuple(sorted(str(path) for path in (StoragePath(final) / "part-*.parquet").glob()))
        )
        if paths:
            sources.append(
                SourceSpec(
                    source.id,
                    manifest["intended_use"],
                    paths,
                    "admitted",
                    manifest_sha256=canonical_sha256(manifest),
                    catalog_snapshot_id=snapshot_id,
                    assessment_curation_uri=assessment.curation_uri,
                )
            )
    return snapshot_id, tuple(sorted(sources, key=lambda source: source.source_id))


def family_source_weights(
    capacities: Mapping[str, int], families: Mapping[str, str], family_weights: Mapping[str, float], target: int
) -> dict[str, int]:
    """Keep family weights while redistributing scarce capacity within the recipe."""
    family_capacities: Counter[str] = Counter()
    for source, count in capacities.items():
        family_capacities[families[source]] += count
    family_quotas = allocate_quotas(family_capacities, target, family_weights)
    result = {}
    for family, quota in family_quotas.items():
        counts = {source: count for source, count in capacities.items() if families[source] == family}
        result.update(allocate_quotas(counts, quota, source_weights(counts, equal_fraction=0)))
    return result


def run(
    *,
    sources: Sequence[SourceSpec],
    recipe: dict,
    output: str,
    target: int = 100_000,
    seed: int = 0,
    holdout_sources: Sequence[SourceSpec] = (),
) -> None:
    """Freeze an admitted-only training selection and its exact allocation policy."""
    families = dict(BEN_SOURCE_FAMILIES)
    families.update(recipe.get("extensions", {}))
    family_weights = recipe.get("family_weights", BEN_FAMILY_WEIGHTS)
    difficulty = recipe.get("difficulty_by_source", {})
    chosen = tuple(source for source in sources if source.source_id in families and source.intended_use == "train")
    if any(source.population != "admitted" for source in chosen):
        raise ValueError("v0 training requires admitted TaskSpecs; converted data cannot substitute for admission")
    if not chosen:
        raise ValueError("No admitted sources match the explicit v0 recipe")
    # Retain all published eval sources even when they are outside the training allowlist.
    evaluation = tuple(source for source in sources if source.intended_use != "train") + tuple(holdout_sources)
    if any(source.intended_use == "train" for source in holdout_sources):
        raise ValueError("Explicit holdout sources must declare a non-training intended_use")
    factory = partial(read_candidates, chosen + evaluation, difficulty_by_source=difficulty)
    # First census determines weights after heldout removal and cross-source deduplication.
    exclusions = {
        "heldout_task_ids": recipe.get("heldout_task_ids", ()),
        "heldout_context_ids": recipe.get("heldout_context_ids", ()),
    }
    census = select_candidates(factory, target=0, seed=seed, **exclusions)
    weights = family_source_weights(census.eligible_counts, families, family_weights, target)
    result = select_candidates(factory, target=target, seed=seed, weights=weights, **exclusions)
    write_selection(result, chosen + evaluation, output, seed=seed, target=target)
    (StoragePath(output) / "recipe.json").write_text(
        json.dumps(
            {**recipe, "resolved_source_families": families, "resolved_family_weights": family_weights},
            indent=2,
            sort_keys=True,
        )
    )
    write_manifest(StoragePath(output))
    print(json.dumps({"selected": len(result.selected), "shortfall": result.shortfall, "output": output}))


def main() -> None:
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sources", help="Frozen JSON source manifest with admitted task_paths")
    inputs.add_argument("--catalog", help="Pinned production catalog URI")
    parser.add_argument("--snapshot-id", help="Use a particular catalog snapshot instead of current")
    parser.add_argument("--migration-receipt", help="Verified migration receipt mapping archived data roots")
    parser.add_argument(
        "--holdout-sources", required=True, help="Frozen eval/holdout manifest; must cover known heldouts"
    )
    parser.add_argument("--recipe", required=True, help="JSON: extensions, family_weights, difficulty_by_source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    recipe = json.loads(StoragePath(args.recipe).read_bytes())
    snapshot_id = None
    if args.catalog:
        source_ids = set(BEN_SOURCE_FAMILIES) | set(recipe.get("extensions", {}))
        snapshot_id, sources = catalog_sources(
            args.catalog, args.snapshot_id, source_ids=source_ids, migration_receipt=args.migration_receipt
        )
    else:
        sources = load_sources(args.sources, population="admitted")
        source_snapshots = {
            source.catalog_snapshot_id
            for source in sources
            if source.population == "admitted" and source.catalog_snapshot_id is not None
        }
        if len(source_snapshots) > 1:
            raise ValueError(f"Admitted sources reference multiple catalog snapshots: {sorted(source_snapshots)}")
        provenance_snapshots = source_snapshots | {
            value for value in (args.snapshot_id, recipe.get("catalog_snapshot_id")) if value is not None
        }
        if len(provenance_snapshots) > 1:
            raise ValueError(f"Frozen recipe and admitted source snapshots differ: {sorted(provenance_snapshots)}")
        snapshot_id = next(iter(provenance_snapshots), None)
    holdout_document = json.loads(StoragePath(args.holdout_sources).read_bytes())
    population: Population = holdout_document.get("population", "converted")
    holdouts = load_sources(args.holdout_sources, population=population)
    recipe["holdout_manifest"] = args.holdout_sources
    for key, value in (
        ("catalog_uri", args.catalog),
        ("catalog_snapshot_id", snapshot_id),
        ("migration_receipt", args.migration_receipt),
    ):
        if value is not None:
            recipe[key] = value
    run(sources=sources, recipe=recipe, output=args.output, target=args.target, seed=args.seed, holdout_sources=holdouts)


if __name__ == "__main__":
    main()
