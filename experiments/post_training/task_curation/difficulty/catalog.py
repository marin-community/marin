# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve frozen task panels against the current served source declarations."""

import json
from collections.abc import Sequence
from dataclasses import replace

from finestore.rl.curation import SourceUse
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from rigging.auth import TokenProvider
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.campaign import campaign_record
from experiments.post_training.task_curation.difficulty.importer import catalog_sources, declaration_population_revision
from experiments.post_training.task_curation.difficulty.ledger import DEFAULT_LEDGER, LEDGER_VERSION, PANELS_DIRECTORY
from experiments.post_training.task_curation.difficulty.state import CatalogSource, CellState, GapReason, PanelReference
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

DECLARATION_BATCH_SIZE = 4
DEFAULT_APPLET = "c9742a15-fb46-4d56-bc88-aa916351ebd3"
# Current-schema metadata for every retained panel matching the served population.
# The migration audit beside sources.json verifies unchanged panel/sample/release bindings.
DEFAULT_QUEUES = (
    (
        "s3://marin-us-east-08a/marin/rl-data/atlas/difficulty-engine-pools/20261123-phase2/"
        "queues/current-schema-20261124/"
        "fa94574a3e7e50055993631ccd148c46033d0b552c58cb08281f1f89620a5741/sources.json",
        "fa94574a3e7e50055993631ccd148c46033d0b552c58cb08281f1f89620a5741",
    ),
)


def exclude_catalog_sources(catalog: Sequence[CatalogSource], excluded_sources: frozenset[str]) -> list[CatalogSource]:
    """Apply explicit source quarantines and reject unknown source IDs."""
    missing = excluded_sources - {source.source_id for source in catalog}
    if missing:
        raise ValueError(f"Excluded sources are absent from the served catalog: {sorted(missing)}")
    return [source for source in catalog if source.source_id not in excluded_sources]


def available_artifact_uris(cells: Sequence[CellState]) -> frozenset[str]:
    artifacts = {
        gap.artifact_uri
        for cell in cells
        for gap in cell.gaps
        if gap.reason == GapReason.ENVIRONMENT_ARTIFACT_MISSING and gap.artifact_uri
    }
    return frozenset(uri for uri in artifacts if StoragePath(uri).exists())


def published_panel_sources(ledger: str) -> list[CatalogSource]:
    sources = []
    for index in sorted((StoragePath(ledger) / LEDGER_VERSION / PANELS_DIRECTORY / "*.json").glob(), key=str):
        reference = PanelReference(**json.loads(index.read_text()))
        (source,) = catalog_sources(reference.queue_uri, reference.queue_sha256)
        if (source.source_id, source.registry_revision) != (reference.source_id, reference.registry_revision):
            raise ValueError(f"Published panel differs from its population binding: {index}")
        sources.append(source)
    return sources


def declarations_for_populations(query: AppletQuery, missing: Sequence[dict]) -> dict[str, dict]:
    """Read complete declarations in bounded batches from their original catalog load."""
    full_declarations = {}
    for start in range(0, len(missing), DECLARATION_BATCH_SIZE):
        batch = missing[start : start + DECLARATION_BATCH_SIZE]
        parameters = {"load_id": batch[0]["load_id"], **{f"source_{i}": row["id"] for i, row in enumerate(batch)}}
        placeholders = ",".join(f":source_{i}" for i in range(len(batch)))
        full = query(
            "SELECT id, declaration FROM atlas_sources WHERE load_id=:load_id AND id IN ("
            + placeholders
            + ") ORDER BY id",
            parameters,
        )["rows"]
        if {row["id"] for row in full} != {row["id"] for row in batch}:
            raise ValueError("Frozen catalog snapshot lacks a requested source declaration")
        full_declarations.update({row["id"]: row["declaration"] for row in full})
    return full_declarations


def served_catalog(
    applet_id: str,
    queues: tuple[tuple[str, str], ...] = DEFAULT_QUEUES,
    token_provider: TokenProvider | None = None,
    ledger: str = DEFAULT_LEDGER,
) -> list[CatalogSource]:
    """Keep every served source; unmatched pins need a fresh frozen task panel."""
    query = applet_query(DEFAULT_MARINA_URL, applet_id, token_provider=token_provider)
    campaign = campaign_record(ledger)
    if campaign is not None:
        queues = tuple(tuple(item) for item in campaign["queues"])
    declarations = query(
        "SELECT id, load_id, declaration->>'intended_use' AS intended_use, "
        "jsonb_build_object('current_release', declaration->'current_release', "
        "'pipeline', declaration->'pipeline', 'actor_census_input', declaration->'actor_census_input') "
        "AS declaration FROM atlas_sources WHERE load_id = "
        "(SELECT load_id FROM atlas_current WHERE name = 'latest') ORDER BY id",
        {},
    )["rows"]
    panels: dict[tuple[str, str], CatalogSource] = {}
    for uri, expected in queues:
        for source in catalog_sources(uri, expected):
            panels[(canonical_source_id(source.source_id), source.registry_revision)] = source
    for source in published_panel_sources(ledger):
        panels.setdefault((canonical_source_id(source.source_id), source.registry_revision), source)
    missing = [
        row
        for row in declarations
        if (canonical_source_id(row["id"]), declaration_population_revision(row["declaration"])) not in panels
    ]
    full_declarations = declarations_for_populations(query, missing)
    sources = []
    for row in declarations:
        declaration = row["declaration"]
        revision = declaration_population_revision(declaration)
        match = panels.get((canonical_source_id(row["id"]), revision))
        if match is not None:
            sources.append(
                replace(match, intended_use=SourceUse(row["intended_use"]) if row.get("intended_use") else None)
            )
            continue
        full = full_declarations[row["id"]]
        if any(full[key] != declaration[key] for key in declaration):
            raise ValueError("Source declaration differs from its frozen population identity")
        sources.append(
            CatalogSource(
                row["id"],
                revision,
                (),
                "",
                "",
                json.dumps(full, sort_keys=True),
                intended_use=SourceUse(row["intended_use"]) if row.get("intended_use") else None,
            )
        )
    if len({canonical_source_id(source.source_id) for source in sources}) != len(sources):
        raise ValueError("Served catalog contains duplicate source identities")
    return sources
