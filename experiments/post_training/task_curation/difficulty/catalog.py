# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve frozen task panels against the current served source declarations."""

import json
from collections.abc import Sequence

from marina.client import DEFAULT_MARINA_URL, applet_query
from rigging.auth import TokenProvider
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.importer import catalog_sources, declaration_population_revision
from experiments.post_training.task_curation.difficulty.ledger import DEFAULT_LEDGER, LEDGER_VERSION, PANELS_DIRECTORY
from experiments.post_training.task_curation.difficulty.state import CatalogSource, CellState, GapReason, PanelReference
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

DEFAULT_APPLET = "c9742a15-fb46-4d56-bc88-aa916351ebd3"
DEFAULT_QUEUES = (
    (
        "s3://marin-rno2a/marin/rl-data/atlas/difficulty/2026.11.13.8/samples/sources.json",
        "9d318f52e75dd0e76acaf450f3b824e236e9666b12458bd8dc517335fbfc5695",
    ),
    (
        "s3://marin-rno2a/marin/rl-data/atlas/difficulty/2026.11.17.5/samples/sources.json",
        "5b2f937f100b2d73fb2fbc92eb9d14d9e0533dfef46272500a58eb69061e0a01",
    ),
)


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


def served_catalog(
    applet_id: str,
    queues: tuple[tuple[str, str], ...] = DEFAULT_QUEUES,
    token_provider: TokenProvider | None = None,
    ledger: str = DEFAULT_LEDGER,
) -> list[CatalogSource]:
    """Keep every served source; unmatched pins need a fresh frozen task panel."""
    query = applet_query(DEFAULT_MARINA_URL, applet_id, token_provider=token_provider)
    declarations = query(
        "SELECT id, declaration FROM atlas_sources WHERE load_id = "
        "(SELECT load_id FROM atlas_current WHERE name = 'latest') ORDER BY id",
        {},
    )["rows"]
    panels: dict[tuple[str, str], CatalogSource] = {}
    for uri, expected in queues:
        for source in catalog_sources(uri, expected):
            panels[(canonical_source_id(source.source_id), source.registry_revision)] = source
    for source in published_panel_sources(ledger):
        panels.setdefault((canonical_source_id(source.source_id), source.registry_revision), source)
    sources = []
    for row in declarations:
        declaration = row["declaration"]
        revision = declaration_population_revision(declaration)
        match = panels.get((canonical_source_id(row["id"]), revision))
        if match is not None:
            sources.append(match)
            continue
        sources.append(CatalogSource(row["id"], revision, (), "", "", json.dumps(declaration, sort_keys=True)))
    if len({canonical_source_id(source.source_id) for source in sources}) != len(sources):
        raise ValueError("Served catalog contains duplicate source identities")
    return sources
