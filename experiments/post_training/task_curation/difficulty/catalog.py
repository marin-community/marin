# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve frozen task panels against the current served source declarations."""

import json

from marina.client import DEFAULT_MARINA_URL, applet_query
from rigging.auth import TokenProvider
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.difficulty.importer import catalog_sources, population_revision
from experiments.post_training.task_curation.difficulty.state import CatalogSource
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


def historical_catalog(queue: str, sha256: str) -> list[CatalogSource]:
    return catalog_sources(queue, sha256)


def served_catalog(
    applet_id: str, queues: tuple[tuple[str, str], ...] = DEFAULT_QUEUES, token_provider: TokenProvider | None = None
) -> list[CatalogSource]:
    """Keep every served source; unmatched pins need a fresh frozen task panel."""
    query = applet_query(DEFAULT_MARINA_URL, applet_id, token_provider=token_provider)
    declarations = query(
        "SELECT id, jsonb_build_object('current_release', declaration->'current_release', "
        "'pipeline', declaration->'pipeline', 'actor_census_input', declaration->'actor_census_input') "
        "AS declaration FROM atlas_sources WHERE load_id = "
        "(SELECT load_id FROM atlas_current WHERE name = 'latest') ORDER BY id",
        {},
    )["rows"]
    panels: dict[tuple[str, str], CatalogSource] = {}
    for uri, expected in queues:
        path = StoragePath(uri)
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen queue changed: {uri}")
        for row in json.loads(path.read_text()):
            sample = row["sample"]
            identity = canonical_source_id(sample["source_id"])
            source = CatalogSource(
                sample["source_id"],
                population_revision(sample),
                tuple((t["task_id"], t["task_sha256"]) for t in sample["tasks"]),
                uri,
                expected,
            )
            panels[(identity, source.registry_revision)] = source
    sources = []
    for row in declarations:
        declaration = row["declaration"]
        census = declaration.get("actor_census_input")
        pipeline = declaration.get("pipeline")
        sample = {
            "release": declaration.get("current_release"),
            "converter_version": (pipeline or {}).get("version"),
            "converter_revision": (census or {}).get("converter_revision"),
            "task_specs_sha256": (census or {}).get("task_specs_sha256"),
        }
        revision = population_revision(sample)
        match = panels.get((canonical_source_id(row["id"]), revision))
        sources.append(match or CatalogSource(row["id"], revision, (), "", ""))
    if len({canonical_source_id(source.source_id) for source in sources}) != len(sources):
        raise ValueError("Served catalog contains duplicate source identities")
    return sources
