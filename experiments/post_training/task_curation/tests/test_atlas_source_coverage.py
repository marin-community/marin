# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Main declares every Atlas-admitted source at the Atlas pin.

The fixture pins one Atlas cut (origin/atlas-env-audit revision and the R40 admit
set); its ``atlas_revision`` and ``r40_plan`` fields record where each value came
from. The fixture's bytes live in the pinned public Hugging Face dataset
``open-athena/atlas-source-pins`` and this module commits only the immutable
reference (revision, sha256, byte size), so refreshing the cut is one upload plus
one constant bump. Regenerate and republish with
``tests/generate_atlas_source_pins.py``.

Three checks, all computed from the catalog:

- coverage: every Atlas-admitted source is declared on main, or carries a
  ``declaration_blockers`` entry — the import that would run it on main is still
  pending, and per the #11903 review resolution a record lands together with its
  import, never pipeline-less;
- stale blockers: a source whose blocker entry still stands while main declares
  it — the ingestion PR that landed the import must remove the entry;
- pin parity: every declared source's durable reference (repo, revision, files,
  recipe, version) equals the Atlas admitted pin, unless a recorded waiver
  states the blocker; parity that has landed makes the waiver stale.

The tests fetch the pinned fixture, so they carry the ``data_integration``
marker and run from the CI workflows that have HF access.
"""

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest
from finestore.rl.source_identity import canonical_source_id
from huggingface_hub import hf_hub_download

from experiments.post_training.task_curation.sources import all_sources

ATLAS_SOURCE_PINS_REPO = "open-athena/atlas-source-pins"
ATLAS_SOURCE_PINS_REVISION = "8b3736584f773f6c83caa4d177bbbb29eed73f08"
ATLAS_SOURCE_PINS_SHA256 = "166f869834973f58359be5b8f7a44924b87fe6d43f6a31558b236ec947596497"
ATLAS_SOURCE_PINS_BYTES = 177059


@dataclass(frozen=True)
class AtlasSourcePinsReference:
    """The immutable fixture reference this test gates against."""

    repo: str
    revision: str
    sha256: str
    byte_size: int

    @property
    def uri(self) -> str:
        return f"hf://datasets/{self.repo}@{self.revision}/atlas_source_pins.json"


ATLAS_SOURCE_PINS_REFERENCE = AtlasSourcePinsReference(
    repo=ATLAS_SOURCE_PINS_REPO,
    revision=ATLAS_SOURCE_PINS_REVISION,
    sha256=ATLAS_SOURCE_PINS_SHA256,
    byte_size=ATLAS_SOURCE_PINS_BYTES,
)


@cache
def read_atlas_source_pins() -> dict:
    """Download the pinned fixture through the cached HF login and verify its bytes."""
    reference = ATLAS_SOURCE_PINS_REFERENCE
    local = Path(
        hf_hub_download(
            repo_id=reference.repo, repo_type="dataset", revision=reference.revision, filename="atlas_source_pins.json"
        )
    )
    raw = local.read_bytes()
    if len(raw) != reference.byte_size or hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError("Atlas source-pins fixture differs from its pinned reference")
    return json.loads(raw)


def _durable_ref(source) -> dict:
    dataset = source.dataset
    pipeline = source.pipeline
    return {
        "dataset_repo": dataset.name if dataset else None,
        "dataset_revision": dataset.revision if dataset else None,
        "pipeline_files": list(source.files) if pipeline is not None else [],
        "pipeline_name": source.name if pipeline is not None else None,
        "pipeline_version": source.version if pipeline is not None else None,
    }


@pytest.mark.data_integration
def test_every_atlas_admitted_source_is_declared_on_main_or_recorded_as_blocked():
    """An Atlas-admitted source missing from the catalog fails unless its blocker stands."""
    fixture = read_atlas_source_pins()
    blockers = fixture["declaration_blockers"]
    sources = {canonical_source_id(source.info.id) for source in all_sources().values()}
    missing = sorted(
        source_id
        for source_id in fixture["required_ids"]
        if canonical_source_id(source_id) not in sources and source_id not in blockers
    )
    assert missing == []


@pytest.mark.data_integration
def test_a_declaration_blocker_expires_when_main_declares_the_source():
    """The ingestion PR that lands the import removes the source's blocker entry."""
    fixture = read_atlas_source_pins()
    blockers = fixture["declaration_blockers"]
    sources = {canonical_source_id(source.info.id): source for source in all_sources().values()}
    stale = sorted(source_id for source_id in blockers if canonical_source_id(source_id) in sources)
    assert stale == []


@pytest.mark.data_integration
def test_main_pins_match_the_atlas_admitted_pins():
    """A declared pin differs from the Atlas admitted pin only under a recorded waiver."""
    fixture = read_atlas_source_pins()
    waivers = fixture["pin_parity_waivers"]
    blockers = fixture["declaration_blockers"]
    unknown = sorted(set(waivers) - set(fixture["required_ids"]) - set(blockers))
    assert unknown == []

    sources = {canonical_source_id(source.info.id): source for source in all_sources().values()}
    stale_waivers, unwaived = [], []
    for source_id, atlas_ref in fixture["pins"].items():
        key = canonical_source_id(source_id)
        source = sources.get(key)
        if source is None or key in blockers:
            continue
        current = _durable_ref(source)
        if key in waivers:
            if current == atlas_ref:
                stale_waivers.append(source_id)
        elif current != atlas_ref:
            unwaived.append(source_id)

    assert stale_waivers == [], f"parity landed; drop the waivers: {stale_waivers}"
    assert unwaived == [], f"pins differ from the Atlas admitted pins with no waiver: {unwaived}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
