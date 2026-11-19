# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from infra.marina.apps.rl_data_catalog.server.licensing import license_projection
from infra.marina.apps.rl_data_catalog.server.publication_scope import PUBLICATION_DECISIONS


def test_public_display_approval_preserves_use_and_expires_on_repin():
    decision = next(row for row in PUBLICATION_DECISIONS if row.license_id == "cc-by-4.0")
    declaration = {
        "id": decision.source_id,
        "dataset_id": decision.dataset_id,
        "dataset_revision": decision.dataset_revision,
        "intended_use": "train",
        "status": "Available",
        "license_receipt": None,
    }
    projected = license_projection(declaration)
    assert {key: projected[key] for key in declaration} == declaration
    assert "publication_scope" not in declaration
    assert projected["publication_scope"]["scope"] == "public_display"
    assert projected["publication_scope"]["upstream_name"] == decision.upstream_name
    assert projected["publication_scope"]["upstream_url"] == decision.upstream_url
    assert "publication_scope" not in license_projection({**declaration, "dataset_revision": "new-pin"})
