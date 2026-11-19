# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A corrected derived-rule receipt cannot erase or silently replace an old census."""

import copy
import json
from pathlib import Path

import pytest

from experiments.post_training.task_curation.derived_contract_census import DerivedContractReceipt
from infra.marina.apps.rl_data_catalog.load_derived_contract import load_contracts
from infra.marina.apps.rl_data_catalog.server.contract_selection import receipt_identity, select_contract_receipts
from infra.marina.tests.test_rl_data_catalog import catalog_database as database_fixture
from infra.marina.tests.test_rl_data_catalog import catalog_query as query_fixture

catalog_database = database_fixture
catalog_query = query_fixture


@pytest.mark.usefixtures("catalog_database")
def test_archived_rule_correction_preserves_original_and_refuses_unlisted_conflicts(catalog_query):
    path = Path(__file__).parents[1] / "apps/rl_data_catalog/tests/fixtures/contract_supersessions.json"
    pair = json.loads(path.read_text())[0]
    old, new = pair["previous"], pair["selected"]
    catalog_query(
        "INSERT INTO atlas_derived_contracts VALUES (:source,:identity,CAST(:receipt AS JSONB))",
        {
            "source": pair["source_id"],
            "identity": receipt_identity(old),
            "receipt": json.dumps(old),
        },
    )
    record = DerivedContractReceipt.model_validate(new)
    assert record.model_dump(mode="json") == new
    assert load_contracts([record], catalog_query) == {"inserted": 1, "records": 1}
    assert load_contracts([record], catalog_query) == {"inserted": 0, "records": 1}
    rows = catalog_query("SELECT receipt FROM atlas_derived_contracts ORDER BY receipt_id", {})["rows"]
    assert len(rows) == 2
    receipts = [row["receipt"] for row in rows]
    assert old in receipts and new in receipts
    assert select_contract_receipts(pair["source_id"], receipts) == [new]
    unlisted = copy.deepcopy(new)
    unlisted["summary"]["single_turn_count"] -= 1
    unlisted["evidence_url"] += "-unlisted"
    with pytest.raises(ValueError, match="Conflicting derived contract summaries"):
        load_contracts([DerivedContractReceipt.model_validate(unlisted)], catalog_query)
    assert catalog_query("SELECT receipt FROM atlas_derived_contracts ORDER BY receipt_id", {})["rows"] == rows
