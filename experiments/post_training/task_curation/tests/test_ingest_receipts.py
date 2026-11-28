# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume requires the exact stage identity and unchanged persisted evidence."""

from pathlib import Path

import pytest
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.ingest_config import load_ingest_config
from experiments.post_training.task_curation.ingest_stages.receipts import (
    IngestStage,
    StageInputs,
    StageReceipt,
    load_receipt,
    seal_receipt,
)

CONFIGS = Path(__file__).parents[1] / "ingest_stages/configs"


@pytest.mark.parametrize("config_name", ["litmus-train", "mimo-repair"])
def test_receipt_round_trip_refuses_changed_upstream_policy_and_modified_evidence(tmp_path, config_name):
    config = load_ingest_config(CONFIGS / f"{config_name}.json")
    dependency = seal_manifest({"permission": "pass"}, str(tmp_path / "license.json"))
    output = seal_manifest({"converted": 7, "unique": 6}, str(tmp_path / "dedup.json"))
    inputs = StageInputs(
        stage=IngestStage.DEDUP,
        source_id=config.source_id,
        config_sha256=config.digest(),
        source_revision="a" * 40,
        worker_argv_sha256=canonical_sha256({"argv": ["worker", "--config", config_name]}),
        dependencies={IngestStage.LICENSE: dependency},
        artifacts={},
    )
    receipt = StageReceipt(inputs=inputs, outputs={"population": output})
    reference = seal_receipt(str(tmp_path / "receipts"), receipt)
    assert load_receipt(reference, inputs) == receipt
    assert seal_receipt(str(tmp_path / "receipts"), receipt) == reference

    revised_config = config.model_copy(update={"upstream": config.upstream.model_copy(update={"revision": "b" * 40})})
    revised_inputs = inputs.model_copy(update={"config_sha256": revised_config.digest()})
    with pytest.raises(ValueError, match="receipt inputs changed"):
        load_receipt(reference, revised_inputs)

    Path(output.uri).write_text('{"converted": 70, "unique": 60}\n')
    with pytest.raises(ValueError, match="evidence digest mismatch"):
        load_receipt(reference, inputs)


def test_receipt_rejects_conflicting_results_without_overwriting_the_original(tmp_path):
    first = seal_manifest({"unique": 3}, str(tmp_path / "first.json"))
    second = seal_manifest({"unique": 4}, str(tmp_path / "second.json"))
    config = load_ingest_config(CONFIGS / "litmus-train.json")
    inputs = StageInputs(
        stage=IngestStage.DEDUP,
        source_id=config.source_id,
        config_sha256=config.digest(),
        source_revision="a" * 40,
        worker_argv_sha256=canonical_sha256({"argv": ["worker"]}),
        dependencies={},
        artifacts={},
    )
    receipt = StageReceipt(inputs=inputs, outputs={"population": first})
    reference = seal_receipt(str(tmp_path / "receipts"), receipt)
    original = Path(reference.uri).read_bytes()

    with pytest.raises(ValueError, match="exists with different content"):
        seal_receipt(str(tmp_path / "receipts"), receipt.model_copy(update={"outputs": {"population": second}}))

    assert Path(reference.uri).read_bytes() == original
    assert load_receipt(reference, inputs).outputs["population"] == first
