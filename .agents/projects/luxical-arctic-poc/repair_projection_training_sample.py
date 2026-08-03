# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the pinned semantic-projection training sample."""

import json
import logging
from pathlib import Path
from typing import Any

from glm_hierarchical_labels import OUTPUT_ROOT, SOURCE_RUN_ROOT
from glm_semantic_labels import SampleDocument, read_jsonl
from label_frozen_hierarchy import MANIFEST_URL
from label_frozen_hierarchy_training import (
    identity_digest,
    projection_training_documents,
    write_projection_training_documents,
)
from ladder_config import read_json
from train_semantic_projection import HIERARCHY_RUN_ID, HIERARCHY_VARIANT
from train_semantic_projection_large import EXCLUDED_EVALUATION_RUN_ID, EXPECTED_DOCUMENTS, LABEL_RUN_ID

RESULT_FILE = Path("/tmp/luxical-projection-training-sample-repair")

logger = logging.getLogger(__name__)


def validate_repair_inputs(config: dict[str, Any], documents: list[SampleDocument]) -> str:
    """Return the sample digest after exact config validation."""
    expected = {
        "training_run_id": LABEL_RUN_ID,
        "pilot_run_id": HIERARCHY_RUN_ID,
        "excluded_evaluation_run_id": EXCLUDED_EVALUATION_RUN_ID,
        "document_count": EXPECTED_DOCUMENTS,
        "purpose": "semantic_projection_training",
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"The saved run config has an incorrect {name}")
    if config.get("variant", {}).get("name") != HIERARCHY_VARIANT:
        raise ValueError("The saved run config has an incorrect hierarchy variant")
    if len(documents) != EXPECTED_DOCUMENTS:
        raise ValueError("The rebuilt training sample has an incorrect document count")
    digest = identity_digest(documents)
    if config.get("training_identity_sha256") != digest:
        raise ValueError("The rebuilt training sample identity differs from the saved run config")
    return digest


def rebuilt_documents() -> list[SampleDocument]:
    """Return the deterministic training sample from its pinned inputs."""
    pilot_documents = [SampleDocument(**row) for row in read_jsonl(SOURCE_RUN_ROOT / "sample-private.jsonl.gz")]
    evaluation_root = OUTPUT_ROOT / HIERARCHY_RUN_ID / HIERARCHY_VARIANT / EXCLUDED_EVALUATION_RUN_ID
    evaluation_documents = [SampleDocument(**row) for row in read_jsonl(evaluation_root / "sample-private.jsonl.gz")]
    return projection_training_documents(
        read_json(MANIFEST_URL), pilot_documents, evaluation_documents, EXPECTED_DOCUMENTS
    )


def main() -> None:
    """Rebuild, validate, and atomically replace the damaged sample."""
    logging.basicConfig(level=logging.INFO)
    training_root = OUTPUT_ROOT / HIERARCHY_RUN_ID / HIERARCHY_VARIANT / LABEL_RUN_ID
    config = read_json(str(training_root / "run-config.json"))
    documents = rebuilt_documents()
    digest = validate_repair_inputs(config, documents)
    stored_documents = write_projection_training_documents(training_root, documents)
    stored_digest = identity_digest(stored_documents)
    if stored_digest != digest:
        raise ValueError("The stored training sample identity differs after replacement")
    result = {
        "sample_url": str(training_root / "sample-private.jsonl.gz"),
        "document_count": len(stored_documents),
        "training_identity_sha256": stored_digest,
    }
    RESULT_FILE.write_text(json.dumps(result, sort_keys=True))
    logger.info("GLM_PROJECTION_TRAINING_SAMPLE_REPAIR=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
