# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record benchmark overlap without modifying evaluation populations or pins."""

import argparse
import json
from pathlib import Path

from finestore.rl.curation import DeconReceiptStatus, LexicalContaminationEvidence, LexicalReceipt, LexicalUse, SourceUse
from pydantic import BaseModel, ConfigDict

from experiments.post_training.task_curation.source import RlDataSource
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId, canonical_source_map


class LexicalInputUse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    dataset_id: str
    dataset_revision: str
    converter_revision: str
    use: LexicalUse


def source_lexical_use(source: RlDataSource) -> LexicalUse:
    """Resolve Use from the registered converter, with the explicit benchmark marker."""
    if source.intended_use is SourceUse.TRAIN:
        return LexicalUse.TRAIN
    return LexicalUse.BENCHMARK if "benchmark" in source.info.tags else LexicalUse.EVAL


def measurement_for_use(evidence: LexicalContaminationEvidence, use: LexicalUse) -> LexicalContaminationEvidence:
    """Benchmarks retain every task; training measurements wait for release readback."""
    if evidence.receipt is not None:
        raise ValueError("Use policy requires a raw measurement without a release receipt")
    if use is LexicalUse.TRAIN:
        return evidence
    receipt = LexicalReceipt(
        use=use,
        status=DeconReceiptStatus.BENCHMARK_MEASURED,
        retained_tasks=evidence.measured_tasks,
        dropped_tasks=0,
        release=None,
    )
    return LexicalContaminationEvidence.model_validate({**evidence.model_dump(mode="json"), "receipt": receipt})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--source-uses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [LexicalInputUse.model_validate(row) for row in json.loads(args.source_uses.read_text())]
    uses = {row.source_id: row for row in inputs}
    raw = json.loads(args.measurements.read_text())
    raw["measurements"] = canonical_source_map(raw["measurements"])
    raw["drop_manifests"] = canonical_source_map(raw["drop_manifests"])
    if len(uses) != len(inputs) or set(uses) != set(raw["measurements"]):
        raise ValueError("Use declarations must cover the frozen measured source set exactly once")
    measurements = {}
    training = []
    for source_id, record in sorted(raw["measurements"].items()):
        evidence = LexicalContaminationEvidence.model_validate(record)
        declared = uses[source_id]
        if (declared.dataset_id, declared.dataset_revision, declared.converter_revision) != (
            evidence.input_dataset_id,
            evidence.input_dataset_revision,
            evidence.converter_revision,
        ):
            raise ValueError(f"Use declaration belongs to another measured input: {source_id}")
        measurements[source_id] = measurement_for_use(evidence, declared.use).model_dump(mode="json")
        if declared.use is LexicalUse.TRAIN:
            training.append(
                {
                    "source_id": source_id,
                    "measured_tasks": evidence.measured_tasks,
                    "contaminated_tasks": evidence.contaminated_tasks,
                    "contaminated_fraction": evidence.contaminated_fraction,
                    "matched_evals": evidence.matched_evals,
                    "max_overlap": evidence.max_overlap,
                    "p95_max_overlap": evidence.p95_max_overlap,
                    "input_dataset_id": evidence.input_dataset_id,
                    "input_dataset_revision": evidence.input_dataset_revision,
                    "drop_manifest_uri": raw["drop_manifests"][source_id],
                    "evidence_url": evidence.evidence_url,
                }
            )
    args.output.write_text(json.dumps({"measurements": measurements, "training_review": training}, indent=2) + "\n")


if __name__ == "__main__":
    main()
