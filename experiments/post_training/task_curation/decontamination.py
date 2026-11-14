# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atlas release policy applied before QA and admission."""

from finestore.rl.curation import DeconReceiptStatus, LexicalContaminationEvidence, require_decontamination


def require_private_release(
    evidence: LexicalContaminationEvidence | None, dataset_id: str | None, dataset_revision: str | None
) -> None:
    require_decontamination(evidence, dataset_id, dataset_revision)
    assert evidence is not None and evidence.receipt is not None
    receipt = evidence.receipt
    release = receipt.release
    assert release is not None
    if not release.private or release.format != "taskcompendium" or not release.dataset_id.startswith("open-athena/"):
        raise ValueError("Atlas admission requires a private open-athena TaskCompendium release")
    if receipt.status is DeconReceiptStatus.REMEDIATED:
        assert release.original_private_dataset_id is not None
        if not release.original_private_dataset_id.startswith("open-athena/"):
            raise ValueError("Atlas remediation requires the original private open-athena snapshot")
