# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a published cohort-conversion release card to its original upstream license.

The repackager rule (rulings/repackager-license-follows-upstream-20261116.md):
a repackager's empty card is not the tasks' license; the card follows the
original upstream. For a cohort whose publication check escalated on the
repackager's README, this tool resolves the original upstream pin through the
license checker (never by hand), records the release-pin receipt whose
``resolved_source`` names the original upstream, regenerates the release card
so its ``license_source`` declares that upstream, uploads the card, and
re-runs the exact-pin publication check. The checker classifies the upstream;
if it still escalates, the failure prints and the cohort stays excluded.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from finestore.rl.licenses import LicenseEvidence, LicenseReceipt, LicenseSourcePin
from huggingface_hub import HfApi

from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX, append_license_receipt
from experiments.post_training.task_curation.license_publication import (
    check_publication,
    licensed_dataset_card,
    prepare_license_publication,
)
from experiments.post_training.task_curation.license_resolution import LicenseTarget, resolve_license


def _card_evidence(release_repo: str, revision: str, card_bytes: bytes) -> LicenseEvidence:
    """The release card's own wrapper declaration: it declares the original upstream."""
    quoted = [
        line
        for line in card_bytes.decode("utf-8").splitlines()
        if line.startswith(("license_source:", "  dataset_id:", "  revision:", "  url:"))
    ][:6]
    if not quoted:
        raise ValueError("The release card declares no license_source to rebind")
    return LicenseEvidence(
        kind="wrapper_source",
        url=f"https://huggingface.co/datasets/{release_repo}/resolve/{revision}/README.md",
        sha256=hashlib.sha256(card_bytes).hexdigest(),
        license_ids=[],
        quoted_lines=quoted,
        restrictive_terms=False,
        unidentified_third_party_terms=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True, help="The cohort source id the release belongs to.")
    parser.add_argument("--release-repo", required=True, help="The published conversion release repo.")
    parser.add_argument("--upstream-dataset", required=True, help="The original upstream dataset id.")
    parser.add_argument("--upstream-revision", required=True, help="The original upstream pin revision.")
    parser.add_argument("--use", default="train", help="The publication use.")
    parser.add_argument("--output", type=Path, required=True, help="Scratch directory for the regenerated card.")
    parser.add_argument("--index", type=Path, default=LICENSE_RECEIPT_INDEX, help="The license receipt index to bump.")
    args = parser.parse_args()

    api = HfApi()
    info = api.repo_info(args.release_repo, repo_type="dataset")
    if not info.private:
        raise ValueError("The release must be private")
    revision = info.sha

    # The checker resolves the original upstream itself; nothing here classifies a license.
    upstream = resolve_license(
        LicenseTarget(
            args.upstream_dataset,
            args.upstream_dataset,
            args.upstream_revision,
            f"https://huggingface.co/datasets/{args.upstream_dataset}/tree/{args.upstream_revision}",
        )
    )
    if upstream.resolved_source is not None:
        raise ValueError(
            f"The original upstream {args.upstream_dataset}@{args.upstream_revision[:12]} itself declares "
            f"another wrapper source ({upstream.resolved_source.dataset_id}); resolve that pin instead"
        )
    primary_card = next(item for item in upstream.evidence if item.kind == "dataset_card")

    card_path = api.hf_hub_download(args.release_repo, "README.md", repo_type="dataset", revision=revision)
    card_bytes = Path(card_path).read_bytes()

    receipt = LicenseReceipt(
        source_id=args.source_id,
        dataset_id=args.release_repo,
        revision=revision,
        resolved_at=datetime.now(UTC).isoformat(),
        resolver_version=upstream.resolver_version,
        evidence=[*upstream.evidence, _card_evidence(args.release_repo, revision, card_bytes)],
        bundled_third_party=upstream.bundled_third_party,
        model_generated_outputs=upstream.model_generated_outputs,
        decisions=[],
        resolved_source=LicenseSourcePin(
            dataset_id=args.upstream_dataset,
            revision=args.upstream_revision,
            url=f"https://huggingface.co/datasets/{args.upstream_dataset}/tree/{args.upstream_revision}",
            declaration=primary_card,
        ),
        wrapper_chain=[],
    )
    commit = append_license_receipt(
        args.index,
        receipt,
        message=f"Bind {args.source_id}'s conversion release to its original upstream license",
    )
    print(json.dumps({"index_commit": commit[:12], "release_pin": f"{args.release_repo}@{revision[:12]}"}, indent=1))

    publication = prepare_license_publication(receipt, args.output, use=args.use)
    body = card_bytes.decode("utf-8").split("---", 2)[-1].strip()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "README.md").write_text(
        licensed_dataset_card(body, publication, pretty_name="Harbor cohort TaskSpec conversion release")
    )
    card_commit = api.upload_folder(
        repo_id=args.release_repo,
        repo_type="dataset",
        folder_path=args.output,
        allow_patterns=["README.md"],
        commit_message="Bind the release card to its original upstream license",
    )
    evidence = check_publication(args.source_id, args.release_repo, card_commit.oid, args.use)
    print(json.dumps({"verdict": evidence.get("verdict"), "card_revision": card_commit.oid[:12]}, indent=1))


if __name__ == "__main__":
    main()
