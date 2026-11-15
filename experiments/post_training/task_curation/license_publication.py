# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve a checked upstream dataset grant in new private publication cards."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import yaml
from finestore.rl.licenses import (
    LicenseCheckVerdict,
    LicenseReceipt,
    LicenseVerdict,
    check_license,
    license_verdict,
    require_license,
)
from huggingface_hub import DatasetCard, HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

from experiments.post_training.task_curation.license_resolution import (
    LICENSE_FILENAME,
    MAX_EVIDENCE_BYTES,
    LicenseTarget,
    resolve_license,
)
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned, source_license_ban

NOTICE_FILENAME = re.compile(r"^(?:notice|attribution|copyright)(?:[._-].*)?$", re.IGNORECASE)
LICENSE_FIELDS = ("license", "license_name", "license_link")


@dataclass(frozen=True)
class LicensePublication:
    metadata: dict
    notice_files: tuple[Path, ...]
    attribution: str


def prepare_license_publication(receipt: LicenseReceipt, output: Path) -> LicensePublication:
    """Archive the pinned grant and notices after validating the exact input receipt."""
    require_license(receipt, receipt.source_id, receipt.dataset_id, receipt.revision, "train")
    require_unbanned(receipt.source_id, receipt.dataset_id, receipt)
    source = receipt.resolved_source
    dataset_id = source.dataset_id if source is not None else receipt.dataset_id
    revision = source.revision if source is not None else receipt.revision
    url = source.url if source is not None else f"https://huggingface.co/datasets/{dataset_id}/tree/{revision}"
    prefix = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/"
    source_evidence = [
        item for item in receipt.evidence if item.url.startswith(prefix) and item.kind != "task_license_file"
    ]
    evidence_urls = {item.url for item in source_evidence}
    if not source_evidence:
        raise ValueError("Publication has no checked evidence for its upstream license source")
    output.mkdir(parents=True, exist_ok=True)
    metadata, notices, provenance = {}, [], []
    for entry in HfApi().list_repo_tree(dataset_id, repo_type="dataset", revision=revision, recursive=False):
        if not isinstance(entry, RepoFile):
            continue
        filename = entry.path
        if (
            filename != "README.md"
            and not LICENSE_FILENAME.fullmatch(filename)
            and not NOTICE_FILENAME.fullmatch(filename)
        ):
            continue
        if entry.size > MAX_EVIDENCE_BYTES:
            raise ValueError(f"Upstream license metadata exceeds the bounded read: {filename}")
        downloaded = Path(hf_hub_download(dataset_id, filename, repo_type="dataset", revision=revision))
        content = downloaded.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        evidence_url = prefix + quote(filename, safe="/")
        if any(item.sha256 != digest for item in source_evidence if item.url == evidence_url):
            raise ValueError(f"Upstream license metadata differs from its checked receipt: {filename}")
        destination = output / ("UPSTREAM_README.md" if filename == "README.md" else filename)
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"Publication notice destination already contains different bytes: {destination}")
        destination.write_bytes(content)
        notices.append(destination)
        provenance.append({"filename": destination.name, "source_url": evidence_url, "sha256": digest})
        if filename == "README.md":
            card = DatasetCard(content.decode()).data.to_dict()
            metadata.update({field: card[field] for field in LICENSE_FIELDS if field in card})
    observed = {item["source_url"] for item in provenance}
    if evidence_urls - observed:
        raise ValueError("Publication did not preserve every checked upstream license document")
    if "license" not in metadata:
        identifiers = sorted({identifier for item in source_evidence for identifier in item.license_ids} - {"missing"})
        if license_verdict(tuple(identifiers)) is not LicenseVerdict.ALLOWED:
            raise ValueError("Checked grant supplies no publishable permissive license identifier")
        metadata["license"] = identifiers[0] if len(identifiers) == 1 else identifiers
    metadata["license_source"] = {"dataset_id": dataset_id, "revision": revision, "url": url}
    metadata["license_evidence_urls"] = sorted(
        {
            item.url
            for item in source_evidence
            if item.license_ids and license_verdict(tuple(item.license_ids)) is LicenseVerdict.ALLOWED
        }
    )
    if not metadata["license_evidence_urls"]:
        raise ValueError("Publication must retain pinned evidence URLs for its dataset grant")
    metadata["license_input"] = {"dataset_id": receipt.dataset_id, "revision": receipt.revision}
    record = output / "publication-provenance.json"
    record.write_text(
        json.dumps({"metadata": metadata, "documents": provenance, "receipt": receipt.model_dump(mode="json")}, indent=2)
        + "\n"
    )
    notices.append(record)
    attribution = (
        f"License source: [{dataset_id}@{revision}]({url}). "
        "Original notices and attribution are preserved in the accompanying upstream files.\n"
    )
    return LicensePublication(metadata, tuple(notices), attribution)


def licensed_dataset_card(body: str, publication: LicensePublication, *, pretty_name: str) -> str:
    metadata = {"pretty_name": pretty_name, **publication.metadata}
    return (
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False)
        + "---\n\n"
        + body.rstrip()
        + "\n\n"
        + publication.attribution
    )


def check_publication(source_id: str, dataset_id: str, revision: str) -> dict:
    """Require the new private pin's own checker result before returning publication evidence."""
    target = LicenseTarget(
        source_id, dataset_id, revision, f"https://huggingface.co/datasets/{dataset_id}/tree/{revision}"
    )
    receipt = resolve_license(target)
    result = check_license(receipt, source_license_ban(source_id, dataset_id, receipt))
    evidence = {**result.model_dump(mode="json"), "receipt": receipt.model_dump(mode="json")}
    if result.verdict is not LicenseCheckVerdict.PASS:
        raise ValueError(f"Published pin failed its exact-pin license check: {json.dumps(evidence)}")
    return evidence
