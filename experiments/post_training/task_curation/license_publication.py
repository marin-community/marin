# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve a checked upstream dataset grant in new private publication cards."""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

import yaml
from finestore.rl.licenses import (
    LicenseCheckVerdict,
    LicenseDecisionKind,
    LicenseEvidence,
    LicenseReceipt,
    LicenseVerdict,
    check_license,
    license_verdict,
    require_license,
)
from huggingface_hub import DatasetCard, HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

from experiments.post_training.task_curation.license_binding import (
    LICENSE_RECEIPT_INDEX,
    read_license_receipts,
)
from experiments.post_training.task_curation.license_resolution import (
    LICENSE_FILENAME,
    MAX_EVIDENCE_BYTES,
    LicenseTarget,
    _fetch,
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
    license_table: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class PreparedPrivateDataset:
    source_id: str
    repo: str
    use: str
    folder: Path
    file_hashes: dict[str, str]
    lineage: dict | None = None
    license_source: dict | None = None


@dataclass(frozen=True)
class PublishedPrivateDataset:
    revision: str
    license_check: dict


def prepare_private_dataset(
    source_id: str,
    repo: str,
    folder: Path,
    files: tuple[str, ...],
    license_receipt: LicenseReceipt,
    *,
    use: str,
    lineage: dict | None = None,
    license_source: dict | None = None,
) -> PreparedPrivateDataset:
    """Validate the input grant and freeze upload bytes without repository writes."""
    require_publication_license(license_receipt, source_id, license_receipt.dataset_id, use)
    check_publication(source_id, license_receipt.dataset_id, license_receipt.revision, use, lineage=lineage)
    if "README.md" not in files or len(files) != len(set(files)):
        raise ValueError("Private publication needs one card and a unique file inventory")
    if any(Path(name).name != name for name in files):
        raise ValueError("Private publication files must be immediate children of the upload folder")
    hashes = {}
    for name in files:
        with (folder / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return PreparedPrivateDataset(source_id, repo, use, folder, hashes, lineage, license_source)


def publish_private_dataset(prepared: PreparedPrivateDataset) -> PublishedPrivateDataset:
    """Write a prepared private dataset and check every uploaded byte and its grant."""
    for name, digest in prepared.file_hashes.items():
        with (prepared.folder / name).open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != digest:
            raise ValueError(f"Prepared private publication changed before creation: {name}")
    api = HfApi()
    try:
        api.create_repo(repo_id=prepared.repo, repo_type="dataset", private=True, exist_ok=False)
    except Exception as error:
        # A prior attempt of this same frozen campaign may have created the repo before
        # its gate stopped; reuse it only when its inventory is exactly this publication's.
        existing = {
            entry.path
            for entry in api.list_repo_tree(prepared.repo, repo_type="dataset", recursive=True)
            if getattr(entry, "path", None)
        }
        # The Hub creates .gitattributes on repo creation; platform boilerplate is not
        # publication content. Anything else outside this publication fails the reuse.
        existing.discard(".gitattributes")
        expected = set(prepared.file_hashes)
        if not existing <= expected or not any(existing):
            raise ValueError(f"Private publication repo already exists with other content: {prepared.repo}") from error
        existing_private = api.repo_info(prepared.repo, repo_type="dataset")
        if not existing_private.private:
            raise ValueError("Private publication destination is not private") from error
        if "README.md" in existing and prepared.license_source is not None:
            # A different population must never overwrite this repo's card: the reused
            # destination may only carry this publication's declared licence pin.
            existing_card = Path(
                hf_hub_download(prepared.repo, "README.md", repo_type="dataset", revision=existing_private.sha)
            ).read_bytes()
            declared = DatasetCard(existing_card.decode()).data.to_dict().get("license_source")
            if declared is not None and (
                declared.get("dataset_id"),
                declared.get("revision"),
            ) != (prepared.license_source["dataset_id"], prepared.license_source["revision"]):
                raise ValueError(
                    f"Private publication repo declares another licence source: {prepared.repo}; "
                    f"use a new version name for this population"
                ) from error
        logging.info("PUBLICATION_REPO_REUSED repo=%s files=%d", prepared.repo, len(existing))
    initial = api.repo_info(prepared.repo, repo_type="dataset")
    if not initial.private:
        raise ValueError("Private publication destination is not private")
    commit = api.upload_folder(
        repo_id=prepared.repo,
        repo_type="dataset",
        folder_path=prepared.folder,
        allow_patterns=list(prepared.file_hashes),
        parent_commit=initial.sha,
        commit_message="Publish verified private dataset",
    )
    if not api.repo_info(prepared.repo, repo_type="dataset", revision=commit.oid).private:
        raise ValueError("Uploaded publication is not private")
    for name, digest in prepared.file_hashes.items():
        local = hf_hub_download(prepared.repo, name, repo_type="dataset", revision=commit.oid)
        with Path(local).open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != digest:
            raise ValueError(f"Private publication readback changed: {name}")
    gate = check_publication(prepared.source_id, prepared.repo, commit.oid, prepared.use, lineage=prepared.lineage)
    return PublishedPrivateDataset(commit.oid, gate)


@dataclass(frozen=True)
class PublishedLicensedPopulation:
    """A published revision carrying its source's ingest-time licence record."""

    source_id: str
    repo: str
    revision: str
    ingest_record: tuple[str, str]
    license_source: dict
    license_check: dict
    file_hashes: dict[str, str]


def publish_licensed_population(
    source_id: str,
    repo: str,
    folder: Path,
    files: tuple[str, ...],
    *,
    body: str,
    pretty_name: str,
    use: str,
    lineage: dict | None = None,
    index_path: Path = LICENSE_RECEIPT_INDEX,
) -> PublishedLicensedPopulation:
    """Publish a revision of an ingested source; its ingest licence record governs.

    A licence is ingest-time provenance (user ruling 2026-11-27): the publication looks
    the licence up by source_id, the card's ``license_source`` declares the ingest
    record's upstream provenance (its permissive licence set, for a source repaired at
    ingest), the upload is byte-frozen, read back and private, and the new pin passes
    the publication gate on the record — no per-pin re-resolution, no per-revision
    receipt, and nothing is written to the receipts registry. A source without a record
    refuses: it is licensed once, at ingest, before any revision publishes.
    """
    ingest = read_license_receipts(index_path).get(source_id)
    if ingest is None:
        raise ValueError(
            f"{source_id}: no ingest licence record; a source is licensed once at ingest — "
            "register its receipt before publishing any revision"
        )
    publication = prepare_license_publication(ingest, folder, use=use, lineage=lineage)
    license_source = publication.metadata["license_source"]
    (folder / "README.md").write_text(licensed_dataset_card(body, publication, pretty_name=pretty_name))
    upload_files = tuple(dict.fromkeys((*files, "README.md", *(notice.name for notice in publication.notice_files))))
    prepared = prepare_private_dataset(
        source_id,
        repo,
        folder,
        upload_files,
        ingest,
        use=use,
        lineage=lineage,
        license_source=license_source,
    )
    published = publish_private_dataset(prepared)
    return PublishedLicensedPopulation(
        source_id=source_id,
        repo=repo,
        revision=published.revision,
        ingest_record=(ingest.dataset_id, ingest.revision),
        license_source=license_source,
        license_check=published.license_check,
        file_hashes=prepared.file_hashes,
    )


def _package_wheel_url(receipt: LicenseReceipt, evidence: LicenseEvidence) -> tuple[str, bytes]:
    """Locate the checked wheel's release URL inside the pinned package metadata."""
    content = _fetch(evidence.url)
    if hashlib.sha256(content).hexdigest() != evidence.sha256:
        raise ValueError(f"Checked package metadata changed since its license check: {evidence.url}")
    document = json.loads(content)
    release = next(
        (
            item
            for item in document["urls"]
            if item["filename"] == receipt.dataset_id and item["digests"]["sha256"] == receipt.revision
        ),
        None,
    )
    if release is None:
        raise ValueError(f"Package metadata does not bind the checked wheel: {receipt.dataset_id}")
    url = release["url"]
    parts = urlsplit(url)
    if parts.hostname != "files.pythonhosted.org" or not parts.path.endswith(".whl"):
        raise ValueError(f"Checked package release is not a pythonhosted wheel: {url}")
    return url, content


def _prepare_package_publication(receipt: LicenseReceipt, evidence: LicenseEvidence, output: Path) -> LicensePublication:
    """Carry a checked package grant into the release card without reinterpreting it.

    A package grant's checked evidence is the pinned PyPI metadata document bound to the
    wheel SHA-256. The release card declares the same package pin, so the new pin's
    exact check resolves the identical grant; the checked document is preserved beside
    the card.
    """
    identifiers = sorted(set(evidence.license_ids) - {"missing"})
    if license_verdict(tuple(identifiers)) is not LicenseVerdict.ALLOWED:
        raise ValueError("Checked package grant supplies no publishable permissive license identifier")
    wheel_url, content = _package_wheel_url(receipt, evidence)
    # HF's card schema accepts its own lowercase identifier spellings.
    identifiers = [identifier.lower() for identifier in identifiers]
    metadata = {
        "license": identifiers[0] if len(identifiers) == 1 else identifiers,
        "license_source": {"dataset_id": receipt.dataset_id, "revision": receipt.revision, "url": wheel_url},
        "license_evidence_urls": [evidence.url],
        "license_input": {"dataset_id": receipt.dataset_id, "revision": receipt.revision},
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "UPSTREAM_PACKAGE_METADATA.json"
    destination.write_bytes(content)
    provenance = [{"filename": destination.name, "source_url": evidence.url, "sha256": evidence.sha256}]
    record = output / "publication-provenance.json"
    record.write_text(
        json.dumps({"metadata": metadata, "documents": provenance, "receipt": receipt.model_dump(mode="json")}, indent=2)
        + "\n"
    )
    attribution = (
        f"License source: [{receipt.dataset_id}@{receipt.revision}]({wheel_url}). "
        "The checked package metadata is preserved in the accompanying upstream file.\n"
    )
    return LicensePublication(metadata, (destination, record), attribution)


def require_publication_license(
    receipt: LicenseReceipt | None, source_id: str, dataset_id: str | None, use: str
) -> None:
    """Check current source and upstream registry bans, then the source's licence record.

    The record is the source's ingest-time licence (ruling 2026-11-27); the dataset id
    feeds the registry ban lookup, and the record's own verdict governs.
    """
    require_unbanned(source_id, dataset_id, receipt, use)
    require_license(receipt, source_id, use, ban=None)


def lineage_folder_identifier(record: dict) -> str:
    """The folder's accepted upstream license identifier from its approved lineage decision.

    The repackager ruling (repackager-license-follows-upstream-20261116) licenses each
    folder under its original upstream. A folder's PASS analog is an approved decision
    whose rationale names the accepted identifier ("accepted ... as MIT"); the parse is
    fail-closed so an ambiguous record stops the publication.
    """
    source_id = record.get("source_id", "")
    approved = [d for d in record.get("decisions", []) if d.get("approved")]
    if not approved:
        raise ValueError(f"Folder lineage lacks an approved decision: {source_id}")
    match = re.search(r"as ([A-Za-z0-9._+-]+)\b", approved[-1].get("rationale", ""))
    if match is None:
        raise ValueError(f"Folder lineage decision does not name an accepted license identifier: {source_id}")
    return match.group(1)


def prepare_license_publication(
    receipt: LicenseReceipt, output: Path, *, use: str, lineage: dict | None = None, folders: list[str] | None = None
) -> LicensePublication:
    """Archive the pinned grant and notices after validating the exact input receipt."""
    require_publication_license(receipt, receipt.source_id, receipt.dataset_id, use)
    package_evidence = [
        item
        for item in receipt.evidence
        if item.kind == "package_metadata" and urlsplit(item.url).hostname == "pypi.org"
    ]
    if receipt.resolved_source is None and package_evidence:
        if len(package_evidence) != 1:
            raise ValueError("Package grants require exactly one checked package metadata document")
        return _prepare_package_publication(receipt, package_evidence[0], output)
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
    # Folder-lineage upstreams keep their license documents in per-dataset subfolders
    # (e.g. endless_v2/README.md), so the listing must be recursive and the preserved
    # destinations must not collide across folders.
    for entry in HfApi().list_repo_tree(dataset_id, repo_type="dataset", revision=revision, recursive=True):
        if not isinstance(entry, RepoFile):
            continue
        filename = entry.path
        basename = PurePosixPath(filename).name
        if (
            basename != "README.md"
            and not LICENSE_FILENAME.fullmatch(basename)
            and not NOTICE_FILENAME.fullmatch(basename)
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
        if "/" in filename:
            # Subfolder license documents keep their folder identity in the name.
            destination = output / ("UPSTREAM_" + filename.replace("/", "_"))
        else:
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
    if lineage is not None:
        # The repackager ruling (repackager-license-follows-upstream-20261116): the license
        # follows each included folder's original upstream. One identifier -> that id on the
        # card; several -> license: other, with the per-folder table returned to the caller
        # for the card body. Non-PASS folders are excluded by the caller before publication.
        included = folders if folders is not None else sorted(lineage)
        if not included:
            raise ValueError("Folder-lineage publication includes no folders")
        license_table = []
        identifiers = set()
        evidence_urls = set()
        for folder in included:
            record = lineage.get(folder)
            if record is None:
                raise ValueError(f"Folder lineage receipt lacks the included folder: {folder}")
            identifier = lineage_folder_identifier(record)
            identifiers.add(identifier.lower())
            license_table.append(
                {
                    "folder": folder,
                    "upstream": f"{record.get('dataset_id')}@{record.get('revision')}",
                    "identifier": identifier,
                    "receipt_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                }
            )
            evidence_urls.update(record.get("decisions", [])[-1].get("evidence_urls", []))
            evidence_urls.update(
                item["url"]
                for item in record.get("evidence", [])
                if isinstance(item, dict) and item.get("url", "").startswith("https://huggingface.co/")
            )
        ordered = sorted(identifiers)
        metadata["license"] = ordered[0] if len(ordered) == 1 else "other"
        metadata["license_evidence_urls"] = sorted(evidence_urls)
    else:
        check = check_license(receipt)
        if check.verdict is LicenseCheckVerdict.PASS:
            # The published card states the licence set the checker verified at the
            # resolved source — every permissive item's identifier, not just the first
            # citation — so a source repaired at ingest to permissive-only items carries
            # its whole set forward (user ruling 2026-11-27). The upstream card's own
            # declared field stays archived.
            identifiers = sorted(
                {
                    identifier.lower()
                    for item in source_evidence
                    if license_verdict(tuple(item.license_ids)) is LicenseVerdict.ALLOWED
                    for identifier in item.license_ids
                }
                - {"missing"}
            )
            if not identifiers:
                raise ValueError("Publication has no checked permissive license evidence for its resolved source")
            metadata["license"] = identifiers[0] if len(identifiers) == 1 else identifiers
        elif "license" not in metadata:
            identifiers = sorted(
                {identifier for item in source_evidence for identifier in item.license_ids} - {"missing"}
            )
            if license_verdict(tuple(identifiers)) is not LicenseVerdict.ALLOWED:
                raise ValueError("Checked grant supplies no publishable permissive license identifier")
            metadata["license"] = identifiers[0] if len(identifiers) == 1 else identifiers
        metadata["license_evidence_urls"] = sorted(
            {
                item.url
                for item in source_evidence
                if item.license_ids and license_verdict(tuple(item.license_ids)) is LicenseVerdict.ALLOWED
            }
        )
        license_table = ()
    # HF's card schema accepts its own lowercase identifier spellings.
    if isinstance(metadata.get("license"), str):
        metadata["license"] = metadata["license"].lower()
    elif isinstance(metadata.get("license"), list):
        metadata["license"] = [str(item).lower() for item in metadata["license"]]
    metadata["license_source"] = {"dataset_id": dataset_id, "revision": revision, "url": url}
    if not metadata.get("license_evidence_urls"):
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
    return LicensePublication(metadata, tuple(notices), attribution, tuple(license_table))


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


def check_publication(source_id: str, dataset_id: str, revision: str, use: str, *, lineage: dict | None = None) -> dict:
    """Require the publication's licence: the source's ingest record governs every revision.

    A source_id with an ingest record passes on that record — the gate looks the licence
    up by source_id, never re-resolves the new repo@revision card, and never refuses a
    revision for lacking its own upstream pin declaration (ruling 2026-11-27). Bans still
    apply at every revision. A source without a record is being ingested by this
    publication: the full ban-and-doubt check runs at this pin, and a card-check
    escalation stays blocked unless the receipts registry binds this exact repo@revision
    to a user-cleared parent lineage exception covering the publication's use. Fail
    verdicts never clear: a restricted or banned pin fails its own publication — except
    at a repackager pin whose folder-lineage receipt carries an approved per-folder
    decision (the repackager-license-follows-upstream ruling): the license follows each
    included folder's original upstream, so the repackager's own empty card does not
    fail the publication.
    """
    ingest = read_license_receipts(LICENSE_RECEIPT_INDEX).get(source_id)
    if ingest is not None:
        # Ingest-time provenance: bans first (they stay at every revision), then the
        # record's own verdict and use scope. The new pin is never resolved.
        require_unbanned(source_id, dataset_id, ingest, use)
        require_license(ingest, source_id, use, ban=None)
        result = check_license(ingest, source_license_ban(source_id, dataset_id, ingest, use))
        evidence = {
            **result.model_dump(mode="json"),
            "receipt": ingest.model_dump(mode="json"),
            "ingest_record": {"dataset_id": ingest.dataset_id, "revision": ingest.revision},
        }
        if result.verdict is LicenseCheckVerdict.PASS:
            return evidence
        exception = next(
            (
                decision
                for decision in ingest.decisions
                if decision.kind is LicenseDecisionKind.EXCEPTION and decision.approved and use in decision.scope
            ),
            None,
        )
        if exception is not None:
            # require_license already cleared this record for the use; report the carry.
            return {
                **evidence,
                "verdict": LicenseCheckVerdict.PASS.value,
                "ingest_exception": exception.model_dump(mode="json"),
            }
        raise ValueError(f"Published pin failed its exact-pin license check: {json.dumps(evidence)}")
    target = LicenseTarget(
        source_id, dataset_id, revision, f"https://huggingface.co/datasets/{dataset_id}/tree/{revision}"
    )
    receipt = resolve_license(target)
    result = check_license(receipt, source_license_ban(source_id, dataset_id, receipt, use))
    evidence = {**result.model_dump(mode="json"), "receipt": receipt.model_dump(mode="json")}
    if result.verdict is LicenseCheckVerdict.PASS:
        return evidence
    if result.verdict is LicenseCheckVerdict.FAIL and lineage is not None:
        record = lineage.get(source_id)
        resolved = (result.dataset_id, result.revision)
        wrapper = evidence.get("receipt", {}).get("resolved_source") or {}
        wrapper_pin = (wrapper.get("dataset_id"), wrapper.get("revision"))
        if record is None or (record.get("dataset_id"), record.get("revision")) not in (resolved, wrapper_pin):
            record = None
            for candidate in lineage.values():
                pin = (candidate.get("dataset_id"), candidate.get("revision"))
                if pin in (resolved, wrapper_pin):
                    record = candidate
                    break
        if record is not None:
            identifier = lineage_folder_identifier(record)
            return {
                **evidence,
                "verdict": LicenseCheckVerdict.PASS.value,
                "folder_lineage": {
                    "source_id": record.get("source_id"),
                    "upstream": f"{record.get('dataset_id')}@{record.get('revision')}",
                    "identifier": identifier,
                    "receipt_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                },
            }
    raise ValueError(f"Published pin failed its exact-pin license check: {json.dumps(evidence)}")
