# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve a checked upstream dataset grant in new private publication cards."""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

import yaml
from finestore.rl.licenses import (
    LicenseCheckVerdict,
    LicenseDecision,
    LicenseDecisionKind,
    LicenseEvidence,
    LicenseReceipt,
    LicenseSourcePin,
    LicenseVerdict,
    check_license,
    license_verdict,
    require_license,
)
from huggingface_hub import DatasetCard, HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

from experiments.post_training.task_curation.license_binding import (
    LICENSE_RECEIPT_INDEX,
    adopt_license_receipt,
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
    require_publication_license(license_receipt, source_id, license_receipt.dataset_id, license_receipt.revision, use)
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
    """A published population pin whose licence evidence carried forward mechanically."""

    source_id: str
    repo: str
    revision: str
    parent: tuple[str, str]
    license_source: dict
    receipt: LicenseReceipt
    registry_commit: str
    license_check: dict
    file_hashes: dict[str, str]


def _publish_namespace(source_id: str, revision: str) -> str:
    """One receipt-shard namespace per published pin, so a re-seal replaces its row."""
    slug = re.sub(r"[^a-z0-9]+", "-", source_id.lower()).strip("-")[:40].rstrip("-")
    if not slug:
        raise ValueError(f"Source id reduces to no registry namespace slug: {source_id}")
    return f"publish-{slug}-{revision[:12]}"


def _card_evidence(release: str, revision: str, card_bytes: bytes) -> LicenseEvidence:
    """The release card's own wrapper declaration, quoting its license_source lines."""
    quoted = [
        line
        for line in card_bytes.decode("utf-8").splitlines()
        if line.startswith(("license_source:", "  dataset_id:", "  revision:", "  url:"))
    ][:6]
    if not quoted:
        raise ValueError("The release card declares no license_source to bind")
    return LicenseEvidence(
        kind="wrapper_source",
        url=f"https://huggingface.co/datasets/{release}/resolve/{revision}/README.md",
        sha256=hashlib.sha256(card_bytes).hexdigest(),
        license_ids=[],
        quoted_lines=quoted,
        restrictive_terms=False,
        unidentified_third_party_terms=False,
    )


def repin_license_receipt(previous: LicenseReceipt, release: str, revision: str, card_bytes: bytes) -> LicenseReceipt:
    """Re-pin a source's passing receipt onto a newly published release pin.

    The release derives from the receipt's own dataset, so the checker-verified
    resolution carries over byte-for-byte: the receipt is the immutable record, and
    re-resolving an aggregate upstream through a different wrapper path would
    manufacture a different pin than the recorded one. The new release's own card is
    arbitrated separately by check_publication. Recorded decisions carry their
    original lineage per the repackager rule.
    """
    carried = []
    for decision in previous.decisions:
        carried.append(
            LicenseDecision(
                kind=decision.kind,
                approved=decision.approved,
                who=decision.who,
                when=datetime.now(UTC).isoformat(),
                source_id=previous.source_id,
                dataset_id=release,
                revision=revision,
                scope=list(decision.scope),
                rationale=(
                    f"{decision.rationale} Lineage: recorded at {decision.dataset_id}@{decision.revision} "
                    f"({decision.when}); carried to this release pin by the repackager rule."
                ),
                evidence_urls=list(decision.evidence_urls),
                issue_url=decision.issue_url,
            )
        )
    return LicenseReceipt(
        source_id=previous.source_id,
        dataset_id=release,
        revision=revision,
        resolved_at=previous.resolved_at,
        resolver_version=previous.resolver_version,
        evidence=[*previous.evidence, _card_evidence(release, revision, card_bytes)],
        bundled_third_party=previous.bundled_third_party,
        model_generated_outputs=previous.model_generated_outputs,
        decisions=carried,
        resolved_source=previous.resolved_source,
        wrapper_chain=previous.wrapper_chain,
    )


def _parent_declared_pin(parent: LicenseReceipt, license_source: dict) -> LicenseSourcePin:
    """The pin the new card declares: the parent's resolved upstream, else the parent itself."""
    if parent.resolved_source is not None:
        return parent.resolved_source
    dataset_id, revision = license_source["dataset_id"], license_source["revision"]
    evidence_at_pin = [
        item
        for item in parent.evidence
        if item.url == license_source["url"]
        or item.url.startswith(f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/")
    ]
    declaration = next((item for item in evidence_at_pin if item.kind == "dataset_card"), None)
    if declaration is None:
        declaration = next((item for item in evidence_at_pin if item.kind == "package_metadata"), None)
    if declaration is None and evidence_at_pin:
        declaration = evidence_at_pin[0]
    if declaration is None:
        raise ValueError(f"The parent receipt at {dataset_id}@{revision} carries no evidence at its declared pin")
    return LicenseSourcePin(
        dataset_id=dataset_id,
        revision=revision,
        url=license_source["url"],
        declaration=declaration,
    )


def publish_licensed_population(
    source_id: str,
    parent_dataset_id: str,
    parent_revision: str,
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
    """Publish a task population whose licence evidence moves forward mechanically.

    The parent pin's receipt in the receipts registry must pass its exact-pin check —
    without it nothing publishes (fail closed). The card is written from the parent's
    checked grant, so its ``license_source`` declares the parent's resolved upstream
    pin; the upload is byte-frozen, read back and private; the new pin then passes
    ``check_publication`` itself; and the new pin's receipt, resolved to that same
    upstream, replaces the parent's row in the registry before this returns. A re-seal
    therefore never loses its licence, and a publish never returns unlicensed.
    """
    receipts = read_license_receipts(index_path)
    parent = receipts.get(source_id)
    if parent is None or (parent.dataset_id, parent.revision) != (parent_dataset_id, parent_revision):
        raise ValueError(
            f"{source_id}: no passing licence receipt at the parent pin "
            f"{parent_dataset_id}@{parent_revision}; the population cannot publish until its "
            "parent's exact-pin evidence is registered"
        )
    require_publication_license(parent, source_id, parent.dataset_id, parent.revision, use)
    publication = prepare_license_publication(parent, folder, use=use, lineage=lineage)
    license_source = publication.metadata["license_source"]
    (folder / "README.md").write_text(licensed_dataset_card(body, publication, pretty_name=pretty_name))
    upload_files = tuple(dict.fromkeys((*files, "README.md", *(notice.name for notice in publication.notice_files))))
    prepared = prepare_private_dataset(
        source_id,
        repo,
        folder,
        upload_files,
        parent,
        use=use,
        lineage=lineage,
        license_source=license_source,
    )
    published = publish_private_dataset(prepared)
    receipt = repin_license_receipt(parent, repo, published.revision, (folder / "README.md").read_bytes())
    if receipt.resolved_source is None:
        receipt = receipt.model_copy(update={"resolved_source": _parent_declared_pin(parent, license_source)})
    registry_commit = adopt_license_receipt(
        index_path,
        receipt,
        namespace=_publish_namespace(source_id, published.revision),
        message=f"Bind {source_id}'s published release {repo} to its carried-forward upstream licence",
    )
    return PublishedLicensedPopulation(
        source_id=source_id,
        repo=repo,
        revision=published.revision,
        parent=(parent_dataset_id, parent_revision),
        license_source=license_source,
        receipt=receipt,
        registry_commit=registry_commit,
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
    receipt: LicenseReceipt | None, source_id: str, dataset_id: str | None, revision: str | None, use: str
) -> None:
    """Check current source and upstream registry bans before the pinned license grant."""
    require_unbanned(source_id, dataset_id, receipt, use)
    require_license(receipt, source_id, dataset_id, revision, use, ban=None)


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
    require_publication_license(receipt, receipt.source_id, receipt.dataset_id, receipt.revision, use)
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
            # The published card states the license the checker verified at the resolved
            # source, not the upstream card's own declared field, which stays archived.
            citation = next(
                (item for item in source_evidence if license_verdict(tuple(item.license_ids)) is LicenseVerdict.ALLOWED),
                None,
            )
            if citation is None:
                raise ValueError("Publication has no checked permissive license evidence for its resolved source")
            identifiers = sorted(set(citation.license_ids) - {"missing"})
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


def recorded_lineage_exception(source_id: str, dataset_id: str, revision: str, use: str) -> LicenseDecision | None:
    """The user-cleared parent exception bound to this exact pin in the receipts registry.

    A lineage receipt binds one source identity at one resolved upstream pin; its typed
    exception decision records the user's clearance for a scoped use. Only that exact
    combination rescues a card-check escalation: the receipt must sit at the published
    pin, carry the resolved upstream, and hold an approved exception for this use.
    """
    receipt = read_license_receipts(LICENSE_RECEIPT_INDEX).get(source_id)
    if receipt is None or (receipt.dataset_id, receipt.revision) != (dataset_id, revision):
        return None
    if receipt.resolved_source is None:
        return None
    return next(
        (
            decision
            for decision in receipt.decisions
            if decision.kind is LicenseDecisionKind.EXCEPTION and decision.approved and use in decision.scope
        ),
        None,
    )


def check_publication(source_id: str, dataset_id: str, revision: str, use: str, *, lineage: dict | None = None) -> dict:
    """Require the new private pin's own checker result before returning publication evidence.

    A card-check escalation stays blocked unless the receipts registry binds this exact
    repo@revision to a user-cleared parent lineage exception covering the publication's
    use; the recorded decision then carries the pass and stays in the evidence. Fail
    verdicts never clear: a restricted or banned pin fails its own publication — except
    at a repackager pin whose folder-lineage receipt carries an approved per-folder
    decision (the repackager-license-follows-upstream ruling): the license follows each
    included folder's original upstream, so the repackager's own empty card does not
    fail the publication.
    """
    target = LicenseTarget(
        source_id, dataset_id, revision, f"https://huggingface.co/datasets/{dataset_id}/tree/{revision}"
    )
    receipt = resolve_license(target)
    result = check_license(receipt, source_license_ban(source_id, dataset_id, receipt, use))
    evidence = {**result.model_dump(mode="json"), "receipt": receipt.model_dump(mode="json")}
    if result.verdict is LicenseCheckVerdict.PASS:
        return evidence
    if result.verdict is LicenseCheckVerdict.ESCALATE:
        exception = recorded_lineage_exception(source_id, dataset_id, revision, use)
        if exception is not None:
            return {
                **evidence,
                "verdict": LicenseCheckVerdict.PASS.value,
                "lineage_exception": exception.model_dump(mode="json"),
            }
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
