# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a pinned dataset's primary license evidence into a receipt."""

import base64
import fnmatch
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests
from finestore.rl.licenses import (
    LICENSE_RESOLVER_VERSION,
    OPENMDW_LICENSE_PATTERN,
    LicenseCheckRule,
    LicenseEvidence,
    LicenseReceipt,
    LicenseSourcePin,
    check_license,
)
from huggingface_hub import HfApi, get_token

from experiments.post_training.task_curation.license_targets import (
    ARCHIVE_SOURCE_PIN,
    HTTP_TIMEOUT,
    MAX_EVIDENCE_BYTES,
    MODEL_OUTPUT_TERMS,
    NATIVE_SOURCE_PIN,
    REPAIR_SOURCE_PIN,
    SOURCE_PIN,
    THIRD_PARTY_TERMS,
    WRAPPER_MANIFESTS,
    WRAPPER_NAME,
    WRAPPER_TITLE,
    LicenseTarget,
    _card_metadata,
    _license_context,
    _license_files,
    license_evidence,
)


def _fetch(url: str, *, huggingface: bool = False) -> bytes:
    token = get_token() if huggingface else None
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, headers=headers, timeout=HTTP_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content) > MAX_EVIDENCE_BYTES:
                raise ValueError(f"License evidence exceeds {MAX_EVIDENCE_BYTES} bytes: {url}")
        return bytes(content)


def _github_json(path: str) -> dict:
    result = subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _is_wrapper_target(target: LicenseTarget, metadata: dict) -> bool:
    return (
        urlsplit(target.url).hostname == "huggingface.co"
        and target.dataset_id.startswith("open-athena/")
        and bool(
            WRAPPER_NAME.search(target.dataset_id)
            or WRAPPER_TITLE.search(str(metadata.get("pretty_name", "")))
            or "license_source" in metadata
        )
    )


@dataclass(frozen=True)
class _ManifestDeclarations:
    """What a wrapper's published manifest declares about its upstream."""

    declarations: list[tuple[str, str, str]]
    manifest_path: str | None
    identity_fields: tuple[str, str]
    declared_by: tuple[str, bytes] | None


def _manifest_declarations(target: LicenseTarget) -> _ManifestDeclarations:
    """The declarations recorded in a wrapper's published manifest, for cards without a license_source."""
    declarations: list[tuple[str, str, str]] = []
    manifest_path = None
    manifest_identity_fields = ("source_dataset_id", "source_revision")
    manifest_declaration: tuple[str, bytes] | None = None
    filenames = {
        item.rfilename for item in HfApi().dataset_info(target.dataset_id, revision=target.revision).siblings or ()
    }
    for filename, paths in WRAPPER_MANIFESTS.items():
        if filename not in filenames:
            continue
        declaration_url = f"https://huggingface.co/datasets/{target.dataset_id}/resolve/{target.revision}/{filename}"
        declaration_content = _fetch(declaration_url, huggingface=True)
        document = json.loads(declaration_content)
        manifest_declaration = (declaration_url, declaration_content)
        for path in paths:
            plan = document
            for key in path.split("/"):
                if not isinstance(plan, dict):
                    raise ValueError(f"Wrapper manifest requires an object at {path}")
                plan = plan.get(key, {})
            if path == "original_release" and path in document and (not isinstance(plan, dict) or not plan):
                raise ValueError("Repair original_release must be a nonempty object")
            if not isinstance(plan, dict) or not plan:
                continue
            source_url = None
            if path == "original_release":
                if set(plan) != {"repo", "revision", "url"}:
                    raise ValueError("Repair original_release must declare repo, revision, and url")
                dataset_id, revision, source_url = plan["repo"], plan["revision"], plan["url"]
                if not isinstance(source_url, str) or not source_url:
                    raise ValueError("Repair original_release URL must be a nonempty string")
                if not source_url.rstrip("/").endswith(f"/tree/{revision}"):
                    raise ValueError("Repair original_release URL must name its exact pinned revision")
            else:
                dataset_id, revision = plan.get("source_dataset_id"), plan.get("source_revision")
            if not isinstance(dataset_id, str) or not isinstance(revision, str):
                raise ValueError(f"Wrapper manifest does not declare a source dataset and revision at {path}")
            declarations.append((dataset_id, revision, source_url or f"https://huggingface.co/datasets/{dataset_id}"))
            manifest_path = path
        if declarations:
            break
    return _ManifestDeclarations(declarations, manifest_path, manifest_identity_fields, manifest_declaration)


def _declared_license_source(target: LicenseTarget, content: bytes, url: str) -> LicenseSourcePin:
    text = content.decode("utf-8")
    metadata = _card_metadata(text)
    declarations = []
    if "license_source" in metadata:
        source = metadata["license_source"]
        if not isinstance(source, dict) or set(source) != {"dataset_id", "revision", "url"}:
            raise ValueError("Wrapper license_source must declare dataset_id, revision, and url")
        if any(not isinstance(value, str) or not value for value in source.values()):
            raise ValueError("Wrapper license_source fields must be nonempty strings")
        declarations.append((source["dataset_id"], source["revision"], source["url"]))
    if not declarations:
        for match in SOURCE_PIN.finditer(text):
            location, revision = match.groups()
            source_url = location if "://" in location else f"https://huggingface.co/datasets/{location}"
            declarations.append((location, revision, source_url))
        declarations.extend(match.groups() for match in NATIVE_SOURCE_PIN.finditer(text))
        for pattern in (ARCHIVE_SOURCE_PIN, REPAIR_SOURCE_PIN):
            declarations.extend(
                (repo, pin, f"https://huggingface.co/datasets/{repo}") for repo, pin in pattern.findall(text)
            )
        table_dataset = re.findall(r"^\| Upstream dataset \| `([^`]+)`", text, re.M)
        table_dataset += re.findall(r"^\| Upstream source \|[^\n`]*`([^`]+/[^`]+)`", text, re.M)
        table_revision = re.findall(r"^\| Upstream revision \| `([0-9a-f]+)`", text, re.M)
        if table_dataset or table_revision:
            if len(table_dataset) != 1 or len(table_revision) != 1:
                raise ValueError("Wrapper provenance table must declare one dataset and revision")
            declarations.append(
                (table_dataset[0], table_revision[0], f"https://huggingface.co/datasets/{table_dataset[0]}")
            )
    declaration_url, declaration_content = url, content
    manifest = None
    if not declarations:
        manifest = _manifest_declarations(target)
        declarations = manifest.declarations
        manifest_path = manifest.manifest_path
        manifest_identity_fields = manifest.identity_fields
        if manifest.declared_by is not None:
            declaration_url, declaration_content = manifest.declared_by
    else:
        manifest_identity_fields = ("source_dataset_id", "source_revision")
    normalized = set()
    for dataset_id, revision, source_url in declarations:
        parts = urlsplit(source_url)
        path = parts.path.strip("/").split("/")
        if parts.hostname == "huggingface.co" and len(path) >= 3 and path[0] == "datasets":
            repository = "/".join(path[1:3])
            source_url = f"https://huggingface.co/datasets/{repository}/tree/{revision}"
        elif parts.hostname == "github.com" and len(path) >= 2:
            repository = "/".join(path[:2])
            source_url = f"https://github.com/{repository}/tree/{revision}"
        elif parts.hostname == "files.pythonhosted.org" and parts.path.endswith(".whl"):
            repository = dataset_id
        else:
            raise ValueError(f"Unsupported declared wrapper source: {source_url}")
        declared_parts = urlsplit(dataset_id)
        declared_path = declared_parts.path.strip("/").split("/")
        if declared_parts.hostname == "huggingface.co" and declared_path[0] == "datasets":
            dataset_id = "/".join(declared_path[1:3])
        elif declared_parts.hostname == "github.com":
            dataset_id = "/".join(declared_path[:2])
        if dataset_id != repository:
            raise ValueError("Wrapper source URL does not identify its declared dataset")
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
            raise ValueError("Wrapper source requires an immutable commit or wheel SHA-256")
        normalized.add((repository, revision, source_url))
    if len(normalized) != 1:
        raise ValueError("Wrapper must declare exactly one unambiguous source pin")
    dataset_id, revision, source_url = normalized.pop()
    if (dataset_id, revision) == (target.dataset_id, target.revision):
        raise ValueError("Wrapper source pin refers to itself")
    lines = text.splitlines()
    selected = set()
    for index, line in enumerate(lines):
        if re.match(r"^license_source:", line):
            selected.add(index)
            for following in range(index + 1, len(lines)):
                if lines[following] and not lines[following][0].isspace():
                    break
                selected.add(following)
        if (
            SOURCE_PIN.match(line)
            or NATIVE_SOURCE_PIN.match(line)
            or ARCHIVE_SOURCE_PIN.search(line)
            or REPAIR_SOURCE_PIN.search(line)
            or re.match(r"^\| Upstream (?:dataset|revision) \|", line)
        ):
            selected.add(index)
    quoted_lines = [f"L{i + 1}: {lines[i]}" for i in sorted(selected)]
    if declaration_url != url:
        quoted_lines = [
            f"JSON /{manifest_path}/{manifest_identity_fields[0]}: {dataset_id}",
            f"JSON /{manifest_path}/{manifest_identity_fields[1]}: {revision}",
        ]
    return LicenseSourcePin(
        dataset_id=dataset_id,
        revision=revision,
        url=source_url,
        declaration=LicenseEvidence(
            url=declaration_url,
            kind="wrapper_source",
            sha256=hashlib.sha256(declaration_content).hexdigest(),
            license_ids=[],
            quoted_lines=quoted_lines,
        ),
    )


def _carries_openmdw_grant(receipt: LicenseReceipt) -> bool:
    """Whether the receipt's own evidence declares the OpenMDW grant at this pin."""
    return any(
        fnmatch.fnmatchcase(identifier, OPENMDW_LICENSE_PATTERN)
        for item in receipt.evidence
        for identifier in item.license_ids
    )


def resolve_license(target: LicenseTarget) -> LicenseReceipt:
    """Normalize our recorded wrapper pins, then check the first dataset without following its lineage."""
    current = replace(target, source_id=target.dataset_id)
    chain = []
    seen = set()
    while True:
        identity = (current.dataset_id, current.revision)
        if identity in seen:
            raise ValueError("Wrapper source declarations form a cycle")
        seen.add(identity)
        primary = _primary_license(current)
        if urlsplit(current.url).hostname != "huggingface.co" or not current.dataset_id.startswith("open-athena/"):
            break
        card = next((item for item in primary.evidence if item.kind == "dataset_card"), None)
        content = _fetch(card.url, huggingface=True) if card else b""
        if not _is_wrapper_target(current, _card_metadata(content.decode("utf-8"))):
            break
        # Explicit restrictions on our own publication still apply to its users.
        if check_license(primary).rule_id in {LicenseCheckRule.RESTRICTED, LicenseCheckRule.THIRD_PARTY}:
            break
        if _carries_openmdw_grant(primary):
            # A first-party publication carrying the OpenMDW grant is licensed from its
            # own card; the declared source records provenance, not terms.
            break
        source = _declared_license_source(current, content, card.url if card else current.url)
        chain.append(source)
        current = LicenseTarget(source.dataset_id, source.dataset_id, source.revision, source.url)
    return primary.model_copy(
        update={
            "source_id": target.source_id,
            "dataset_id": target.dataset_id,
            "revision": target.revision,
            "resolved_source": chain[-1] if chain else None,
            "wrapper_chain": chain[:-1],
        }
    )


@cache
def _primary_license(target: LicenseTarget) -> LicenseReceipt:
    """Read HF card and license files, GitHub pinned licenses, explicit terms URLs, or PyPI
    wheel metadata.

    Authentication, transport, and parsing failures propagate. They must never become a
    permissive or missing verdict. Callers can retain them as unresolved audit errors.
    """
    evidence = []
    documents = []
    parts = urlsplit(target.url)
    if parts.hostname == "huggingface.co":
        info = HfApi().dataset_info(target.dataset_id, revision=target.revision)
        if info.sha != target.revision:
            raise ValueError(f"License resolution requires an immutable dataset commit: {target.dataset_id}")
        filenames = [item.rfilename for item in info.siblings or ()]
        license_files = _license_files(filenames)
        for filename in filenames:
            kind = "dataset_card" if filename == "README.md" else "license_file"
            if kind == "license_file" and filename not in license_files:
                continue
            if kind == "license_file" and "/" in filename:
                kind = "task_license_file"
            url = f"https://huggingface.co/datasets/{target.dataset_id}/resolve/{target.revision}/{quote(filename)}"
            content = _fetch(url, huggingface=True)
            documents.append(_license_context(url, content.decode("utf-8")))
            evidence.append(license_evidence(url, content, kind))
            if filename == "README.md":
                evidence.extend(license_evidence(url, content, item) for item in ("readme_license", "readme_notice"))
        # The pinned repository tree is evidence of absent card/license files too.
        if not evidence:
            url = f"https://huggingface.co/api/datasets/{target.dataset_id}/tree/{target.revision}?recursive=true"
            evidence.append(license_evidence(url, json.dumps(sorted(filenames)).encode(), "repository_tree"))
            evidence[-1] = evidence[-1].model_copy(update={"license_ids": ["missing"]})
    elif parts.hostname == "github.com":
        repo = "/".join(parts.path.strip("/").split("/")[:2])
        commit = _github_json(f"repos/{repo}/commits/{target.revision}")
        if commit["sha"] != target.revision:
            raise ValueError(f"License resolution requires an immutable GitHub commit: {repo}")
        tree = _github_json(f"repos/{repo}/git/trees/{target.revision}?recursive=1")
        if tree.get("truncated"):
            raise ValueError(f"GitHub license tree is truncated: {repo}")
        license_files = _license_files([entry["path"] for entry in tree["tree"] if entry["type"] == "blob"])
        # Current API metadata is evidence, never authority for the pinned verdict.
        for path, document, kind in (
            (f"repos/{repo}/git/trees/{target.revision}?recursive=1", tree, "repository_tree"),
            (f"repos/{repo}", _github_json(f"repos/{repo}"), "github_license_api"),
        ):
            metadata = json.dumps(document, sort_keys=True).encode()
            item = license_evidence(f"https://api.github.com/{path}", metadata, kind)
            evidence.append(item.model_copy(update={"license_ids": [], "restrictive_terms": False}))
        for entry in tree["tree"]:
            filename = entry["path"]
            if entry["type"] != "blob" or not (filename in license_files or filename == "README.md"):
                continue
            if entry["size"] > MAX_EVIDENCE_BYTES:
                raise ValueError(f"License evidence exceeds size limit: {entry['path']}")
            blob = _github_json(f"repos/{repo}/git/blobs/{entry['sha']}")
            content = base64.b64decode(blob["content"])
            url = f"https://github.com/{repo}/blob/{target.revision}/{quote(entry['path'])}"
            documents.append(content.decode("utf-8"))
            if filename == "README.md":
                evidence.extend(
                    license_evidence(url, content, item) for item in ("dataset_card", "readme_license", "readme_notice")
                )
            else:
                evidence.append(
                    license_evidence(url, content, "task_license_file" if "/" in filename else "license_file")
                )
    elif parts.hostname == "files.pythonhosted.org" and parts.path.endswith(".whl"):
        filename = Path(parts.path).name
        package, version, _ = filename.split("-", 2)
        url = f"https://pypi.org/pypi/{package}/{version}/json"
        content = _fetch(url)
        metadata = json.loads(content)
        if not any(
            item["filename"] == filename and item["digests"]["sha256"] == target.revision for item in metadata["urls"]
        ):
            raise ValueError("Package metadata does not bind the selected wheel SHA-256")
        evidence.append(license_evidence(url, content, "package_metadata"))
    elif target.license_url:
        content = _fetch(target.license_url)
        documents.append(content.decode("utf-8"))
        evidence.append(license_evidence(target.license_url, content, "license_file"))
    else:
        raise ValueError(f"Source requires a declared primary license URL: {target.source_id}")
    text = "\n".join(documents)
    return LicenseReceipt(
        resolver_version=LICENSE_RESOLVER_VERSION,
        source_id=target.source_id,
        dataset_id=target.dataset_id,
        revision=target.revision,
        resolved_at=datetime.now(UTC).isoformat(),
        evidence=evidence,
        bundled_third_party=bool(THIRD_PARTY_TERMS.search(text)) or sum(e.kind == "license_file" for e in evidence) > 1,
        model_generated_outputs=bool(MODEL_OUTPUT_TERMS.search(text)),
    )
