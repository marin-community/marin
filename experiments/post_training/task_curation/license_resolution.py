# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve primary license evidence without downloading task payloads."""

import base64
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests
import yaml
from finestore.rl.licenses import LICENSE_RESOLVER_VERSION, LicenseEvidence, LicenseReceipt
from huggingface_hub import HfApi, get_token

MAX_EVIDENCE_BYTES = 1024 * 1024
HTTP_TIMEOUT = 30
LICENSE_FILENAME = re.compile(r"^(licen[cs]e|copying|terms)([._-].*)?$", re.IGNORECASE)
RESTRICTIVE_TERMS = re.compile(
    r"non[- ]commercial|research[- ]only|no[- ]derivatives|not (?:be )?(?:used|use).{0,80}"
    r"(?:train|redistribut)|(?:train|redistribut).{0,80}(?:prohibited|forbidden|not permitted)",
    re.IGNORECASE,
)
MODEL_OUTPUT_TERMS = re.compile(
    r"synthetic|model.generated|generated (?:by|with|using)|gpt[- ]?[345]|claude", re.IGNORECASE
)
THIRD_PARTY_TERMS = re.compile(r"third.party|bundled|mixture|compilation|aggregat(?:e|ed|ion)", re.IGNORECASE)


@dataclass(frozen=True)
class LicenseTarget:
    source_id: str
    dataset_id: str
    revision: str
    url: str
    license_url: str | None = None


def _license_ids(text: str, kind: str) -> list[str]:
    if kind == "dataset_card":
        if not text.startswith("---"):
            return ["missing"]
        parts = text.split("---", 2)
        if len(parts) < 3:
            return ["missing"]
        metadata = yaml.safe_load(parts[1]) or {}
        value = metadata.get("license")
        if value is None:
            return ["missing"]
        return [str(item) for item in value] if isinstance(value, list) else [str(value)]
    spdx = re.findall(r"SPDX-License-Identifier:\s*([^\n]+)", text)
    if spdx:
        return [value.strip() for value in spdx]
    lowered = text.lower()
    if "apache license" in lowered and "version 2.0" in lowered:
        return ["Apache-2.0"]
    if "permission is hereby granted, free of charge" in lowered:
        return ["MIT"]
    if "bsd 3-clause" in lowered or "bsd-3-clause" in lowered:
        return ["BSD-3-Clause"]
    if "bsd 2-clause" in lowered or "bsd-2-clause" in lowered:
        return ["BSD-2-Clause"]
    if "creative commons attribution" in lowered or "creativecommons.org/licenses/" in lowered:
        match = re.search(r"creativecommons.org/licenses/(by(?:-nc)?(?:-nd|-sa)?)/(\d\.\d)", lowered)
        if match:
            return [f"cc-{match[1]}-{match[2]}"]
    if "creativecommons.org/publicdomain/zero/1.0" in lowered:
        return ["CC0-1.0"]
    return ["custom"]


def _restrictive_lines(text: str) -> list[tuple[int, str]]:
    return [
        (i, line)
        for i, line in enumerate(text.splitlines(), 1)
        if RESTRICTIVE_TERMS.search(line)
        and not re.search(
            r"(?:commercial (?:or|and) non-commercial|non-commercial (?:or|and) commercial) uses", line, re.IGNORECASE
        )
    ]


def license_evidence(url: str, content: bytes, kind: str) -> LicenseEvidence:
    """Keep content hashes and bounded quotations, including restrictive clauses."""
    text = content.decode("utf-8")
    lines = text.splitlines()
    selected = [f"L{i}: {line[:500]}" for i, line in _restrictive_lines(text)]
    selected += [f"L{i}: {line[:500]}" for i, line in enumerate(lines, 1) if line.strip()][:12]
    return LicenseEvidence(
        url=url,
        kind=kind,
        sha256=hashlib.sha256(content).hexdigest(),
        license_ids=_license_ids(text, kind),
        quoted_lines=list(dict.fromkeys(selected)) or ["(empty file)"],
        restrictive_terms=bool(_restrictive_lines(text)),
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


def resolve_license(target: LicenseTarget) -> LicenseReceipt:
    """Read HF card and license files, GitHub pinned license, or an explicit terms URL.

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
        for filename in filenames:
            kind = "dataset_card" if filename == "README.md" else "license_file"
            if kind == "license_file" and not LICENSE_FILENAME.match(Path(filename).name):
                continue
            url = f"https://huggingface.co/datasets/{target.dataset_id}/resolve/{target.revision}/{quote(filename)}"
            content = _fetch(url, huggingface=True)
            documents.append(content.decode("utf-8"))
            evidence.append(license_evidence(url, content, kind))
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
        # Current API metadata is evidence, never authority for the pinned verdict.
        for path, document, kind in (
            (f"repos/{repo}/git/trees/{target.revision}?recursive=1", tree, "repository_tree"),
            (f"repos/{repo}", _github_json(f"repos/{repo}"), "github_license_api"),
        ):
            metadata = json.dumps(document, sort_keys=True).encode()
            item = license_evidence(f"https://api.github.com/{path}", metadata, kind)
            evidence.append(item.model_copy(update={"license_ids": [], "restrictive_terms": False}))
        for entry in tree["tree"]:
            if entry["type"] != "blob" or not LICENSE_FILENAME.match(Path(entry["path"]).name):
                continue
            if entry["size"] > MAX_EVIDENCE_BYTES:
                raise ValueError(f"License evidence exceeds size limit: {entry['path']}")
            blob = _github_json(f"repos/{repo}/git/blobs/{entry['sha']}")
            content = base64.b64decode(blob["content"])
            url = f"https://github.com/{repo}/blob/{target.revision}/{quote(entry['path'])}"
            documents.append(content.decode("utf-8"))
            evidence.append(license_evidence(url, content, "license_file"))
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
