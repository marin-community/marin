# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""License targets and the primary-evidence parsing the resolver and checker share."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from finestore.rl.licenses import LicenseEvidence

MAX_EVIDENCE_BYTES = 1024 * 1024
HTTP_TIMEOUT = 30
LICENSE_FILENAME = re.compile(r"^(licen[cs]e|copying|terms)([._-].*)?$", re.IGNORECASE)
DEPENDENCY_DIRECTORIES = frozenset({"node_modules", "vendor", ".venv", "venv", "site-packages", ".git"})
TASK_MARKERS = frozenset({"task.toml", "task.yaml", "task.yml", "task.json", "instruction.md", "instructions.md"})
TASK_CONTENT_LICENSE = re.compile(r"^(?:licen[cs]e|copying|terms)[._-](?:content|data|tasks?)(?:[._-]|$)", re.I)
LICENSE_SECTION_HEADINGS = frozenset(
    {
        "license",
        "licence",
        "licensing",
        "licensing information",
        "copyright",
        "terms of use",
        "terms and conditions",
    }
)
RESTRICTIVE_TERMS = re.compile(
    r"\bcc[- ]by[- ](?:nc|nd)\b|non[- ]?commercial|non[- ]?derivatives|research[- ]only|"
    r"no[- ](?:commercial|derivatives|model[- ]training)|not (?:be )?(?:used|use).{0,80}"
    r"(?:train|redistribut)|(?:train|redistribut).{0,80}(?:prohibited|forbidden|not permitted)",
    re.IGNORECASE,
)
MODEL_OUTPUT_TERMS = re.compile(
    r"synthetic|model.generated|generated (?:by|with|using)|gpt[- ]?[345]|claude", re.IGNORECASE
)
THIRD_PARTY_TERMS = re.compile(r"third.party|bundled|mixture|compilation|aggregat(?:e|ed|ion)", re.IGNORECASE)
WRAPPER_BOILERPLATE = (
    "Upstream licenses and third-party asset terms continue to apply.",
    "Upstream licenses and third-party asset terms apply; this conversion grants no new rights.",
)
WRAPPER_NAME = re.compile(r"retained|filtered|taskcompendium|oracle|quick|rebuilt|repaired", re.I)
WRAPPER_TITLE = re.compile(r"taskcompendium|retention release|QUICK inspection release", re.I)
SOURCE_PIN = re.compile(r"^Source pin:\s*`([^`@]+)@([0-9a-f]+)`", re.M | re.I)
NATIVE_SOURCE_PIN = re.compile(r"^Upstream:\s*(\S+) at ([0-9a-f]+)\.\s+(https://\S+)", re.M | re.I)
ARCHIVE_SOURCE_PIN = re.compile(r"archive `([^`]+)`\s*@\s*`([0-9a-f]+)`", re.I)
REPAIR_SOURCE_PIN = re.compile(r"\bfrom `([^`@]+/[^`@]+)@([0-9a-f]+)`", re.I)
WRAPPER_MANIFESTS = {
    "selection-manifest.json": ("original_selection_plan", "prior_dedup_and_lexical_selection", "original_release"),
    "manifest.json": ("source_plan_before_oracle_filter/plan",),
    "filtered-source.json": ("prior_retention_plan",),
}
SEPARATE_TERMS = re.compile(
    r"(?:subject to|governed by|under|respect).{0,80}(?:separate|third.party|upstream|respective).{0,80}"
    r"(?:terms|licen[cs]es)|(?:separate|third.party|upstream|respective).{0,50}(?:terms|licen[cs]es).{0,50}(?:apply|govern)",
    re.IGNORECASE,
)
CC_LICENSE_TITLE = re.compile(
    r"\b(?:cc[- ]by|(?:creative commons )?attribution)"
    r"(?P<conditions>(?:[- ](?:nc|nd|sa|noncommercial|noderivatives|sharealike))*)"
    r"[- ]+(?P<version>[1-4]\.\d)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LicenseTarget:
    source_id: str
    dataset_id: str
    revision: str
    url: str
    license_url: str | None = None


def _card_metadata(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Dataset-card YAML is not terminated")
    metadata = yaml.safe_load(parts[1]) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Dataset-card YAML must contain a mapping")
    return metadata


def _license_ids(text: str, kind: str) -> list[str]:
    if kind == "dataset_card":
        metadata = _card_metadata(text)
        value = metadata.get("license")
        if value is None:
            return ["missing"]
        if value == "other" and _license_alias(str(metadata.get("license_name", ""))) == "cc0-1.0":
            return ["cc0-1.0"]
        values = [str(item) for item in value] if isinstance(value, list) else [str(value)]
        return [_license_alias(value) for value in values]
    if kind == "package_metadata":
        return [json.loads(text)["info"].get("license_expression") or "missing"]
    if kind in {"repository_tree", "github_license_api", "readme_notice"}:
        return []
    spdx = re.findall(r"SPDX-License-Identifier:\s*([^\n]+)", text)
    if spdx:
        return [_license_alias(value.strip()) for value in spdx]
    lowered = text.lower()
    if cc := CC_LICENSE_TITLE.search(text):
        conditions = cc["conditions"].lower()
        for word, abbreviation in (("noncommercial", "nc"), ("noderivatives", "nd"), ("sharealike", "sa")):
            conditions = conditions.replace(word, abbreviation)
        suffix = conditions.replace(" ", "-")
        return [f"cc-by{suffix}-{cc['version']}"]
    if re.search(r"\bodc[- ]by\b|open data commons attribution licen[cs]e", lowered):
        return ["odc-by"]
    if match := re.search(
        r"OpenMDW License Agreement,\s*version\s*([0-9]+\.[0-9]+)\s*\(OpenMDW-([0-9]+\.[0-9]+)\)", text
    ):
        return [f"openmdw-{match.group(2)}"]
    if re.search(r"\bcc0(?:[- ]1\.0)?\b", lowered) or "creativecommons.org/publicdomain/zero/1.0" in lowered:
        return ["CC0-1.0"]
    if re.search(r"\bapache(?: license)?[- ,]*(?:version )?2\.0\b|apache.org/licenses/license-2.0", lowered):
        return ["Apache-2.0"]
    if "permission is hereby granted, free of charge" in lowered:
        return ["MIT"]
    if re.search(r"\bmit licen[cs]e(?:d)?\b|\blicen[cs]ed under (?:the )?mit\b", lowered):
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
    if not text.strip():
        return ["missing"]
    return ["custom"]


def _license_alias(value: str) -> str:
    if value.strip().lower() in {
        "cc0",
        "cc0-with-notice",
        "cc0-1.0-with-notice",
        "cc0 with notice",
        "cc0-with-bytedance-notice",
    }:
        return "cc0-1.0"
    return value


def _readme_license_lines(text: str) -> list[tuple[int, str]]:
    selected = []
    section = False
    frontmatter = text.startswith("---")
    fenced = False
    for index, line in enumerate(text.splitlines(), 1):
        if frontmatter:
            if index > 1 and line.strip() == "---":
                frontmatter = False
            continue
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        if fenced and not section:
            continue
        if line.lstrip().startswith("#"):
            section = bool(re.search(r"\blicen[cs]e|\blicensing|terms (?:of use|and conditions)|copyright", line, re.I))
        statement = re.search(
            r"\b(?:licenses|licensed|licenced|released|distributed|available)\b.{0,50}\b(?:under|license|licence|terms)\b"
            r"|^\s*licen[cs]e\s*(?::|mirrors)",
            line,
            re.I,
        )
        if section or statement:
            selected.append((index, line))
    return selected


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
    scoped_text = _license_context(url, text)
    lines = list(enumerate(scoped_text.splitlines(), 1))
    relevant = lines
    unidentified = []
    if kind == "dataset_card":
        closing = next((i for i, line in lines[1:] if line.strip() == "---"), 0) if text.startswith("---") else 0
        relevant = [(i, line) for i, line in lines if i <= closing]
    elif kind == "readme_license":
        relevant = [
            (i, line)
            for i, line in _readme_license_lines(scoped_text)
            if line.strip().lstrip("#").strip().lower() not in LICENSE_SECTION_HEADINGS
            and not re.search(r"more information needed|^\s*(?:TODO|TBD)\b", line, re.I)
        ]
    elif kind == "readme_notice":
        unidentified = [
            (i, line)
            for i, line in lines
            if SEPARATE_TERMS.search(line)
            and not re.search(r"https?://", line)
            and _license_ids(line, "license_file") == ["custom"]
        ]
        relevant = unidentified
    relevant_text = "\n".join(line for _, line in relevant)
    restrictions = _restrictive_lines(relevant_text) if kind != "readme_notice" else []
    restricted_content = {line for _, line in restrictions}
    selected = [f"L{i}: {line[:500]}" for i, line in relevant if line in restricted_content]
    selected += [f"L{i}: {line[:500]}" for i, line in relevant if re.search(r"licen[cs]|cc[- ]?by|cc0|odc", line, re.I)][
        :12
    ]
    selected += [f"L{i}: {line[:500]}" for i, line in relevant if line.strip()][:12]
    if kind == "package_metadata":
        selected = [f"JSON /info/license_expression: {json.dumps(json.loads(text)['info'].get('license_expression'))}"]
    return LicenseEvidence(
        url=url,
        kind=kind,
        sha256=hashlib.sha256(content).hexdigest(),
        license_ids=_license_ids(text if kind in {"dataset_card", "package_metadata"} else relevant_text, kind),
        quoted_lines=list(dict.fromkeys(selected))
        or ["(no license statement in this document)" if text else "(empty file)"],
        restrictive_terms=bool(restrictions),
        unidentified_third_party_terms=bool(unidentified),
    )


def _license_files(filenames: list[str]) -> set[str]:
    task_roots = {Path(name).parent for name in filenames if Path(name).name in TASK_MARKERS}
    return {
        name
        for name in filenames
        if LICENSE_FILENAME.match(Path(name).name)
        and not DEPENDENCY_DIRECTORIES.intersection(Path(name).parts)
        and (
            Path(name).parent == Path(".")
            or Path(name).parent in task_roots
            or TASK_CONTENT_LICENSE.match(Path(name).name)
        )
    }


def _license_context(url: str, text: str) -> str:
    parts = urlsplit(url)
    if parts.hostname == "huggingface.co" and parts.path.startswith("/datasets/open-athena/"):
        for sentence in WRAPPER_BOILERPLATE:
            text = text.replace(sentence, "")
    return text
