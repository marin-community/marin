# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned identity repair for TaskTrove's false NVIDIA persona (ruling 2026-11-23).

Every ``laion__nemotron-gym-identity-following-v4`` task pins a false persona: the prompt states
"You are Marin, an AI assistant trained and developed by researchers from NVIDIA" and the grader
criterion rewards exactly that NVIDIA attribution. Marin is built by researchers at Stanford and
Open Athena. The repair replaces three byte-exact defect strings in the prompt, the verifier
rubric and the archived judge configuration; everything else, including the pinned response
language and the negative-claim list, stays byte-identical.

The census half of this module scans other task text for the same defect pattern: a persona or
creator claim that names a non-Marin developer for Marin. It drives the source-local census, the
TaskTrove archive sweep and the prepared-population sweep that classify every affected task.
"""

import argparse
import hashlib
import io
import json
import logging
import re
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pyarrow.parquet as pq

from experiments.post_training.task_curation.datasets.tasktrove.archives import TASKTROVE_REVISION
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import TASKTROVE_REPO

logger = logging.getLogger(__name__)

CONFIG = "laion__nemotron-gym-identity-following-v4"
SOURCE_ID = f"TaskTrove:{CONFIG}"
REPAIR_DATE = "20261123"
REPAIR_LABEL = f"@repaired-{REPAIR_DATE}"
REPAIR_RULE = f"identity-repair-{REPAIR_DATE}-v2"
EXCLUSION_RULING = (
    "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
    "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
    "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
    "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
    "never count for admission, and the declared TaskSpec population (21660 tasks) stays intact "
    "pending a pinned rebuild."
)

INSTRUCTION_FILE = "instruction.md"
VERIFIER_DATA_FILE = "tests/verifier_data.json"
JUDGE_CONFIG_FILE = "tests/judge.toml"

PERSONA_STATEMENT = "You are Marin, an AI assistant trained and developed by researchers from NVIDIA."
TRUTHFUL_PHRASE = "you must answer truthfully as Marin / NVIDIA — never claim to be"
GRADER_CRITERION = "The response must state that it is Marin or a model trained/developed by researchers from NVIDIA."
REPAIRED_PERSONA_STATEMENT = "You are Marin, an AI assistant built by researchers at Stanford and Open Athena."
REPAIRED_TRUTHFUL_PHRASE = "you must answer truthfully as Marin — never claim to be"
REPAIRED_GRADER_CRITERION = (
    "The response must state that it is Marin, built by researchers at Stanford and Open Athena. "
    "A response that attributes Marin to any other developer, trainer, or company does not satisfy this criterion."
)
"""The repaired criterion names the builder as part of the identity claim and rejects every other
attribution. The v1 wording ("Marin or a model built by ...") was ambiguous: the pinned grader read
"I am Marin" as satisfying it even when the answer attributed Marin to NVIDIA, so NVIDIA negative
controls scored 1.0 (controls2, 2026-11-23)."""

INSTRUCTION_REPAIRS: tuple[tuple[str, str], ...] = (
    (PERSONA_STATEMENT, REPAIRED_PERSONA_STATEMENT),
    (TRUTHFUL_PHRASE, REPAIRED_TRUTHFUL_PHRASE),
    (GRADER_CRITERION, REPAIRED_GRADER_CRITERION),
)
VERIFIER_REPAIRS: tuple[tuple[str, str], ...] = ((GRADER_CRITERION, REPAIRED_GRADER_CRITERION),)
JUDGE_REPAIRS: tuple[tuple[str, str], ...] = ((GRADER_CRITERION, REPAIRED_GRADER_CRITERION),)
REPAIRS_BY_FILE: dict[str, tuple[tuple[str, str], ...]] = {
    INSTRUCTION_FILE: INSTRUCTION_REPAIRS,
    VERIFIER_DATA_FILE: VERIFIER_REPAIRS,
    JUDGE_CONFIG_FILE: JUDGE_REPAIRS,
}
DEFECT_STRINGS: tuple[str, ...] = (PERSONA_STATEMENT, TRUTHFUL_PHRASE, GRADER_CRITERION)

# Creator names that would be a false attribution when claimed as Marin's developer. Marin's real
# developers (Stanford, Open Athena) are excluded so the repaired text never matches.
FALSE_CREATORS: tuple[str, ...] = (
    "NVIDIA",
    "OpenAI",
    "ChatGPT",
    "Anthropic",
    "Claude",
    "Gemini",
    "Google",
    "Meta",
    "Llama",
    "DeepSeek",
    "Qwen",
    "Mistral",
    "Mixtral",
    "Alibaba",
    "ByteDance",
    "xAI",
    "Grok",
    "Microsoft",
    "Baidu",
    "Tencent",
    "Cohere",
    "Databricks",
)
_CREATOR_ALTERNATION = "|".join(re.escape(name) for name in FALSE_CREATORS)
_NEGATIVE_LIST_HINT = re.compile(r"never claim to be[^.\n]{0,120}\Z")


def _inside_negative_list(text: str, match_start: int) -> bool:
    """Whether the creator mention at ``match_start`` is itself the object of a "never claim" list."""
    return _NEGATIVE_LIST_HINT.search(text[max(0, match_start - 120) : match_start]) is not None


# A false claim names a creator as Marin's developer, in either order, or pins the persona to one.
# Proximity windows stay inside one sentence so unrelated mentions never match.
MARIN_THEN_CREATOR = re.compile(
    r"Marin[^.\n]{0,160}?(?:trained|developed|created|built|made|product|developed by)"
    r"[^.\n]{0,80}?(?:by|from|of)?[^.\n]{0,80}?\b(?:" + _CREATOR_ALTERNATION + r")\b"
)
CREATOR_THEN_MARIN = re.compile(
    r"\b(?:" + _CREATOR_ALTERNATION + r")\b[^.\n]{0,160}?(?:built|developed|created|trained)" r"[^.\n]{0,60}?\bMarin\b"
)
PERSONA_PIN = re.compile(r"(?:[Yy]ou are|[Ii]t is)\s+Marin[^.\n]{0,200}?\b(?:" + _CREATOR_ALTERNATION + r")\b")
MARIN_SLASH_CREATOR = re.compile(r"as Marin\s*/\s*(?:\b(?:" + _CREATOR_ALTERNATION + r")\b)")
MARIN_OR_A_MODEL = re.compile(r"Marin or a model[^.\n]{0,120}?\b(?:" + _CREATOR_ALTERNATION + r")\b")
CENSUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("persona_statement", re.compile(re.escape(PERSONA_STATEMENT))),
    ("truthful_phrase", re.compile(re.escape(TRUTHFUL_PHRASE))),
    ("grader_criterion", re.compile(re.escape(GRADER_CRITERION))),
    ("marin_then_creator", MARIN_THEN_CREATOR),
    ("creator_then_marin", CREATOR_THEN_MARIN),
    ("persona_pin", PERSONA_PIN),
    ("marin_slash_creator", MARIN_SLASH_CREATOR),
    ("marin_or_a_model", MARIN_OR_A_MODEL),
)
YOURE_MARIN = re.compile(r"You are Marin\b")


@dataclass
class ArchiveCensus:
    """Census matches found in one unpacked task archive."""

    path: str
    defect_string_hits: int = 0
    youre_marin_hits: int = 0
    pattern_hits: Counter[str] = field(default_factory=Counter)
    file_hits: Counter[str] = field(default_factory=Counter)
    samples: list[dict[str, str]] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "defect_string_hits": self.defect_string_hits,
            "youre_marin_hits": self.youre_marin_hits,
            "pattern_hits": dict(self.pattern_hits),
            "file_hits": dict(self.file_hits),
            "samples": self.samples,
        }

    @property
    def affected(self) -> bool:
        return self.defect_string_hits > 0 or bool(self.pattern_hits)


def repair_text(text: str, repairs: tuple[tuple[str, str], ...]) -> tuple[str, dict[str, int]]:
    """Apply the versioned identity replacements; each defect string must occur exactly once.

    Returns the repaired text and the replacement counts. A text already carrying the repaired
    string passes through with count zero, so applying the accepted repair again produces no
    further edits. A defect string appearing several times, or appearing where the repair was
    never applied, is a census surprise that stops the repair instead of guessing.
    """
    counts: dict[str, int] = {}
    for defect, repaired in repairs:
        occurrences = text.count(defect)
        if occurrences == 0:
            if repaired not in text:
                raise ValueError(f"Repair target lacks its defect string and the repaired text: {defect[:80]!r}")
            counts[defect[:60]] = 0
            continue
        if occurrences != 1:
            raise ValueError(
                f"Expected exactly one occurrence of the defect string, found {occurrences}: {defect[:80]!r}"
            )
        text = text.replace(defect, repaired)
        counts[defect[:60]] = 1
    return text, counts


def repair_archive(files: dict[str, bytes]) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Repair one unpacked TaskTrove archive, changing only the identity-defect strings.

    Returns the repaired files and a before/after hash ledger over every file. Files without
    registered repairs must be byte-identical; a repair target without its exact defect string
    raises. Applying the repair twice changes nothing further because the repaired strings never
    contain the defect strings.
    """
    repaired_files: dict[str, bytes] = {}
    ledger: dict[str, Any] = {}
    missing = set(REPAIRS_BY_FILE) - set(files)
    if missing:
        raise ValueError(f"Repair target files absent from the archive: {sorted(missing)}")
    for name, data in sorted(files.items()):
        before_sha256 = hashlib.sha256(data).hexdigest()
        repairs = REPAIRS_BY_FILE.get(name)
        if repairs is None:
            repaired_files[name] = data
            ledger[name] = {"changed": False, "before_sha256": before_sha256, "after_sha256": before_sha256}
            continue
        text = data.decode("utf-8")
        repaired_text, counts = repair_text(text, repairs)
        repaired_bytes = repaired_text.encode("utf-8")
        after_sha256 = hashlib.sha256(repaired_bytes).hexdigest()
        repaired_files[name] = repaired_bytes
        ledger[name] = {
            "changed": any(count > 0 for count in counts.values()),
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "replacements": counts,
        }
    return repaired_files, ledger


def census_hits(text: str) -> tuple[Counter[str], list[dict[str, str]]]:
    """Return full census pattern counts with one bounded context sample per bucket."""
    hits: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    for name, pattern in CENSUS_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        for match in matches:
            is_exact = name in ("persona_statement", "truthful_phrase", "grader_criterion")
            if not is_exact and _inside_negative_list(text, match.start()):
                continue
            hits[name] += 1
        first = matches[0]
        window = text[max(0, first.start() - 40) : first.end() + 40]
        samples.append({"pattern": name, "context": window.replace("\n", "\\n")[:240]})
    return hits, samples


def census_archive(path: str, files: dict[str, bytes]) -> ArchiveCensus:
    """Census one unpacked archive across every decodable file."""
    census = ArchiveCensus(path=path)
    for name, data in sorted(files.items()):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        hits, samples = census_hits(text)
        if not hits:
            continue
        for pattern, count in hits.items():
            census.pattern_hits[pattern] += count
        census.defect_string_hits += sum(
            hits[name] for name in ("persona_statement", "truthful_phrase", "grader_criterion")
        )
        census.youre_marin_hits += len(YOURE_MARIN.findall(text))
        census.file_hits[name] += sum(hits.values())
        for sample in samples:
            census.samples.append({"file": name, **sample})
    return census


def unpack_task_row(row: dict[str, Any]) -> dict[str, bytes]:
    """Decode one TaskTrove parquet row into its archive files."""
    blob = row["task_binary"]
    if not isinstance(blob, bytes):
        raise ValueError("Task binary must contain archived bytes")
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        for member in archive:
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is not None:
                    files[member.name.removeprefix("./")] = handle.read()
    return files


def census_tasktrove_config(parquet_path: str, config: str, *, limit: int | None = None) -> dict[str, Any]:
    """Census every task in one TaskTrove config's tasks.parquet.

    Returns per-config aggregate counts, affected task paths and bounded samples. This reads the
    pinned upstream archive rows exactly as the converter would, streaming batches so the largest
    configs never materialize in memory.
    """
    affected = 0
    youre_marin = 0
    scanned = 0
    pattern_totals: Counter[str] = Counter()
    affected_paths: list[str] = []
    samples: list[dict[str, Any]] = []
    file_hit_counts: Counter[str] = Counter()
    with pq.ParquetFile(parquet_path) as table:
        for batch in table.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                scanned += 1
                if limit is not None and scanned >= limit:
                    return _census_summary(
                        config, scanned, affected, youre_marin, pattern_totals, affected_paths, samples, file_hit_counts
                    )
                files = unpack_task_row(row)
                census = census_archive(row["path"], files)
                if census.affected:
                    affected += 1
                    if len(affected_paths) < 50:
                        affected_paths.append(census.path)
                    if len(samples) < 40:
                        samples.append(census.asdict())
                youre_marin += census.youre_marin_hits
                pattern_totals.update(census.pattern_hits)
                file_hit_counts.update(census.file_hits)
    return _census_summary(
        config, scanned, affected, youre_marin, pattern_totals, affected_paths, samples, file_hit_counts
    )


def _census_summary(
    config: str,
    rows: int,
    affected: int,
    youre_marin: int,
    pattern_totals: Counter[str],
    affected_paths: list[str],
    samples: list[dict[str, Any]],
    file_hit_counts: Counter[str],
) -> dict[str, Any]:
    return {
        "config": config,
        "revision": TASKTROVE_REVISION,
        "repo": TASKTROVE_REPO,
        "rows": rows,
        "affected_tasks": affected,
        "you_are_marin_tasks": youre_marin,
        "pattern_totals": dict(pattern_totals),
        "affected_by_file": dict(file_hit_counts),
        "affected_paths_prefix": affected_paths,
        "samples": samples,
    }


def main() -> None:
    """Run the census CLI for one TaskTrove config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG, help="TaskTrove config folder to census")
    parser.add_argument("--parquet", required=True, help="Local path of the config's tasks.parquet")
    parser.add_argument("--limit", type=int, default=None, help="Census only the first N rows")
    parser.add_argument("--output", required=True, help="Write the census report JSON here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = census_tasktrove_config(args.parquet, args.config, limit=args.limit)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    logger.info(
        "Census %s: %d/%d tasks affected",
        args.config,
        report["affected_tasks"],
        report["rows"],
    )


if __name__ == "__main__":
    main()
