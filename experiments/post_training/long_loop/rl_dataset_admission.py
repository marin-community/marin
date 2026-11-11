# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize pinned Smartie packages and exclude exact evaluation identities.

Run in SkyRL's existing CPU environment. This does not execute task setup,
solutions or verifiers, and does not certify semantic decontamination.
"""

import argparse
import hashlib
import io
import json
import re
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from taskcompendium.models import Source, TaskSpec

RELEASES = {
    "swesmith": ("0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c", 8045),
    "calibforge": ("6a6429ccba098e11ea9f6297a70cb43a34670c0f", 2205),
}
SCHEMA = pa.schema([pa.field("task_spec", pa.string(), nullable=False)])


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_digest(text: str) -> str:
    return digest(" ".join(text.split()).encode())


def repository_ids(text: str) -> set[str]:
    """Recognize explicit GitHub repos and SWE instance names, not semantic similarity."""
    repos = set()
    for owner, name in re.findall(r"github\.com/([\w.-]+)/([\w.-]+)", text):
        name = name.removesuffix(".git")
        if owner.lower() == "swesmith" and "__" in name:
            owner, name = name.split("__", 1)
            name = name.split(".")[0]
        repos.add(f"{owner}/{name}".lower())
    for owner, name in re.findall(r"\b([A-Za-z0-9_.-]+)__([A-Za-z0-9_.-]+)-\d+\b", text):
        repos.add(f"{owner}/{name}".lower())
    return repos


def task_text(task: TaskSpec) -> str:
    return "\n".join(event.content for event in task.context.events if isinstance(event.content, str))


def identity_keys(identifier: str) -> set[str]:
    keys = {identifier, identifier.rsplit("/", 1)[-1]}
    return {key.lower() for key in keys if len(key) >= 8}


def exclusion_index(micro_rows: list[dict], policies: list[dict], legacy_heldout: list[dict], upstream: dict) -> dict:
    ids = set()
    repos = set()
    texts = set()
    records = []
    for row in micro_rows:
        # Read frozen historical identities without migrating their TaskSpec schema.
        task = json.loads(row["task_spec"])
        text = "\n".join(
            event["content"] for event in task["context"]["events"] if isinstance(event.get("content"), str)
        )
        row_ids = identity_keys(task["id"]) | identity_keys(row["source_task_id"])
        row_repos = (
            repository_ids(row["source_task_id"] + "\n" + text) if row["benchmark"] == "swebench-verified" else set()
        )
        ids.update(row_ids)
        repos.update(row_repos)
        texts.add(text_digest(text))
        records.append(
            {
                "kind": "micro210",
                "id": task["id"],
                "benchmark": row["benchmark"],
                "source": task["source"],
                "identity_keys": sorted(row_ids),
                "repos": sorted(row_repos),
                "task_sha256": digest(row["task_spec"].encode()),
            }
        )
    for row in policies:
        ids.update(identity_keys(row["task_id"]) | identity_keys(row["source_task_id"]))
        if row["benchmark"] == "swebench-verified":
            repos.update(repository_ids(row["source_task_id"]))
        records.append(
            {
                "kind": "policy-index",
                "id": row["task_id"],
                "benchmark": row["benchmark"],
                "source": row["policy"].get("source"),
                "instruction_sha256": row["policy"].get("file_sha256", {}).get("instruction.md"),
            }
        )
    for benchmark, universe in upstream.items():
        for identifier in universe.get("task_ids", []):
            ids.update(identity_keys(identifier))
            if benchmark == "swebench-verified":
                repos.update(repository_ids(identifier))
        records.append({"kind": "full-upstream-task-universe", "benchmark": benchmark, **universe})
    for row in legacy_heldout:
        task = TaskSpec.model_validate_json(row["task_spec"])
        ids.update(identity_keys(task.id))
        texts.add(text_digest(task_text(task)))
        records.append({"kind": "legacy-rl-heldout", "id": task.id, "source": task.source.model_dump()})
    return {"ids": sorted(ids), "repos": sorted(repos), "text_sha256": sorted(texts), "records": records}


def question_anchors(records: list[dict]) -> dict[tuple[str, ...], list[str]]:
    """Index exact upstream questions by eight-word prefixes for bounded scans."""
    anchors = {}
    for record in records:
        for text in record.get("texts", {}).values():
            words = text.split()
            if len(words) >= 8 and len(text) >= 40:
                normalized = " ".join(words)
                anchors.setdefault(tuple(words[:8]), []).append(normalized)
    return anchors


def matching_question(text: str, anchors: dict[tuple[str, ...], list[str]]) -> bool:
    words = text.split()
    normalized = " ".join(words)
    return any(
        question in normalized
        for start in range(max(0, len(words) - 7))
        for question in anchors.get(tuple(words[start : start + 8]), ())
    )


def exclusion_reasons(identifier: str, text: str, index: dict, *, check_repo: bool) -> list[str]:
    reasons = []
    if identity_keys(identifier).intersection(index["ids"]):
        reasons.append("exact_or_basename_task_id")
    if text_digest(text) in index["text_sha256"]:
        reasons.append("whitespace_normalized_instruction_sha256")
    if check_repo and repository_ids(text).intersection(index["repos"]):
        reasons.append("swebench_repository_overlap")
    return reasons


def archived_task(row: dict, *, cohort: str, revision: str) -> TaskSpec:
    """Return a source-pinned TaskSpec preserving the archived task package."""
    # SkyRL is an optional runtime dependency, needed only during normalization.
    from skyrl_train.dataset.harbor_import import harbor_trial_task  # noqa: PLC0415 - optional SkyRL runtime

    with tempfile.TemporaryDirectory(prefix="long-loop-task-") as temporary:
        directory = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:*") as archive:
            archive.extractall(directory, filter="data")
        source = Source(
            dataset=f"long-loop-smartie-{cohort}",
            revision=revision,
            row=f"{row['source']}/{row['path']}",
            importer_revision="marin-long-loop-harbor-trial-v1",
        )
        return harbor_trial_task(directory, source=source)


def write_rows(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression="zstd")


def build(inputs: Path, output: Path, *, seed: int, validation_per_cohort: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    micro = pq.read_table(inputs / "micro210.parquet").to_pylist()
    policies = [json.loads(line) for line in (inputs / "policy-index.jsonl").read_text().splitlines()]
    legacy = pq.read_table(inputs / "nonagentic-full-validation.parquet").to_pylist()
    upstream = json.loads((inputs / "upstream-git-task-universes.json").read_text())
    index = exclusion_index(micro, policies, legacy, upstream)
    (output / "evaluation-exclusions.json").write_text(json.dumps(index, indent=2) + "\n")
    for key in ("ids", "repos", "text_sha256"):
        index[key] = set(index[key])
    upstream_questions = []
    universe_directory = inputs / "hf-universes"
    if universe_directory.is_dir():
        for path in sorted(universe_directory.glob("*.json")):
            if path.name not in {"manifest.json", "mrcr.json", "NUPA.json"}:
                upstream_questions.extend(json.loads(path.read_text()))
    anchors = question_anchors(upstream_questions)
    evidence = []
    totals = {}
    routes = {}
    validation_rows = pq.read_table(inputs / "nonagentic-validation.parquet").to_pylist()
    for cohort, (revision, expected_rows) in RELEASES.items():
        path = inputs / cohort / "tasks.parquet"
        rows = pq.read_table(path).to_pylist()
        if len(rows) != expected_rows:
            raise ValueError(f"Unexpected cohort size: {cohort}")
        source_manifest = json.loads((inputs / cohort / "manifest.json").read_text())
        declared = source_manifest.get("outputs_sha256", {}).get("tasks.parquet")
        if declared is not None and declared != file_digest(path):
            raise ValueError(f"Source manifest digest mismatch: {cohort}")
        eligible = []
        for ordinal, row in enumerate(rows):
            with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:*") as archive:
                text = archive.extractfile("instruction.md").read().decode()
            identifier = f"{row['source']}/{row['path']}"
            reasons = exclusion_reasons(identifier, text, index, check_repo=cohort == "swesmith")
            if matching_question(text, anchors):
                reasons.append("exact_upstream_question_substring")
            record = {
                "cohort": cohort,
                "source_revision": revision,
                "source_row": ordinal,
                "id": identifier,
                "archive_sha256": digest(row["task_binary"]),
                "instruction_sha256": text_digest(text),
                "repos": sorted(repository_ids(text)),
                "exclusion_reasons": reasons,
            }
            evidence.append(record)
            if not reasons:
                eligible.append((digest(f"{seed}:{identifier}".encode()), row, record))
        eligible.sort(key=lambda item: item[0])
        if len(eligible) <= validation_per_cohort:
            raise ValueError(f"Insufficient eligible tasks: {cohort}")
        train_rows = []
        failures = []
        for position, (_, row, record) in enumerate(eligible):
            try:
                task = archived_task(row, cohort=cohort, revision=revision)
            except (ValueError, tarfile.TarError, OSError) as error:
                record.update(disposition="normalization_failed", error=str(error))
                failures.append(record["id"])
                continue
            normalized = {"task_spec": task.model_dump_json()}
            record["normalized_task_sha256"] = digest(normalized["task_spec"].encode())
            record["task_id"] = task.id
            if position < validation_per_cohort:
                record["disposition"] = "validation"
                validation_rows.append(normalized)
            else:
                record["disposition"] = "train"
                train_rows.append(normalized)
        write_rows(output / f"{cohort}-train.parquet", train_rows)
        totals[cohort] = {
            "source": expected_rows,
            "overlap_excluded": expected_rows - len(eligible),
            "train": len(train_rows),
            "validation": sum(r.get("disposition") == "validation" for r in evidence if r["cohort"] == cohort),
            "normalization_failures": failures,
        }
        routes[f"long-loop-smartie-{cohort}"] = "agentic"
    nonagentic_rows = []
    excluded_nonagentic = 0
    nonagentic_ordinal = 0
    for batch in pq.ParquetFile(inputs / "nonagentic-train.parquet").iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            task = TaskSpec.model_validate_json(row["task_spec"])
            reasons = exclusion_reasons(task.id, task_text(task), index, check_repo=False)
            if matching_question(task_text(task), anchors):
                reasons.append("exact_upstream_question_substring")
            evidence.append(
                {
                    "cohort": "nonagentic",
                    "id": task.id,
                    "source_row": nonagentic_ordinal,
                    "source": task.source.model_dump(),
                    "exclusion_reasons": reasons,
                    "normalized_task_sha256": digest(row["task_spec"].encode()),
                    "disposition": "excluded" if reasons else "train",
                }
            )
            nonagentic_ordinal += 1
            if reasons:
                excluded_nonagentic += 1
            else:
                nonagentic_rows.append(row)
                routes[task.source.dataset] = "nonagentic"
    write_rows(output / "nonagentic-train.parquet", nonagentic_rows)
    write_rows(output / "validation.parquet", validation_rows)
    (output / "task-manifest.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in evidence) + "\n")
    summary = {
        "seed": seed,
        "validation_per_agentic_cohort": validation_per_cohort,
        "cohorts": totals,
        "nonagentic_train": len(nonagentic_rows),
        "nonagentic_excluded": excluded_nonagentic,
        "validation_rows": len(validation_rows),
        "source_routes": routes,
        "consumed_route_quotas": {"agentic": 32, "nonagentic": 32},
        "full_upstream_task_universes": {b: u.get("count") for b, u in upstream.items()},
        "micro210_rows": len(micro),
        "policy_index_rows": len(policies),
        "legacy_heldout_rows": len(legacy),
        "upstream_question_records": len(upstream_questions),
        "benchmark_coverage": dict(Counter(row["benchmark"] for row in micro)),
        "admission": "blocked",
        "admission_pending": [
            "Complete all benchmark universes and historical identity",
            "Qualify Harbor runtime and CP4 learner",
        ],
        "limitations": [
            "No semantic or paraphrase decontamination",
            "Pinned agentic task universes covered; historical nonagentic universe coverage remains incomplete",
            "SWE repository exclusion is conservative; absent aliases and indirect ancestry may escape",
            "CalibForge2205 has text-quality admission only; original oracle waiver remains",
        ],
        "files": {
            path.name: {"sha256": file_digest(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validation-per-cohort", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.inputs, args.output, seed=args.seed, validation_per_cohort=args.validation_per_cohort), indent=2
        )
    )


if __name__ == "__main__":
    main()
