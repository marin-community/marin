# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Numina row-quality report and task manifest from saved decisions.

The build reads three saved decision files from the public bucket
hf://buckets/flufcloud/marin_numina_autoinformalization (OpenMDW-1.1 and Apache-2.0
terms are in its README). Each file is pinned by SHA-256 in ``INPUTS`` and the build
stops if the bucket holds a different version:

- answer_disclosure_labels.csv: row_key and label columns.
- golden.csv: reviewer verdicts with numina_row, uuid, golden_control, cause and note.
- golden_gemini.csv: judge verdicts with row_key, golden_control, note, formal_sha256,
  judge_model and judge_spec_sha256.

Run from the repository root:

    module=experiments.post_training.task_curation.datasets.lean.autoinformalization.build_task_selection
    uv run python -m "$module" numina.parquet --prime-lists prime-lists --output-dir quality-output

numina.parquet is data/train-00000-of-00001.parquet at the pinned NuminaMath-LEAN
revision, and prime-lists holds Prime Intellect's unported.json.gz and
misformalized.json.gz. The build applies source exclusions, deduplication, split and
evaluation-overlap rules, and retains rows with a "no" disclosure label that both golden
controls pass. It writes numina_autoinformalization_row_quality_report.csv and tasks.json,
which lists the selected rows of each split and reproduces the checked-in file byte for
byte. Copy tasks.json beside this module to activate the selection. To use new decisions,
upload them to the bucket and update their digests in ``INPUTS``.
"""

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.pipeline.models import ImportRejection
from verifyit.spec import JudgePromptFormat, JudgeSpec

from experiments.post_training.task_curation.datasets.lean import numina
from experiments.post_training.task_curation.datasets.lean.autoinformalization import dataset as auto
from experiments.post_training.task_curation.datasets.lean.autoinformalization import deduplicate_and_split as dedup
from experiments.post_training.task_curation.datasets.lean.lean_problem_similarity import statement_key

DEDUP = json.loads(dedup.DEDUP_FILE.read_text())
SOURCE_SHA256 = DEDUP["input"]["sha256"]
CLUSTER_SPLITS, REMOVED_ROWS = dedup.cluster_splits(DEDUP)
EXCLUDED_TYPES = {"MCQ", "math-word-problem"}
VERDICT_CAUSES = {"PASS": "", "FAIL": "golden_failed", "MALFORMED": "judge_malformed"}
BUCKET = "hf://buckets/flufcloud/marin_numina_autoinformalization"
INPUTS = {
    "labels": ("answer_disclosure_labels.csv", "247d0ee459f6b5469cf9d10a22159e50112a3fcac8b866057933cd7dcd78c99a"),
    "reviewer": ("golden.csv", "60a87003e04e095f2e371ac6f93dd743914f36602be6634ec1a6e6fb1e575435"),
    "gemini": ("golden_gemini.csv", "522d1f6865099a80cafe2f9f31f711096ba09f430a025b3902f5f967eed71769"),
}
"""Each saved decision file in ``BUCKET``, with the SHA-256 the build requires."""

FIELDS = (
    "numina_row",
    "uuid",
    "split",
    "question_type",
    "golden_control",
    "golden_source",
    "golden_cause",
    "primary_verdict",
    "required_verdict",
    "required_note",
    "required_cause",
    "included",
    "cause",
    "note",
    "row_key",
    "disclosure",
    "reference",
    "formal",
    "formal_sha256",
    "judge_model",
    "judge_spec_sha256",
)


def read_input(name: str) -> list[dict[str, str]]:
    """Read one pinned decision file from the bucket, refusing any other version."""
    filename, digest = INPUTS[name]
    with StoragePath(f"{BUCKET}/{filename}").open("rb") as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{filename} in {BUCKET} does not match its pinned SHA-256")
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def read_labels(records: list[dict[str, str]]) -> dict[str, str]:
    labels = {}
    for record in records:
        key, label = record["row_key"], record["label"]
        if key in labels or label not in {"yes", "no", "uncertain", "processing_failure"}:
            raise ValueError(f"Duplicate key or invalid disclosure label at {key}")
        labels[key] = label
    return labels


def assigned_split(data: dict) -> str:
    """Return the row's split, or an empty string for invalid inputs and removed duplicates."""
    key = dedup.row_key(data)
    if key is None or key in REMOVED_ROWS:
        return ""
    return dedup.row_split(key, data["uuid"], CLUSTER_SPLITS)


def prepare_rows(parquet: Path, prime_dir: Path, labels: dict[str, str]) -> Iterator[dict]:
    if file_sha256(StoragePath(str(parquet))) != SOURCE_SHA256:
        raise ValueError("Unexpected Numina parquet digest")
    lists = {}
    for name, digest in numina.PRIME_LISTS.items():
        path = StoragePath(str(prime_dir / f"{name}.json.gz"))
        if file_sha256(path) != digest:
            raise ValueError(f"Unexpected Prime list digest: {name}")
        lists[name] = numina.prime_list(path)

    def listed(uuid: str) -> str | None:
        return next((name for name, values in lists.items() if uuid in values), None)

    for data in auto.indexed_rows(StoragePath(str(parquet)), None):
        key = dedup.row_key(data)
        index = str(data["_numina_index"])
        row: dict = dict.fromkeys(FIELDS, "")
        row.update(
            numina_row=index,
            uuid=data["uuid"],
            split=assigned_split(data),
            question_type=data.get("question_type") or "",
            row_key=key or "",
            golden_control="not_run",
        )
        if not numina.is_human_statement_row(data, None):
            row["cause"] = "not_human"
        elif row["question_type"] in EXCLUDED_TYPES:
            row["cause"] = "excluded_question_type"
        elif not row["split"]:
            row["cause"] = "duplicate_or_missing_input"
        else:
            contract = numina.statement_row_contract(data, listed)
            if isinstance(contract, ImportRejection):
                row["cause"] = contract.reason
            else:
                label = labels.get(key, "missing")
                row["disclosure"] = label
                if label == "no":
                    row["formal"] = auto.formal_source(contract.prefix, contract.statement)
                else:
                    row["cause"] = "answer_disclosed" if label == "yes" else f"disclosure_{label}"
                if row["formal"]:
                    row["reference"] = data["problem"].strip()
                    row["formal_sha256"] = hashlib.sha256(row["formal"].encode()).hexdigest()
        row["included"] = not bool(row["cause"])
        yield row


def source_rows(parquet: Path, prime_lists: Path, labels: dict[str, str]) -> list[dict]:
    """Return source decisions with eval overlap checks applied before either model runs."""
    rows = list(prepare_rows(parquet, prime_lists, labels))
    # Check against proof training statements even when their auto-informalization pair was filtered out.
    proof_types = set()
    for data in auto.indexed_rows(StoragePath(str(parquet)), None):
        if data["uuid"] in dedup.PROOF_TRAIN_UUIDS and numina.proof_key(data) in numina.PASSING_PROOFS:
            proof_types.add(statement_key(data["formal_statement"]))
    types = {r["row_key"]: statement_key(r["formal"] + " :=") for r in rows if r["formal"]}
    train_types = proof_types | {types[r["row_key"]] for r in rows if r["formal"] and r["split"] == "train"}
    for row in rows:
        if row["formal"] and row["split"] == "eval" and types[row["row_key"]] in train_types:
            row["cause"] = "final_statement_train_overlap"
    for row in rows:
        row["included"] = not bool(row["cause"])
    return rows


def golden_spec(formal: str, reference: str, max_output_tokens: int = auto.JUDGE_OUTPUT_TOKENS) -> JudgeSpec:
    spec = auto.judge_spec(formal, reference)
    return replace(spec, max_completion_tokens=max_output_tokens)


def spec_digest(spec: JudgeSpec) -> str:
    fields = asdict(spec)
    # The saved Numina verdicts predate the pairwise judge settings.
    if fields["verdict_patterns"] == ():
        del fields["verdict_patterns"]
    if fields["label_weights"] == {}:
        del fields["label_weights"]
    # They also record the JSON prompt as the template "{fields_json}", which sent the same message.
    if fields.pop("prompt_format") == JudgePromptFormat.JSON:
        fields["prompt_template"] = "{fields_json}"
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def eligible_for_golden(row: dict) -> bool:
    """Apply the disclosure decision and type filter."""
    return row["disclosure"] == "no" and row["question_type"] not in EXCLUDED_TYPES


def final_manifest(rows: Iterator[dict]) -> dict:
    tasks: dict[str, list[str]] = {"train": [], "eval": []}
    for row in rows:
        if row["cause"]:
            continue
        if not eligible_for_golden(row):
            raise ValueError("Invalid type or disclosure decision in final selection")
        if row["golden_control"] != "PASS":
            raise ValueError(f"Unfinished quality checks for row {row['numina_row']}")
        if row["formal_sha256"] != hashlib.sha256(row["formal"].encode()).hexdigest():
            raise ValueError("Changed formal statement")
        tasks[row["split"]].append(row["row_key"])
    selected = [key for keys in tasks.values() for key in keys]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Empty or duplicate final selection")
    return dict(tasks={split: sorted(keys) for split, keys in tasks.items()})


def apply_golden(rows: list[dict], records: list[dict[str, str]], from_reviewer: bool) -> list[dict]:
    """Return copies of the rows with one set of golden verdicts applied and validated."""
    rows = copy.deepcopy(rows)
    if from_reviewer:
        locators = {(row["numina_row"], row["uuid"]): row["row_key"] for row in rows}
        for result in records:
            locator = (result["numina_row"], result["uuid"])
            if locator not in locators:
                raise ValueError(f"Unknown golden CSV row {locator}")
            result["row_key"] = locators[locator]
    results = {}
    for result in records:
        key = result["row_key"]
        if key in results:
            raise ValueError(f"Duplicate golden result {key}")
        results[key] = result
    if results.keys() - {row["row_key"] for row in rows}:
        raise ValueError("Golden results contain unknown source rows")
    for row in rows:
        if row["cause"]:
            continue
        result = results.get(row["row_key"])
        if result is None:
            raise ValueError(f"Missing golden result for {row['row_key']}")
        if not from_reviewer and result["formal_sha256"] != row["formal_sha256"]:
            raise ValueError("Golden result belongs to a different formal statement")
        label = result["golden_control"]
        if label not in VERDICT_CAUSES:
            raise ValueError("Invalid golden verdict")
        # Check rejected results too, so a stale rubric cannot silently discard good rows.
        if not from_reviewer and not any(
            result["judge_spec_sha256"] == spec_digest(golden_spec(row["formal"], row["reference"], tokens))
            for tokens in (1024, 2048, auto.JUDGE_OUTPUT_TOKENS)
        ):
            raise ValueError("Stale golden control")
        row["golden_control"] = label
        row["note"] = result["note"]
        row["golden_source"] = "reviewer_csv" if from_reviewer else "gemini_batch"
        row["golden_cause"] = result.get("cause", "")
        if not from_reviewer:
            row["judge_model"] = result["judge_model"]
            row["judge_spec_sha256"] = result["judge_spec_sha256"]
    return rows


def build(parquet: Path, prime_lists: Path, output_dir: Path) -> dict:
    """Keep rows that pass both the reviewer and the Gemini golden controls."""
    rows = source_rows(parquet, prime_lists, read_labels(read_input("labels")))
    gemini_records = read_input("gemini")
    reviewer = apply_golden(rows, read_input("reviewer"), from_reviewer=True)
    gemini = apply_golden(rows, gemini_records, from_reviewer=False)
    for row, extra in zip(reviewer, gemini, strict=True):
        if not row["golden_source"]:  # Excluded before the golden controls.
            continue
        verdicts = (row["golden_control"], extra["golden_control"])
        label = "PASS" if verdicts == ("PASS", "PASS") else "FAIL" if "FAIL" in verdicts else "MALFORMED"
        row.update(
            primary_verdict=verdicts[0],
            required_verdict=verdicts[1],
            required_note=extra["note"],
            required_cause=extra["golden_cause"],
            golden_control=label,
            golden_source="intersection",
            included=label == "PASS",
            cause=VERDICT_CAUSES[label],
        )
    metadata = dict(
        dataset_revision=numina.NUMINA_REVISION,
        source_sha256=SOURCE_SHA256,
        labels_sha256=INPUTS["labels"][1],
        golden_format="intersection",
        golden_inputs=[
            dict(format="reviewer_csv", sha256=INPUTS["reviewer"][1]),
            dict(format="gemini_batch", sha256=INPUTS["gemini"][1]),
        ],
        judge_models=sorted({r["judge_model"] for r in gemini_records if r["judge_model"]}),
    )
    return write_artifacts(reviewer, metadata, output_dir)


def write_artifacts(rows: list[dict], metadata: dict, output_dir: Path) -> dict:
    """Write a validated selection and its report from the same rows."""
    manifest = {**final_manifest(iter(rows)), **metadata}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        report = Path(temporary) / "numina_autoinformalization_row_quality_report.csv"
        with report.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest["report_sha256"] = file_sha256(StoragePath(str(report)))
        target = Path(temporary) / "tasks.json"
        target.write_text(json.dumps(manifest, indent=1) + "\n")
        os.replace(report, output_dir / report.name)
        os.replace(target, output_dir / target.name)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--prime-lists", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.parquet, args.prime_lists, args.output_dir)
    print(json.dumps({split: len(keys) for split, keys in manifest["tasks"].items()}))


if __name__ == "__main__":
    main()
