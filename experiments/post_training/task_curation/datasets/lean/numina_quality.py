# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Numina row report and final manifest together in one offline run.

Run from the repository root. To download the pinned inputs:

    revision=51fa67f1f647ae1ecd81eef9f19306aa8a7b3a94
    hub=https://huggingface.co/datasets/project-numina/NuminaMath-LEAN
    curl -fL "$hub/resolve/$revision/data/train-00000-of-00001.parquet" -o numina.parquet
    prime=https://raw.githubusercontent.com/PrimeIntellect-ai/prime-envs
    prime_commit=15e4e9d2f7d52d8d725a2668e483cf51a067e5ba
    mkdir -p prime-lists
    for name in unported misformalized; do
        curl -fL "$prime/$prime_commit/environments/lean/numina/numina/data/$name.json.gz" -o "prime-lists/$name.json.gz"
    done

The three scripts each complete their own step in one invocation:

    lean=experiments.post_training.task_curation.datasets.lean
    uv run python -m "$lean.classify_answer_disclosure" numina.parquet \\
        --prime-lists prime-lists \\
        --output labels.jsonl
    uv run python -m "$lean.numina_golden_batch" numina.parquet \\
        --prime-lists prime-lists \\
        --labels labels.jsonl \\
        --output golden.jsonl
    uv run python -m "$lean.numina_quality" numina.parquet \\
        --prime-lists prime-lists \\
        --labels labels.jsonl \\
        --golden golden.jsonl \\
        --output-dir quality-output

The first two commands submit paid Gemini batches using an exported GEMINI_API_KEY, wait,
and collect results in the same invocation. They make one request per row with no automatic
retry. The golden judge defaults to 4,096 output tokens. --max-output-tokens chooses the cap
for a new run. Batch state lives beside the output in a directory ending .batch.
After an interruption, rerun the exact command to resume its job. If submission is uncertain,
the error identifies the state.json file to reconcile against the job in Google AI Studio.

Use --golden REVIEW.csv to rebuild from a supplied reviewer CSV instead. It must have
numina_row, uuid, golden_control, cause, and note columns. Rows are matched to the pinned
source by row index and UUID. Only PASS verdicts are retained, regardless of cause annotations.
The CSV hash is recorded as provenance; its unknown judge configuration is left blank.
Add --require-golden SECOND_RESULT_FILE to require PASS in both result files. Both individual
verdicts and notes remain in the report, with both file hashes recorded in the manifest.

This script reads the parquet, disclosure labels, and golden verdicts. It applies the shared
source exclusions, deduplication, split and eval-overlap rules, removes MCQs and word problems,
and retains only "no" disclosure labels and PASS golden controls. It writes
numina_row_quality_report.csv and numina_autoinformalization_tasks.json. Excluded rows have
included=False and a reason in cause. Missing golden results stop the build.

Saved disclosure labels default to numina_answer_disclosure_labels.jsonl beside this module.
Disclosure labels, golden verdicts, and the quality CSV are generated local files.
For the existing local run, golden results are saved under
artifacts/numina-answer-disclosure/numina_golden_controls.jsonl. Rebuild both quality
artifacts from those saved results without another paid run:

    uv run python -m "$lean.numina_quality" numina.parquet --prime-lists prime-lists --output-dir quality-output

Review the output and copy numina_autoinformalization_tasks.json to
experiments/post_training/task_curation/datasets/lean/numina_autoinformalization_tasks.json
to activate the selection.
"""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.pipeline.models import ImportRejection
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation.datasets.lean import autoinformalization as auto
from experiments.post_training.task_curation.datasets.lean import numina
from experiments.post_training.task_curation.datasets.lean.lean_problem_similarity import row_key, statement_key

LABELS = Path(__file__).with_name("numina_answer_disclosure_labels.jsonl")
GOLDEN = Path("artifacts/numina-answer-disclosure/numina_golden_controls.jsonl")
SOURCE_SHA256 = auto.MANIFEST["input"]["sha256"]

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


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_labels(path: Path) -> dict[str, str]:
    labels = {}
    for record in read_records(path):
        key, label = record["row"], record["label"]
        if key in labels or label not in {"yes", "no", "uncertain", "processing_failure"}:
            raise ValueError(f"Duplicate key or invalid disclosure label at {key}")
        labels[key] = label
    return labels


def write_records(path: Path, records: list[dict]) -> None:
    """Replace a JSONL output after writing it completely."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_rows(parquet: Path, prime_dir: Path, labels: dict | None) -> Iterator[dict]:
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
        key = row_key(data)
        index = str(data["_numina_index"])
        row: dict = dict.fromkeys(FIELDS, "")
        row.update(
            numina_row=index,
            uuid=data["uuid"],
            split=auto.assigned_split(data) or "",
            question_type=data.get("question_type") or "",
            row_key=key or "",
            golden_control="not_run",
        )
        if not numina.is_human_statement_row(data, None):
            row["cause"] = "not_human"
        elif row["question_type"] in auto.EXCLUDED_TYPES:
            row["cause"] = "excluded_question_type"
        elif not row["split"]:
            row["cause"] = "duplicate_or_missing_input"
        else:
            contract = numina.statement_row_contract(data, listed)
            if isinstance(contract, ImportRejection):
                row["cause"] = contract.reason
            else:
                label = labels.get(key, "missing") if labels is not None else ""
                row["disclosure"] = label
                if labels is None or label == "no":
                    row["formal"] = auto.formal_source(contract.prefix, contract.statement)
                else:
                    row["cause"] = "answer_disclosed" if label == "yes" else f"disclosure_{label}"
                if row["formal"]:
                    row["reference"] = data["problem"].strip()
                    row["formal_sha256"] = hashlib.sha256(row["formal"].encode()).hexdigest()
        row["included"] = not bool(row["cause"])
        yield row


def source_rows(parquet: Path, prime_lists: Path, labels: dict | None = None) -> list[dict]:
    """Return source decisions with eval overlap checks applied before either model runs."""
    rows = list(prepare_rows(parquet, prime_lists, labels))
    # Check against proof training statements even when their auto-informalization pair was filtered out.
    proof_types = set()
    for data in auto.indexed_rows(StoragePath(str(parquet)), None):
        if data["uuid"] in auto.PROOF_TRAIN_UUIDS and numina.proof_key(data) in numina.PASSING_PROOFS:
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
    return hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()


def eligible_for_golden(row: dict) -> bool:
    """Apply the shared compilation exclusions, disclosure decision, and type filter."""
    return (
        row["uuid"] not in numina.UNCOMPILABLE
        and row["disclosure"] == "no"
        and row["question_type"] not in auto.EXCLUDED_TYPES
    )


def final_manifest(rows: Iterator[dict]) -> dict:
    accepted = []
    counts = Counter()
    for row in rows:
        if row["cause"]:
            continue
        if not eligible_for_golden(row):
            raise ValueError("Invalid type or disclosure decision in final selection")
        if row["golden_control"] != "PASS":
            raise ValueError(f"Unfinished quality checks for row {row['numina_row']}")
        if row["formal_sha256"] != hashlib.sha256(row["formal"].encode()).hexdigest():
            raise ValueError("Changed formal statement")
        key = row["row_key"]
        accepted.append(key)
        counts[row["split"]] += 1
    if not accepted or len(set(accepted)) != len(accepted):
        raise ValueError("Empty or duplicate final selection")
    return dict(accepted=sorted(accepted), task_counts={s: counts[s] for s in ("train", "eval")})


def build(parquet: Path, prime_lists: Path, labels: Path, golden: Path, output_dir: Path) -> dict:
    """Join frozen decisions and write both quality artifacts in one invocation."""
    rows = source_rows(parquet, prime_lists, read_labels(labels))
    results = {}
    from_csv = golden.suffix == ".csv"
    if from_csv:
        locators = {(row["numina_row"], row["uuid"]): row["row_key"] for row in rows}
        with golden.open(newline="", encoding="utf-8-sig") as stream:
            records = []
            for result in csv.DictReader(stream):
                locator = (result["numina_row"], result["uuid"])
                if locator not in locators:
                    raise ValueError(f"Unknown golden CSV row {locator}")
                records.append({**result, "row_key": locators[locator]})
    else:
        records = read_records(golden)
    for result in records:
        key = result["row_key"]
        if key in results:
            raise ValueError(f"Duplicate golden result {key}")
        results[key] = result
    source_keys = {row["row_key"] for row in rows}
    if results.keys() - source_keys:
        raise ValueError("Golden results contain unknown source rows")
    for row in rows:
        if row["cause"]:
            continue
        result = results.get(row["row_key"])
        if result is None:
            raise ValueError(f"Missing golden result for {row['row_key']}")
        if not from_csv and result["formal_sha256"] != row["formal_sha256"]:
            raise ValueError("Golden result belongs to a different formal statement")
        label = result["golden_control"]
        if label not in {"PASS", "FAIL", "MALFORMED"}:
            raise ValueError("Invalid golden verdict")
        # Check rejected results too, so a stale rubric cannot silently discard good rows.
        if not from_csv and not any(
            result["judge_spec_sha256"] == spec_digest(golden_spec(row["formal"], row["reference"], tokens))
            for tokens in (1024, 2048, auto.JUDGE_OUTPUT_TOKENS)
        ):
            raise ValueError("Stale golden control")
        for field in ("golden_control", "note"):
            row[field] = result[field]
        row["golden_source"] = "reviewer_csv" if from_csv else "gemini_batch"
        row["golden_cause"] = result.get("cause", "")
        if not from_csv:
            row["judge_model"] = result["judge_model"]
            row["judge_spec_sha256"] = result["judge_spec_sha256"]
        row["cause"] = {"PASS": "", "FAIL": "golden_failed", "MALFORMED": "judge_malformed"}[label]
        row["included"] = label == "PASS"
    metadata = dict(
        dataset_revision=numina.NUMINA_REVISION,
        source_sha256=SOURCE_SHA256,
        labels_sha256=file_sha256(StoragePath(str(labels))),
        golden_sha256=file_sha256(StoragePath(str(golden))),
        golden_format="reviewer_csv" if from_csv else "gemini_batch",
        judge_models=sorted({r["judge_model"] for r in results.values() if r.get("judge_model")}),
    )
    return write_artifacts(rows, metadata, output_dir)


def write_artifacts(rows: list[dict], metadata: dict, output_dir: Path) -> dict:
    """Write a validated selection and its report from the same rows."""
    manifest = {**final_manifest(iter(rows)), **metadata}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        report = Path(temporary) / "numina_row_quality_report.csv"
        with report.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest["report_sha256"] = file_sha256(StoragePath(str(report)))
        target = Path(temporary) / "numina_autoinformalization_tasks.json"
        target.write_text(json.dumps(manifest, indent=1) + "\n")
        os.replace(report, output_dir / report.name)
        os.replace(target, output_dir / target.name)
    return manifest


def build_intersection(
    parquet: Path, prime_lists: Path, labels: Path, primary: Path, required: Path, output_dir: Path
) -> dict:
    """Keep rows passed by both independently validated sets of golden results."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = build(parquet, prime_lists, labels, primary, root / "primary")
        second = build(parquet, prime_lists, labels, required, root / "required")
        with (root / "primary/numina_row_quality_report.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (root / "required/numina_row_quality_report.csv").open(newline="") as stream:
            other = list(csv.DictReader(stream))
        for row, extra in zip(rows, other, strict=True):
            if row["row_key"] != extra["row_key"] or row["formal_sha256"] != extra["formal_sha256"]:
                raise ValueError("Golden reports refer to different source rows")
            if row["golden_control"] == "not_run":
                continue
            verdicts = (row["golden_control"], extra["golden_control"])
            row.update(
                primary_verdict=verdicts[0],
                required_verdict=verdicts[1],
                required_note=extra["note"],
                required_cause=extra["golden_cause"],
            )
            label = "PASS" if verdicts == ("PASS", "PASS") else "FAIL" if "FAIL" in verdicts else "MALFORMED"
            row.update(
                golden_control=label,
                golden_source="intersection",
                included=label == "PASS",
                cause={"PASS": "", "FAIL": "golden_failed", "MALFORMED": "judge_malformed"}[label],
            )
        metadata = dict(
            dataset_revision=numina.NUMINA_REVISION,
            source_sha256=SOURCE_SHA256,
            labels_sha256=first["labels_sha256"],
            golden_format="intersection",
            golden_inputs=[dict(format=m["golden_format"], sha256=m["golden_sha256"]) for m in (first, second)],
            judge_models=sorted(set(first["judge_models"]) | set(second["judge_models"])),
        )
        return write_artifacts(rows, metadata, output_dir)


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--prime-lists", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_arguments(parser)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--require-golden", type=Path, help="Require PASS in this second set of results too")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.require_golden:
        manifest = build_intersection(
            args.parquet, args.prime_lists, args.labels, args.golden, args.require_golden, args.output_dir
        )
    else:
        manifest = build(args.parquet, args.prime_lists, args.labels, args.golden, args.output_dir)
    print(json.dumps(manifest["task_counts"]))


if __name__ == "__main__":
    main()
