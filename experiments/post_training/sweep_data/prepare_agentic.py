# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the agentic half of the LR-sweep set: TMax + Kimi SWE-smith pilot traces on Snowball v0.

Writes prebuilt Levanter rows (input_ids, loss_weights) for training and one validation file per family,
plus the held-out task ids, a per-row token audit and a manifest with file hashes.
"""

import argparse
import glob
import gzip
import hashlib
import json
import random
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.post_training.sweep_data.render import training_row
from experiments.post_training.sweep_data.select import (
    CONTEXT,
    GIT_HISTORY_READ,
    KIMI_TASKS,
    SEED,
    SWEBENCH_VERIFIED_REPOS,
    Selection,
    TrainingRecord,
    select,
)
from experiments.post_training.sweep_data.tmax_leaks import ANSWER_PATH, answer_paths

STUDENT_MODEL = "open-athena/Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12"
STUDENT_REVISION = "4eeb756b404db25bc04e4daf49dfb9817bceb741"
STUDENT_WEIGHTS_REVISION = "271c4b3c6b38e8f645c88b8938a6ccf2b2195f2a"
STUDENT_REGIONAL_COPY = "s3://marin-us-east-02a/marin/weight-merging/campaign3-20261011/dare-linear-02/model"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
)
TMAX = (
    "allenai/tmax-sft",
    "9d6ab7471ffa2884b2ef0cd2b7e5e22a027ff1b4",
    "skill_tax_20260505_2.2k_combined_balanced_thinking_only_success",
)
TMAX_ENVS = ("allenai/TMax-SFT-16.5K", "848e218270672080a92a94791c028ee43ae33387", "tasks.zip")
KIMI = (
    "open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
    "1b87a9cf78b897a2355105d47afd44d818cad25d",
)
ALLKIMI = ("laion/calibforge-relay-traces", "66feab4d24bfc37af4c38d95a6a26187b5648308", "kimi_trials.txt", 4392)
BATCH = 16
PAD_TOKEN = 128001


def audit_hashes() -> dict[str, str]:
    """The sha256 of every file in the regional copy, from the campaign-3 audit published with v0."""
    audit_path = hf_hub_download(STUDENT_MODEL, "provenance/campaign3/audit.json", revision=STUDENT_REVISION)
    return {name: entry["sha256"] for name, entry in json.loads(Path(audit_path).read_text())["files"].items()}


def load_tokenizer(directory: Path, expected: dict[str, str]):
    """Load v0's tokenizer from the regional copy after checking each file against the campaign-3 audit hash."""
    for name in TOKENIZER_FILES:
        data = (StoragePath(STUDENT_REGIONAL_COPY) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError(f"Regional tokenizer file differs from the audit: {name}")
        (directory / name).write_bytes(data)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    if tokenizer.pad_token_id != PAD_TOKEN:
        raise ValueError(f"Unexpected pad token {tokenizer.pad_token_id}")
    return tokenizer


def row_line(record: TrainingRecord) -> str:
    """One prebuilt row as its JSONL line (no newline)."""
    return json.dumps(training_row(record["ids"], record["mask"], CONTEXT, PAD_TOKEN), separators=(",", ":"))


def padding_row() -> str:
    """The zero-loss padding row as its JSONL line (no newline)."""
    return json.dumps({"input_ids": [PAD_TOKEN] * CONTEXT, "loss_weights": [0] * CONTEXT}, separators=(",", ":"))


def write_prebuilt(path: Path, lines: Iterable[str]) -> tuple[int, int]:
    """Write row lines, then zero-loss padding rows up to a multiple of the batch size; return (rows, padding)."""
    rows = 0
    with gzip.open(path, "wt", compresslevel=6) as stream:
        for line in lines:
            stream.write(line + "\n")
            rows += 1
        padding = -rows % BATCH
        empty = padding_row()
        for _ in range(padding):
            stream.write(empty + "\n")
    return rows, padding


def write_rows(path: Path, records: list[TrainingRecord]) -> tuple[int, int]:
    """Write prebuilt rows, then zero-loss padding rows up to a multiple of the batch size; return (rows, padding)."""
    return write_prebuilt(path, (row_line(record) for record in records))


def verify_rows(path: Path, records: list[TrainingRecord], padding: int) -> int:
    """Re-read a written file: shapes, binary weights, zero loss past each row's end and on padding rows.

    Returns the file's supervised-token count (the sum of all loss weights).
    """
    supervised = 0
    with gzip.open(path, "rt") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            ids, weights = row["input_ids"], row["loss_weights"]
            if len(ids) != CONTEXT or len(weights) != CONTEXT or not set(weights) <= {0, 1}:
                raise ValueError(f"Malformed row {index} in {path.name}")
            if index < len(records):
                length = len(records[index]["ids"])
                if ids[:length] != records[index]["ids"] or weights[: length - 1] != records[index]["mask"][1:]:
                    raise ValueError(f"Row {index} differs from its rendering")
                if any(weights[length - 1 :]) or any(token != PAD_TOKEN for token in ids[length:]):
                    raise ValueError(f"Row {index} has loss or content in its padding")
            elif any(weights) or set(ids) != {PAD_TOKEN}:
                raise ValueError(f"Padding row {index} is not empty")
            supervised += sum(weights)
    if index + 1 != len(records) + padding:
        raise ValueError(f"Row count differs in {path.name}")
    return supervised


def read_component(root: StoragePath, names: Iterable[str] | None = None) -> tuple[dict, bytes, dict[str, bytes]]:
    """Read a prepared prefix's manifest and files, checking each file against the manifest's output_hashes.

    `names` defaults to every file in output_hashes. Returns (manifest, manifest bytes, {name: file bytes}).
    """
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    files = {}
    for name in manifest["output_hashes"] if names is None else names:
        files[name] = (root / name).read_bytes()
        if hashlib.sha256(files[name]).hexdigest() != manifest["output_hashes"][name]:
            raise ValueError(f"{root}/{name} changed since it was prepared")
    return manifest, manifest_bytes, files


def output_hashes(out: Path) -> dict[str, str]:
    """sha256 of every file in a local output directory, by name."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir())}


def upload_checked(out: Path, output: StoragePath, manifest: dict) -> None:
    """Write manifest.json into `out`, then copy every file to `output` (manifest last), re-hashing each upload."""
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output.mkdirs()
    for path in sorted(out.iterdir(), key=lambda path: path.name == "manifest.json"):
        (output / path.name).write_bytes(path.read_bytes())
        if (
            path.name != "manifest.json"
            and hashlib.sha256((output / path.name).read_bytes()).hexdigest() != manifest["output_hashes"][path.name]
        ):
            raise ValueError(f"Upload hash mismatch: {path.name}")


def audit(record: dict, split: str) -> dict:
    return {
        "split": split,
        "family": record["family"],
        "task": record["task"],
        "trajectory": record["trajectory"],
        "repo": record.get("repo"),
        "supervised_tokens": sum(record["mask"][1:]),
        "context_tokens": len(record["ids"]),
        "sha256": hashlib.sha256(json.dumps(record["ids"]).encode()).hexdigest(),
    }


def family_totals(records: list[dict]) -> dict:
    return {
        "rows": len(records),
        "tasks": len({record["task"] for record in records}),
        "supervised_tokens": sum(sum(record["mask"][1:]) for record in records),
        "context_tokens": sum(len(record["ids"]) for record in records),
        "longest_row": max(len(record["ids"]) for record in records),
    }


class Sources(NamedTuple):
    tmax_rows: list[dict]
    kimi_rows: list[dict]
    allkimi: frozenset[str]
    allkimi_bytes: bytes
    leaks: dict[str, dict[str, list[str]]]
    tmax_hidden: dict[str, list[str]]


def load_sources() -> Sources:
    """Download the pinned TMax, Kimi, allkimi and TMax-environment sources and check their row counts."""
    tmax_dir = snapshot_download(TMAX[0], repo_type="dataset", revision=TMAX[1], allow_patterns=[f"{TMAX[2]}/*"])
    kimi_dir = snapshot_download(KIMI[0], repo_type="dataset", revision=KIMI[1], allow_patterns=["data/*"])
    tmax_rows = pq.read_table(glob.glob(f"{tmax_dir}/{TMAX[2]}/*.parquet")[0]).to_pylist()
    kimi_rows = pa.concat_tables(
        [pq.read_table(path) for path in sorted(glob.glob(f"{kimi_dir}/data/*.parquet"))]
    ).to_pylist()
    if len(tmax_rows) != 5795 or len(kimi_rows) != 9356:
        raise ValueError("Source row counts differ from the pinned revisions")
    allkimi_path = hf_hub_download(ALLKIMI[0], ALLKIMI[2], repo_type="dataset", revision=ALLKIMI[1])
    allkimi_bytes = Path(allkimi_path).read_bytes()
    allkimi = frozenset(line.strip() for line in allkimi_bytes.decode().splitlines() if line.strip())
    known = {row["trial_name"] for row in kimi_rows}
    if len(allkimi) != ALLKIMI[3] or not allkimi <= known:
        raise ValueError("allkimi trial list does not match the Kimi dataset's trial names")
    envs_path = hf_hub_download(TMAX_ENVS[0], TMAX_ENVS[2], repo_type="dataset", revision=TMAX_ENVS[1])
    with zipfile.ZipFile(envs_path) as archive:
        leaks = answer_paths(archive)
    tmax_hidden = {task: paths["hidden"] for task, paths in leaks.items() if paths["hidden"]}
    return Sources(tmax_rows, kimi_rows, allkimi, allkimi_bytes, leaks, tmax_hidden)


def write_row_files(out: Path, files: dict[str, list[TrainingRecord]]) -> dict[str, dict[str, int]]:
    """Write and re-read each prebuilt file; return its rows, padding rows and supervised tokens by file name."""
    counts = {}
    for name, records in files.items():
        rows, padding = write_rows(out / name, records)
        supervised = verify_rows(out / name, records, padding)
        if supervised != sum(sum(record["mask"][1:]) for record in records):
            raise ValueError(f"Supervised count changed in {name}")
        counts[name] = {"rows": rows, "padding_rows": padding, "supervised_tokens": supervised}
    return counts


def write_audit_files(out: Path, tokenizer, train: list[TrainingRecord], selection: Selection) -> None:
    """Write the per-row token audit, the held-out ids and decoded loss-mask probes of a few rows."""
    audits = [audit(record, "train") for record in train] + [
        audit(record, f"validation-{family}")
        for family in ("tmax", "swesmith")
        for record in selection["validation"][family]
    ]
    (out / "token-audit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audits))
    (out / "heldout-ids.json").write_text(json.dumps(selection["heldout"], indent=1) + "\n")
    probes = []
    for record in train[:2] + selection["validation"]["tmax"][:1] + selection["validation"]["swesmith"][:1]:
        spans, previous = [], None
        for token, weight in zip(record["ids"], record["mask"], strict=True):
            if weight != previous:
                spans.append({"trained": bool(weight), "text": ""})
                previous = weight
            spans[-1]["text"] += tokenizer.decode([token])
        probes.append(audit(record, "probe") | {"spans": spans})
    (out / "row-probes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in probes))


def build_manifest(
    out: Path,
    expected: dict[str, str],
    sources: Sources,
    selection: Selection,
    train: list[TrainingRecord],
    counts: dict[str, dict[str, int]],
) -> dict:
    """The agentic manifest; output_hashes covers every file already written to `out`."""
    tmax_rows, kimi_rows, allkimi, allkimi_bytes, leaks, tmax_hidden = sources
    val = {family: counts[f"validation-{family}-prebuilt.jsonl.gz"] for family in ("tmax", "swesmith")}
    return {
        "task": 202,
        "component": "agentic",
        "student_model": STUDENT_MODEL,
        "student_revision": STUDENT_REVISION,
        "student_weights_identical_to_revision": STUDENT_WEIGHTS_REVISION,
        "student_regional_copy": STUDENT_REGIONAL_COPY,
        "student_regional_copy_exceptions": {
            "README.md": "pre-correction card from 271c4b3c; HF 4eeb756b replaced it",
            "merge-provenance.json": "pre-correction provenance from 271c4b3c; HF 4eeb756b replaced it",
        },
        "tokenizer_files_sha256": {name: expected[name] for name in TOKENIZER_FILES},
        "pad_token_id": PAD_TOKEN,
        "max_sequence_tokens": CONTEXT,
        "batch_size": BATCH,
        "training_rows": counts["train-prebuilt.jsonl.gz"]["rows"],
        "padding_rows": counts["train-prebuilt.jsonl.gz"]["padding_rows"],
        "steps_per_epoch": sum(counts["train-prebuilt.jsonl.gz"][key] for key in ("rows", "padding_rows")) // BATCH,
        "supervised_tokens": counts["train-prebuilt.jsonl.gz"]["supervised_tokens"],
        "context_tokens": sum(len(record["ids"]) for record in train),
        "validation_files": {family: f"validation-{family}-prebuilt.jsonl.gz" for family in ("tmax", "swesmith")},
        "validation_rows": {family: val[family]["rows"] for family in val},
        "validation_padding_rows": {family: val[family]["padding_rows"] for family in val},
        "validation_supervised_tokens": {family: val[family]["supervised_tokens"] for family in val},
        "families": {
            family: {
                "train": family_totals(selection["train"][family]),
                "validation": family_totals(selection["validation"][family]),
                "eligible_tasks": selection["eligible_tasks"][family],
                "heldout_tasks": len(selection["heldout"][family]),
            }
            for family in ("tmax", "swesmith")
        },
        "sources": {
            "tmax": {"dataset": TMAX[0], "revision": TMAX[1], "config": TMAX[2], "rows": len(tmax_rows)},
            "swesmith": {"dataset": KIMI[0], "revision": KIMI[1], "rows": len(kimi_rows)},
        },
        "selection": {
            "seed": SEED,
            "tmax": "every success-split row; held-out tasks removed; all trajectories of a training task kept",
            "swesmith": (
                f'result == "1.0", one SWE-smith repo parsed from verifier_output, repo not in SWE-bench Verified, '
                "no command reading code from git history, "
                f"then {KIMI_TASKS} tasks allocated across repos in proportion to eligible tasks (largest remainder)"
            ),
            "swesmith_git_history_rule": {
                "why": (
                    "the SWE-smith packages leave the clean 'Initial commit', the 'Bug Patch' commit and a local "
                    "main branch in the agent's container, so 'git diff HEAD main' prints the fix; passes that read "
                    "history may have copied it"
                ),
                "pattern": GIT_HISTORY_READ.pattern,
                "applied_to": "every terminus-2 keystrokes string in the assistant turns of passing rows",
                "passing_rows_matched": selection["dropped"].get("swesmith_reads_git_history", 0),
                "passing_rows_total": sum(row["result"] == "1.0" for row in kimi_rows),
            },
            "tmax_hidden_asset_rule": {
                "why": (
                    "some TMax environments build verifier assets (oracle programs, ground-truth files) into the "
                    "agent's image; a trajectory that names one may have read the answer"
                ),
                "environments": f"{TMAX_ENVS[0]}@{TMAX_ENVS[1]}/{TMAX_ENVS[2]}",
                "answer_path_pattern": ANSWER_PATH.pattern,
                "envs_with_answer_paths": len(leaks),
                "envs_with_hidden_answer_paths": len(tmax_hidden),
                "rule": "drop a trajectory whose bash tool calls mention a hidden answer path of its task",
                "heuristic": "a mention does not prove the answer was copied; this is a conservative filter",
                "trajectories_dropped": selection["dropped"].get("tmax_references_hidden_verifier_asset", 0),
            },
            "swebench_verified_repos_excluded": sorted(SWEBENCH_VERIFIED_REPOS),
            "heldout": (
                "round(5%) of each family's eligible tasks, seeded, drawn before training rows are built; "
                "SWE-smith candidates exclude allkimi trials"
            ),
            "validation_trajectory": "first trajectory of each held-out task in trajectory-name order",
            "context_rule": "trajectories longer than max_sequence_tokens are dropped, never truncated",
            "dropped": selection["dropped"],
            "tmax_tasks_lost_to_context": selection["tmax_source_tasks"] - selection["eligible_tasks"]["tmax"],
            "family_balance": (
                "pilot only: TMax keeps every success trajectory (several per task), SWE-smith has one per task; "
                "the real arms use one trajectory per task, so this TMax/SWE-smith token split is not theirs"
            ),
            "allkimi_trials": {
                "dataset": ALLKIMI[0],
                "revision": ALLKIMI[1],
                "file": ALLKIMI[2],
                "sha256": hashlib.sha256(allkimi_bytes).hexdigest(),
                "trials": len(allkimi),
                "use": (
                    "SWE-smith held-out tasks drawn only from tasks with no trial in this list (task identity); "
                    "training rows not filtered"
                ),
                "discrepancy": (
                    "every listed trial name is in the Kimi dataset, but only "
                    f"{selection['allkimi_overlap']['allkimi_tasks_passing_in_this_dataset']} of them have "
                    'result == "1.0" here, so the allkimi filter did not equal this dataset\'s verdict'
                ),
                "overlap": selection["allkimi_overlap"],
                "residual_base_exposure": (
                    "unknown: v0's other ancestors may have seen these tasks; held-out only avoids allkimi trials"
                ),
            },
            "swesmith_repo_allocation": selection["swesmith_allocation"],
        },
        "loss_mask": (
            "Grug chat template generation blocks: every assistant turn (reasoning, content, tool calls, "
            "<|eot_id|>) "
            "is trained; system, user and tool turns and assistant headers are context. "
            "loss_weights[i] scores the prediction of input_ids[i + 1]."
        ),
        "rendering": {
            "template": "v0 chat_template.jinja, enable_thinking=True",
            "tmax": "reasoning_content -> <|start_think|>; tool_calls -> <tool_call>; tools listed in the system header",
            "swesmith": "inline <think>...</think> moved to reasoning_content; terminus-2 JSON content kept verbatim",
        },
        "test_exposed": False,
        "output_hashes": output_hashes(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    expected = audit_hashes()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "tokenizer").mkdir()
        tokenizer = load_tokenizer(directory / "tokenizer", expected)
        sources = load_sources()
        selection = select(tokenizer, sources.tmax_rows, sources.kimi_rows, sources.allkimi, sources.tmax_hidden)

        train = selection["train"]["tmax"] + selection["train"]["swesmith"]
        random.Random(SEED).shuffle(train)
        out = directory / "out"
        out.mkdir()
        files = {"train-prebuilt.jsonl.gz": train}
        for family in ("tmax", "swesmith"):
            files[f"validation-{family}-prebuilt.jsonl.gz"] = selection["validation"][family]
        counts = write_row_files(out, files)
        write_audit_files(out, tokenizer, train, selection)
        manifest = build_manifest(out, expected, sources, selection, train, counts)
        upload_checked(out, output, manifest)
        print(
            "AGENTIC_PREPARED " + json.dumps({key: value for key, value in manifest.items() if key != "selection"}),
            flush=True,
        )
        print("AGENTIC_SELECTION " + json.dumps(manifest["selection"]), flush=True)


if __name__ == "__main__":
    main()
