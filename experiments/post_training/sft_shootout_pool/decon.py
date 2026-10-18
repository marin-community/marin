# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontaminate pool P against SWE-bench Verified, Terminal-Bench 2.1 and TBLite 2.0 with DataKit's decon step.

Eval documents (one per benchmark task):
- SWE-bench Verified (princeton-nlp/SWE-bench_Verified@c104f840): problem statement, gold patch and test patch.
- Terminal-Bench 2.1 (harbor-framework/terminal-bench-2-1@c5ee500c, the commit DataKit's eval corpus pins) and
  TBLite 2.0 (open-thoughts/OpenThoughts-TBLite@f075e463, the commit Harbor's registry pins for openthoughts-tblite
  2.0): instruction.md and the text files under tests/ and solution/.

Pool documents (one per task): the instruction plus the verifier's test files (TMax tests/test_final_state.py;
SWE-smith the restored fail-to-pass test files).

Matching uses marin.datakit.decon.decon_to_parquet with the reference pipeline's thresholds (13-gram, overlap 0.5,
at least 2 matched features).
"""

import io
import json
import subprocess
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from datasets import load_dataset
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.decon import NGramConfig, decon_to_parquet
from marin.datakit.normalize import NormalizedData

from experiments.post_training.sft_shootout_pool.swesmith_packages import read_package

SWEBENCH_VERIFIED = ("princeton-nlp/SWE-bench_Verified", "c104f840cc67f8b6eec6f759ebc8b2693d585d4a")
TERMINAL_BENCH_21 = ("harbor-framework/terminal-bench-2-1", "c5ee500c185224c97cd6caff7866a990a0057f41")
TBLITE = ("open-thoughts/OpenThoughts-TBLite", "f075e463472c7790b85793b392dff1fff20cc0e3")
NGRAM = NGramConfig(ngram_length=13, overlap_threshold=0.5, min_matched_features=2)
# environment/ is excluded: it vendors upstream projects (license headers, build files) that are not benchmark content.
EVAL_TASK_PARTS = ("instruction.md", "tests/", "solution/")
TEXT_SUFFIXES = (".md", ".py", ".sh", ".txt", ".json", ".toml", ".yaml", ".yml", ".c", ".js", ".ts", ".go", ".rs", "")


def _text(data: bytes) -> str | None:
    try:
        return data.decode()
    except UnicodeDecodeError:
        return None


def _folder_docs(prefix: str, files: dict[str, bytes]) -> list[dict]:
    """Group {'<task>/<path>': bytes} into one document per task folder."""
    tasks: dict[str, list[str]] = {}
    for path in sorted(files):
        task, _, rest = path.partition("/")
        if (
            rest.startswith(EVAL_TASK_PARTS)
            and Path(rest).suffix in TEXT_SUFFIXES
            and (text := _text(files[path])) is not None
        ):
            tasks.setdefault(task, []).append(f"{rest}\n{text}")
    return [{"id": f"{prefix}/{task}", "text": "\n\n".join(parts)} for task, parts in tasks.items()]


def eval_documents(work: Path) -> dict[str, list[dict]]:
    verified = load_dataset(SWEBENCH_VERIFIED[0], split="test", revision=SWEBENCH_VERIFIED[1])
    docs = {
        "swebench-verified": [
            {
                "id": f"swebench-verified/{row['instance_id']}",
                "text": "\n\n".join((row["problem_statement"], row["patch"], row["test_patch"])),
            }
            for row in verified
        ]
    }
    archive = requests.get(f"https://github.com/{TERMINAL_BENCH_21[0]}/archive/{TERMINAL_BENCH_21[1]}.zip", timeout=300)
    archive.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(archive.content)) as tb:
        root = tb.namelist()[0].split("/")[0]
        files = {
            name.split("/", 2)[2]: tb.read(name)
            for name in tb.namelist()
            if name.startswith(f"{root}/tasks/") and not name.endswith("/") and name.count("/") >= 3
        }
    docs["terminal-bench-2.1"] = _folder_docs("terminal-bench-2.1", files)
    clone = work / "tblite"
    subprocess.run(["git", "clone", "-q", f"https://github.com/{TBLITE[0]}.git", str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", TBLITE[1]], check=True)
    files = {
        str(path.relative_to(clone)): path.read_bytes()
        for path in clone.rglob("*")
        if path.is_file() and ".git" not in path.parts and len(path.relative_to(clone).parts) >= 2
    }
    docs["tblite-2.0"] = _folder_docs("tblite-2.0", files)
    return docs


def tmax_documents(converted: Path, tasks: list[str]) -> list[dict]:
    return [
        {
            "id": task,
            "text": (
                (converted / task / "instruction.md").read_text()
                + "\n\n"
                + (converted / task / "tests" / "test_final_state.py").read_text()
            ),
        }
        for task in tasks
    ]


def swesmith_documents(packages: Path, tasks: set[str]) -> list[dict]:
    docs = []
    for row in pq.read_table(packages).to_pylist():
        if row["path"] not in tasks:
            continue
        files = read_package(row["task_binary"])
        tests = [
            f"{name}\n{text}"
            for name in sorted(files)
            if name.startswith("tests/f2p_restore/files/") and (text := _text(files[name])) is not None
        ]
        docs.append({"id": row["path"], "text": "\n\n".join([files["instruction.md"].decode(), *tests])})
    return docs


def run(pool_docs: list[dict], evals: dict[str, list[dict]], work: Path) -> dict[str, dict]:
    """Return {pool task id: {contaminated, max_overlap, matched_evals}} from DataKit's decon marks."""
    corpus = work / "corpus" / "main"
    corpus.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(pool_docs), corpus / "part-00000-of-00001.parquet")
    (work / "corpus" / "dup").mkdir()
    eval_dir = work / "evals"
    eval_dir.mkdir()
    for name, docs in evals.items():
        (eval_dir / f"{name}.jsonl").write_text("".join(json.dumps(doc) + "\n" for doc in docs))
    normalized = NormalizedData(main_output_dir=str(corpus), dup_output_dir=str(work / "corpus" / "dup"), counters={})
    # The corpus is a few thousand documents; run the zephyr stages in this process, not as cluster jobs.
    with set_current_client(LocalClient()):
        decon_to_parquet(
            normalized_data=normalized,
            eval_data_sources=[str(eval_dir)],
            output_path=str(work / "marks"),
            ngram=NGRAM,
            # The bloom holds one entry per eval n-gram feature (about 1M here), not per document.
            estimated_doc_count=10_000_000,
        )
    index = pq.read_table(work / "marks" / "_bloom" / "eval_hash_index.parquet").to_pylist()
    owners: dict[int, set[str]] = {}
    for entry in index:
        owners.setdefault(entry["hash"], set()).add(entry["eval_id"])
    results = {}
    for path in sorted((work / "marks" / "outputs" / "main").glob("*.parquet")):
        for mark in pq.read_table(path).to_pylist():
            confirmed = [value for value in mark["matched_hashes"] if value in owners]
            results[mark["id"]] = {
                "contaminated": mark["contaminated"],
                "max_overlap": mark["max_overlap"],
                "matched_features": len(mark["matched_hashes"]),
                "confirmed_features": len(confirmed),
                "matched_evals": sorted({owner for value in confirmed for owner in owners[value]}),
            }
    if set(results) != {doc["id"] for doc in pool_docs}:
        raise ValueError("Decon marks do not cover every pool task")
    return results
