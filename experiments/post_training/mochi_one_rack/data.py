# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin a small subset of Ben's long-loop tasks and an explicit visible-length probe."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SOURCE_FILES = {
    "agentic-train.parquet": "7e658efb2a3b53544552400e0e78d6ba4dd99ab31953828d31158dc38c7cf549",
    "nonagentic-train.parquet": "c8948b3c13957ee054b4b8b5553127f75c44a50577b5b2b798261e9f8a5abc45",
}
AGENTIC_IDS = (
    "long-loop-smartie-calibforge:AweAI-Team__CalibForge/contrastive_solver/software-engineering_20260606_172606_007",
    "long-loop-smartie-calibforge:AweAI-Team__CalibForge/contrastive_solver/software-engineering_20260604_002443_002",
)
NONAGENTIC_IDS = (
    "nemotron_ultra_rlvr1_ultra_sft_step3200_reasoning_gym-e29ff0e4b35890886ebf161edbbcaa807234f8a7a4119b121b4e29755dcb9f76",
    "reasoning_gym_generated-60f84bd9d331d39b51ce0befb27ba00cb0c7204b809dc3baa9c5d9d2324e428b",
)


def selected_tasks(root: Path) -> dict[str, dict]:
    selected = {}
    wanted = set((*AGENTIC_IDS, *NONAGENTIC_IDS))
    for name, expected in SOURCE_FILES.items():
        path = root / name
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                task = json.loads(row["task_spec"])
                if task["id"] in wanted:
                    selected[task["id"]] = task
    assert set(selected) == wanted
    return selected


def length_probe(task: dict, tokenizer) -> tuple[dict, int]:
    """Keep the real arithmetic verifier; add visible, fully attended ledger text."""
    task = json.loads(json.dumps(task))
    question = task["context"]["events"][-1]["content"]
    lines = [f"Ledger entry {index}: recorded value {index % 97}.\n" for index in range(4000)]
    start, stop = 0, len(lines)
    while start < stop:
        count = (start + stop + 1) // 2
        text = "This is a controlled context-length probe. Read the final question.\n" + "".join(lines[:count])
        text += "\nFinal question: " + question
        if len(tokenizer.encode(text, add_special_tokens=False)) <= 31_600:
            start = count
        else:
            stop = count - 1
    text = "This is a controlled context-length probe. Read the final question.\n" + "".join(lines[:start])
    text += "\nFinal question: " + question
    tokens = len(tokenizer.encode(text, add_special_tokens=False))
    assert 31_500 <= tokens <= 31_600
    task["context"]["events"][-1]["content"] = text
    task["id"] = "mochi-controlled-length-probe:visible-ledger-arithmetic"
    task["source"] = {
        "dataset": "mochi-controlled-length-probe",
        "revision": hashlib.sha256(text.encode()).hexdigest(),
        "row": "visible-ledger-arithmetic",
        "importer_revision": "1",
    }
    task["tags"] = [*task["tags"], "controlled-length-probe"]
    return task, tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = selected_tasks(args.source_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    probe, tokens = length_probe(tasks[NONAGENTIC_IDS[0]], tokenizer)
    selections = {
        "4k": [tasks[AGENTIC_IDS[0]], tasks[NONAGENTIC_IDS[0]], tasks[AGENTIC_IDS[1]], tasks[NONAGENTIC_IDS[1]]],
        "32k": [tasks[AGENTIC_IDS[0]], probe, tasks[AGENTIC_IDS[1]], tasks[NONAGENTIC_IDS[1]]],
    }
    manifest = {"source_files": SOURCE_FILES, "controlled_probe_prompt_tokens": tokens, "stages": {}}
    for stage, rows in selections.items():
        path = args.output / stage / "train.parquet"
        path.parent.mkdir(parents=True, exist_ok=False)
        pq.write_table(pa.table({"task_spec": [json.dumps(task) for task in rows]}), path)
        manifest["stages"][stage] = {
            "task_ids": [task["id"] for task in rows],
            "sources": [task["source"] for task in rows],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
