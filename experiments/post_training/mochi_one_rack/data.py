# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin a small subset of Ben's long-loop tasks and an explicit visible-length probe."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from rigging.filesystem.storage_path import StoragePath
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
PROBE_TEXT_TOKEN_LIMIT = 31_600
PROBE_CONTEXT_OVERHEAD_TOKENS = 208


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


def length_probe(task: dict, tokenizer, token_limit: int = PROBE_TEXT_TOKEN_LIMIT) -> tuple[dict, int]:
    """Keep the real arithmetic verifier; add visible, fully attended ledger text."""
    task = json.loads(json.dumps(task))
    question = task["context"]["events"][-1]["content"]
    line_count = 4000
    while True:
        lines = [f"Ledger entry {index}: recorded value {index % 97}.\n" for index in range(line_count)]
        full_text = "This is a controlled context-length probe. Read the final question.\n" + "".join(lines)
        full_text += "\nFinal question: " + question
        if len(tokenizer.encode(full_text, add_special_tokens=False)) >= token_limit:
            break
        line_count *= 2
    start, stop = 0, len(lines)
    while start < stop:
        count = (start + stop + 1) // 2
        text = "This is a controlled context-length probe. Read the final question.\n" + "".join(lines[:count])
        text += "\nFinal question: " + question
        if len(tokenizer.encode(text, add_special_tokens=False)) <= token_limit:
            start = count
        else:
            stop = count - 1
    text = "This is a controlled context-length probe. Read the final question.\n" + "".join(lines[:start])
    text += "\nFinal question: " + question
    tokens = len(tokenizer.encode(text, add_special_tokens=False))
    assert token_limit - 100 <= tokens <= token_limit
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


def write_context_trial_data(
    *,
    retained_data: str,
    retained_data_sha256: str,
    tokenizer_source: str,
    output_path: str,
    context_cap: int,
    updates: int,
    max_new_tokens: int,
) -> Artifact:
    """Write consecutive mixed-task pairs with a near-limit probe in every pair."""
    source = StoragePath(retained_data) / "train.parquet"
    with source.open("rb") as stream:
        data = stream.read()
    observed_sha256 = hashlib.sha256(data).hexdigest()
    if observed_sha256 != retained_data_sha256:
        raise ValueError(f"Retained task artifact changed: {source}")
    tasks = {
        task["id"]: task
        for row in pq.read_table(io.BytesIO(data)).to_pylist()
        for task in [json.loads(row["task_spec"])]
    }
    if set(tasks) != set((*AGENTIC_IDS, *NONAGENTIC_IDS)):
        raise ValueError("Context trials require the four retained mixed-task inputs")
    tokenizer_hashes = {}
    with TemporaryDirectory(prefix="mochi-context-tokenizer-") as temporary:
        local = Path(temporary)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            with (StoragePath(tokenizer_source) / name).open("rb") as stream:
                contents = stream.read()
            (local / name).write_bytes(contents)
            tokenizer_hashes[name] = hashlib.sha256(contents).hexdigest()
        model_maximum = json.loads((local / "config.json").read_text())["max_position_embeddings"]
        if context_cap > model_maximum:
            raise ValueError(f"Context {context_cap} exceeds the model maximum {model_maximum}")
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        probe, tokens = length_probe(
            tasks[NONAGENTIC_IDS[0]], tokenizer, context_cap - max_new_tokens - PROBE_CONTEXT_OVERHEAD_TOKENS
        )
    rows = [task for index in range(updates) for task in (tasks[AGENTIC_IDS[index % len(AGENTIC_IDS)]], probe)]
    destination = StoragePath(output_path)
    destination.mkdirs()
    encoded = pa.BufferOutputStream()
    pq.write_table(pa.table({"task_spec": [json.dumps(task) for task in rows]}), encoded)
    parquet_bytes = encoded.getvalue().to_pybytes()
    with (destination / "train.parquet").open("wb") as stream:
        stream.write(parquet_bytes)
    manifest = {
        "retained_data": str(source),
        "retained_data_sha256": observed_sha256,
        "tokenizer_source": tokenizer_source,
        "tokenizer_hashes": tokenizer_hashes,
        "context_cap": context_cap,
        "model_max_position_embeddings": model_maximum,
        "updates": updates,
        "max_new_tokens": max_new_tokens,
        "controlled_probe_text_tokens": tokens,
        "train_parquet_sha256": hashlib.sha256(parquet_bytes).hexdigest(),
        "train_parquet_bytes": len(parquet_bytes),
        "task_ids": [task["id"] for task in rows],
    }
    with (destination / "manifest.json").open("wt") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return Artifact(path=output_path)


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
        "32k": [
            tasks[AGENTIC_IDS[0]],
            probe,
            tasks[AGENTIC_IDS[1]],
            tasks[NONAGENTIC_IDS[1]],
            tasks[AGENTIC_IDS[0]],
            tasks[NONAGENTIC_IDS[0]],
            tasks[AGENTIC_IDS[1]],
            tasks[NONAGENTIC_IDS[1]],
            tasks[AGENTIC_IDS[0]],
            tasks[NONAGENTIC_IDS[0]],
        ],
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
