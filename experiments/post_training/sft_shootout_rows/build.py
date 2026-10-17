# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build one arm's Levanter SFT rows from #207 canonical traces (task #211).

Inputs: one or more rebuild_traces.py outputs (one row per conversation of a Harbor trial attempt; task/207 @
fc55028ddd or later), the pool v2 task table, the frozen
task #202 replay slice, and for RS-multi and relay the plain arm's built prefix (it fixes the task slots and B).

Rows follow task #202: input_ids padded to 32,768 with pad id 128001, loss_weights shifted so weight i scores token
i + 1, zero-loss padding rows to a multiple of 16, nothing truncated. The replay slice is reused unchanged in every
arm; its rows are mixed with the arm's rows by a seeded shuffle.

Outputs under --output: train-prebuilt.jsonl.gz (plain, relay) or train-epoch-<e>-prebuilt.jsonl.gz (RS-multi, one
file per epoch, same row count), train-index.jsonl (row -> source), selection.jsonl (task or slot -> chosen trial),
dropped.jsonl, and manifest.json with per-family supervised and context tokens and file hashes.

Usage:
    python -m experiments.post_training.sft_shootout_rows.build --arm plain --traces <rebuilt.jsonl> ... \\
        --pool <tasks.jsonl> --replay <#202 replay prefix> --output <prefix>
"""

import argparse
import collections
import gzip
import hashlib
import json
import random
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.post_training.sft_shootout_rows.arms import (
    SEED,
    Attempt,
    first_passing,
    relay_subsample,
    rs_multi_epochs,
)
from experiments.post_training.sft_shootout_rows.traces import TEACHER, render_trace
from experiments.post_training.sweep_data.prepare_agentic import (
    BATCH,
    PAD_TOKEN,
    STUDENT_MODEL,
    STUDENT_REVISION,
    TOKENIZER_FILES,
    load_tokenizer,
)
from experiments.post_training.sweep_data.render import training_row
from experiments.post_training.sweep_data.select import CONTEXT

POOL_MANIFEST_SHA256 = "ae4347bdec1976e50cd3fd4a377d6200cce7cba6d36de47ed71938600b15f6e2"
EPOCHS = 3


def load_pool(uri: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in StoragePath(uri).read_text().splitlines()]
    return {row["task_id"]: row for row in rows}


def load_traces(uris: list[str]) -> list[dict]:
    traces = []
    for uri in uris:
        traces += [json.loads(line) | {"source": uri} for line in StoragePath(uri).read_text().splitlines()]
    keys = [(trace["trial_id"], trace.get("conversation", 0)) for trace in traces]
    if len(set(keys)) != len(keys):
        raise ValueError("A (trial id, conversation) pair appears twice across the trace files")
    return traces


def candidates(arm: str, traces: list[dict], pool: dict[str, dict], tokenizer) -> tuple[dict, list[dict]]:
    """Render every attempt the arm may use; return ({trial_id: (Attempt, rendered)}, dropped records)."""
    kept, dropped = {}, []
    for trace in traces:
        reason = None
        task = pool.get(trace.get("task_id"))
        if task is None:
            reason = "task_not_in_pool"
        elif task["heldout"]:
            reason = "heldout_task"
        elif task["family"] != trace["family"]:
            reason = "family_mismatch"
        elif trace["relay"] != (arm == "relay"):
            reason = "relay_run" if trace["relay"] else "plain_run"
        elif trace.get("n_conversations", 1) > 1:
            # Terminus-2 summarized this trial's history: the trial does not fit one 32k row.
            reason = "summarized_trial"
        elif trace["problems"]:
            reason = "trace_problems"
        elif not trace["passed"]:
            reason = "not_passed"
        if reason is None:
            authors = [m.get("author") for m in trace["messages"] if m["role"] == "assistant"]
            if TEACHER not in authors:
                reason = "no_teacher_turn"
            elif arm != "relay" and set(authors) != {TEACHER}:
                reason = "student_turn_in_plain_run"
        if reason is None:
            try:
                rendered = render_trace(tokenizer, trace["messages"], trace.get("tools"))
            except ValueError as error:
                reason = f"render_error: {error}"
            else:
                if len(rendered.ids) > CONTEXT:
                    reason = "over_context"
        if reason:
            dropped.append({"trial_id": trace["trial_id"], "task_id": trace.get("task_id"), "reason": reason})
            continue
        attempt = Attempt(
            task_id=trace["task_id"],
            family=trace["family"],
            trial_id=trace["trial_id"],
            sample=trace["sample"],
            retry=trace["retry"],
            supervised=rendered.supervised,
            context=len(rendered.ids),
            authors=rendered.authors,
        )
        kept[trace["trial_id"]] = (attempt, rendered, trace)
    return kept, dropped


def replay_rows(prefix: str) -> tuple[list[str], list[dict], dict]:
    root = StoragePath(prefix)
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    files = {}
    for name in ("replay-prebuilt.jsonl.gz", "replay-rows.jsonl"):
        files[name] = (root / name).read_bytes()
        if hashlib.sha256(files[name]).hexdigest() != manifest["output_hashes"][name]:
            raise ValueError(f"Replay file changed: {name}")
    lines = gzip.decompress(files["replay-prebuilt.jsonl.gz"]).decode().splitlines()[: manifest["rows"]]
    index = []
    for line in files["replay-rows.jsonl"].decode().splitlines():
        entry = json.loads(line)
        if "row" in entry:
            entry["row_in_group"] = entry.pop("row")
        index.append({"family": "replay"} | entry)
    info = {
        "uri": prefix,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "supervised_tokens": manifest["supervised_tokens"],
    }
    return lines, index, info


def write_split(path: Path, entries: list[tuple[str, dict]], seed: str) -> tuple[list[dict], dict]:
    """Shuffle (row json, index entry) pairs with `seed`, write rows plus zero-loss padding, re-read and check."""
    order = list(range(len(entries)))
    random.Random(seed).shuffle(order)
    padding = -len(entries) % BATCH
    index = []
    with gzip.open(path, "wt", compresslevel=6) as stream:
        for row_number, position in enumerate(order):
            line, entry = entries[position]
            stream.write(line + "\n")
            index.append({"row": row_number} | entry)
        empty = json.dumps({"input_ids": [PAD_TOKEN] * CONTEXT, "loss_weights": [0] * CONTEXT}, separators=(",", ":"))
        for _ in range(padding):
            stream.write(empty + "\n")
    supervised = collections.Counter()
    context = collections.Counter()
    with gzip.open(path, "rt") as stream:
        for row_number, line in enumerate(stream):
            row = json.loads(line)
            if len(row["input_ids"]) != CONTEXT or len(row["loss_weights"]) != CONTEXT:
                raise ValueError(f"Bad row shape at {row_number}")
            if row_number >= len(index):
                if any(row["loss_weights"]):
                    raise ValueError("Padding row carries loss")
                continue
            entry = index[row_number]
            if sum(row["loss_weights"]) != entry["supervised_tokens"] or any(
                row["loss_weights"][entry["context_tokens"] - 1 :]
            ):
                raise ValueError(f"Row {row_number} differs from its index entry")
            supervised[entry["family"]] += entry["supervised_tokens"]
            context[entry["family"]] += entry["context_tokens"]
    return index, {
        "rows": len(index),
        "padding_rows": padding,
        "steps": (len(index) + padding) // BATCH,
        "supervised_tokens": dict(supervised),
        "context_tokens": dict(context),
        "supervised_total": sum(supervised.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("plain", "rs-multi", "relay"), required=True)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--pool", required=True, help="pool v2 tasks.jsonl")
    parser.add_argument("--pool-manifest", required=True, help="pool v2 manifest.json (hash-checked)")
    parser.add_argument("--replay", required=True, help="task #202 replay component prefix")
    parser.add_argument("--plain", help="built plain arm prefix (required for rs-multi and relay)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-dir", help="local v0 tokenizer files (hash-checked); default: the regional copy")
    parser.add_argument("--test-pool", action="store_true", help="skip the pool v2 hash checks (local validation only)")
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    pool = load_pool(args.pool)
    if not args.test_pool:
        pool_manifest = StoragePath(args.pool_manifest).read_bytes()
        if hashlib.sha256(pool_manifest).hexdigest() != POOL_MANIFEST_SHA256:
            raise ValueError("Pool manifest is not pool v2")
        pool_hash = hashlib.sha256(StoragePath(args.pool).read_bytes()).hexdigest()
        if pool_hash != json.loads(pool_manifest)["file_sha256"]["tasks.jsonl"]:
            raise ValueError("tasks.jsonl does not match the pool manifest")
    plain = json.loads((StoragePath(args.plain) / "manifest.json").read_text()) if args.plain else None
    if args.arm != "plain" and plain is None:
        raise ValueError("--plain is required for rs-multi and relay")

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        audit = hf_hub_download(STUDENT_MODEL, "provenance/campaign3/audit.json", revision=STUDENT_REVISION)
        expected = {name: entry["sha256"] for name, entry in json.loads(Path(audit).read_text())["files"].items()}
        if args.tokenizer_dir:
            for name in TOKENIZER_FILES:
                if hashlib.sha256((Path(args.tokenizer_dir) / name).read_bytes()).hexdigest() != expected[name]:
                    raise ValueError(f"Local tokenizer file differs from the audit: {name}")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        else:
            (directory / "tokenizer").mkdir()
            tokenizer = load_tokenizer(directory / "tokenizer", expected)
        traces = load_traces(args.traces)
        kept, dropped = candidates(args.arm, traces, pool, tokenizer)
        attempts = [attempt for attempt, _, _ in kept.values()]
        replay_lines, replay_index, replay_info = replay_rows(args.replay)
        replay_entries = list(zip(replay_lines, replay_index, strict=True))

        def entry(attempt: Attempt, extra: dict) -> tuple[str, dict]:
            _, rendered, trace = kept[attempt.trial_id]
            line = json.dumps(training_row(rendered.ids, rendered.mask, CONTEXT, PAD_TOKEN), separators=(",", ":"))
            return (
                line,
                {
                    "family": attempt.family,
                    "task_id": attempt.task_id,
                    "trial_id": attempt.trial_id,
                    "trial_name": trace.get("trial_name"),
                    "sample": attempt.sample,
                    "retry": attempt.retry,
                    "run": trace.get("run"),
                    "turns": dict(collections.Counter(attempt.authors)),
                    "supervised_tokens": attempt.supervised,
                    "context_tokens": attempt.context,
                }
                | extra,
            )

        out = directory / "out"
        out.mkdir()
        selection = []
        splits = {}
        if args.arm == "plain":
            chosen = first_passing(attempts)
            selection = [
                {"task_id": task, "trial_id": a.trial_id, "sample": a.sample} for task, a in sorted(chosen.items())
            ]
            rows = [entry(a, {}) for _, a in sorted(chosen.items())] + replay_entries
            index, splits["train"] = write_split(out / "train-prebuilt.jsonl.gz", rows, f"{SEED}:plain")
            b_tokens = sum(a.supervised for a in chosen.values())
        elif args.arm == "rs-multi":
            tasks = sorted(plain["task_slots"])
            missing = [task for task in tasks if not any(a.task_id == task for a in attempts)]
            if missing:
                raise ValueError(f"Plain task slots without RS candidates: {missing[:5]}")
            plan = rs_multi_epochs([a for a in attempts if a.task_id in set(tasks)], tasks, EPOCHS)
            index = []
            for epoch, chosen in enumerate(plan):
                selection += [
                    {"epoch": epoch, "task_id": task, "trial_id": a.trial_id, "sample": a.sample}
                    for task, a in sorted(chosen.items())
                ]
                rows = [entry(a, {"epoch": epoch}) for _, a in sorted(chosen.items())] + replay_entries
                epoch_index, splits[f"epoch-{epoch}"] = write_split(
                    out / f"train-epoch-{epoch}-prebuilt.jsonl.gz", rows, f"{SEED}:rs-multi:shuffle:{epoch}"
                )
                index += [{"epoch": epoch} | item for item in epoch_index]
            b_tokens = plain["B_supervised_tokens"]
        else:
            b_tokens = plain["B_supervised_tokens"]
            chosen_episodes, _ = relay_subsample(attempts, b_tokens)
            selection = [{"trial_id": a.trial_id, "task_id": a.task_id, "sample": a.sample} for a in chosen_episodes]
            rows = [entry(a, {}) for a in chosen_episodes] + replay_entries
            index, splits["train"] = write_split(out / "train-prebuilt.jsonl.gz", rows, f"{SEED}:relay")

        (out / "train-index.jsonl").write_text("".join(json.dumps(item) + "\n" for item in index))
        (out / "selection.jsonl").write_text("".join(json.dumps(item) + "\n" for item in selection))
        (out / "dropped.jsonl").write_text("".join(json.dumps(item) + "\n" for item in dropped))
        heldout_rows = sum(1 for item in index if item.get("task_id") and pool[item["task_id"]]["heldout"])
        if heldout_rows:
            raise ValueError(f"{heldout_rows} rows come from held-out tasks")
        manifest = {
            "task": 211,
            "arm": args.arm,
            "student_model": STUDENT_MODEL,
            "student_revision": STUDENT_REVISION,
            "pad_token_id": PAD_TOKEN,
            "max_sequence_tokens": CONTEXT,
            "batch_size": BATCH,
            "seed": SEED,
            "pool_manifest_sha256": None if args.test_pool else POOL_MANIFEST_SHA256,
            "test_pool": args.test_pool,
            "traces": args.traces,
            "candidates": {
                "kept": len(kept),
                "dropped": dict(collections.Counter(d["reason"].split(":")[0] for d in dropped)),
            },
            "B_supervised_tokens": b_tokens,
            "task_slots": sorted({item["task_id"] for item in selection}) if args.arm == "plain" else None,
            "splits": splits,
            "replay": (
                replay_info
                | {
                    "share_of_supervised": {
                        name: s["supervised_tokens"].get("replay", 0) / s["supervised_total"]
                        for name, s in splits.items()
                    },
                    "ratio_to_B": replay_info["supervised_tokens"] / b_tokens if b_tokens else None,
                }
            ),
            "selection_rule": {
                "plain": "first passing attempt per eligible task by sample index",
                "rs-multi": (
                    f"plain's task slots; per epoch (seed '{SEED}:rs-multi:<epoch>') one passing attempt per slot, "
                    "without replacement until a task's passes are used, then refilled"
                ),
                "relay": f"passing relay episodes with >=1 GLM turn, seeded order '{SEED}:relay', whole episodes to B",
            }[args.arm],
            "loss_mask": "assistant turns authored by glm; v0 turns, prompts and tool output are context",
            "heldout_rows": 0,
            "output_hashes": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir())
            },
        }
        if args.arm == "relay":
            manifest["relay_supervised_tokens"] = splits["train"]["supervised_total"] - replay_info["supervised_tokens"]
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        output.mkdirs()
        for path in sorted(out.iterdir(), key=lambda path: path.name == "manifest.json"):
            (output / path.name).write_bytes(path.read_bytes())
            if path.name != "manifest.json":
                if hashlib.sha256((output / path.name).read_bytes()).hexdigest() != manifest["output_hashes"][path.name]:
                    raise ValueError(f"Upload hash mismatch: {path.name}")
        print(
            "ARM_BUILT " + json.dumps({k: v for k, v in manifest.items() if k not in ("task_slots", "traces")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
