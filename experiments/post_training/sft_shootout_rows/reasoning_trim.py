# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning-trimmed variants of a built plain arm, for a verbosity control (dry builds only).

The input is a built plain prefix. Each variant keeps plain's tasks, its first-passing selection, its replay rows and
its epoch orders. Only agentic turns whose reasoning is longer than a cap change. Reasoning here means the tokens
strictly inside <|start_think|> ... <|end_think|>. The block is the whole span, both markers included.

- mask: the block stays in the row as context, with loss weight 0; the turn's action/content stays trained.
- drop: the block is removed from the row text, so the turn renders as v0's template renders a turn without
  reasoning (no think block). Later turns see the shorter context; the row is re-padded.

Replay rows and padding rows are copied unchanged. Each row is rebuilt from its index entry, as in the builder: the
content is input_ids[:context_tokens], and the per-token mask is [0] + loss_weights. A rebuilt row with nothing
trimmed is checked to be byte-identical to plain's.

Checks on every changed row:
- mask: input_ids are unchanged, and weights change only inside over-cap blocks.
- drop: the decoded text equals plain's decoded text with exactly those blocks removed, and the trained tokens
  outside the blocks are the same sequence.

Outputs go to <output>/<mode>-cap<cap>/: three epoch files, train-index.jsonl, manifest.json and a
DRY_RUN_NOT_FOR_TRAINING marker.

    python -m experiments.post_training.sft_shootout_rows.reasoning_trim --plain <plain prefix> --output <dry prefix>
"""

import argparse
import contextlib
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.build import END_THINK, EPOCHS, START_THINK, v0_tokenizer
from experiments.post_training.sweep_data.prepare_agentic import (
    CONTEXT,
    PAD_TOKEN,
    open_prebuilt,
    output_hashes,
    read_component,
)
from experiments.post_training.sweep_data.render import training_row

MODES = ("mask", "drop")
CAPS = (1_000, 2_000, 4_000)


def row_line(row: dict) -> str:
    return json.dumps(row, separators=(",", ":"))


def think_blocks(ids: list[int], mask: list[int]) -> list[tuple[int, int, int]]:
    """(start, end, reasoning length) of each trained think block; positions inclusive of both markers."""
    blocks, position = [], 0
    while position < len(ids):
        if ids[position] == START_THINK and mask[position]:
            end = ids.index(END_THINK, position)
            if not all(mask[position : end + 1]):
                raise ValueError("A think block is only partly trained")
            blocks.append((position, end, end - position - 1))
            position = end
        position += 1
    return blocks


def trim(ids: list[int], mask: list[int], blocks: list[tuple[int, int, int]], mode: str):
    """Apply the mode to the given blocks; return (ids, mask)."""
    if mode == "mask":
        mask = list(mask)
        for start, end, _ in blocks:
            mask[start : end + 1] = [0] * (end + 1 - start)
        return ids, mask
    removed = {p for start, end, _ in blocks for p in range(start, end + 1)}
    keep = [p for p in range(len(ids)) if p not in removed]
    return [ids[p] for p in keep], [mask[p] for p in keep]


def check(tokenizer, ids, mask, new_ids, new_mask, blocks, mode) -> None:
    inside = {p for start, end, _ in blocks for p in range(start, end + 1)}
    if mode == "mask":
        if new_ids != ids or any(new_mask[p] != mask[p] for p in range(len(ids)) if p not in inside):
            raise ValueError("Masking changed tokens or weights outside the over-cap blocks")
        if any(new_mask[p] for p in inside):
            raise ValueError("An over-cap block is still trained")
        return
    expected, last = [], 0
    for start, end, _ in blocks:
        expected.append(tokenizer.decode(ids[last:start]))
        last = end + 1
    expected.append(tokenizer.decode(ids[last:]))
    if tokenizer.decode(new_ids) != "".join(expected):
        raise ValueError("Re-decoded text differs from plain's text without the dropped blocks")
    outside = [ids[p] for p in range(len(ids)) if mask[p] and p not in inside]
    if [new_ids[p] for p in range(len(new_ids)) if new_mask[p]] != outside:
        raise ValueError("Dropping changed the trained tokens outside the blocks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    plain_root = StoragePath(args.plain)
    plain, plain_bytes, files = read_component(plain_root)
    if plain["arm"] != "plain":
        raise ValueError("--plain must be a plain arm prefix")
    index = [json.loads(line) for line in files["train-index.jsonl"].decode().splitlines()]
    variants = [(mode, cap) for mode in MODES for cap in CAPS]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        tokenizer = v0_tokenizer(None, directory)
        outs = {v: directory / f"{v[0]}-cap{v[1]}" for v in variants}
        for out in outs.values():
            out.mkdir()
        new_index = {v: [] for v in variants}
        splits = {v: {} for v in variants}
        turns = {v: {} for v in variants}  # per family: trained reasoning length of every think-bearing or plain turn
        affected = {v: {} for v in variants}
        for epoch in range(EPOCHS):
            name = f"train-epoch-{epoch}-prebuilt.jsonl.gz"
            lines = gzip.decompress(files[name]).decode().splitlines()
            entries = [item for item in index if item["epoch"] == epoch]
            with contextlib.ExitStack() as stack:
                streams = {v: stack.enter_context(open_prebuilt(outs[v] / name)) for v in variants}
                digests = {v: hashlib.sha256() for v in variants}
                totals = {v: ({}, {}) for v in variants}
                for number, line in enumerate(lines):
                    entry = entries[number] if number < len(entries) else None
                    if entry is None or entry["family"] == "replay":
                        for v in variants:
                            streams[v].write(line + "\n")
                            digests[v].update((line + "\n").encode())
                            if entry is not None:
                                new_index[v].append(entry)
                                for target, key in zip(totals[v], ("supervised_tokens", "context_tokens"), strict=True):
                                    target[entry["family"]] = target.get(entry["family"], 0) + entry[key]
                        continue
                    row = json.loads(line)
                    ids = row["input_ids"][: entry["context_tokens"]]
                    mask = [0, *row["loss_weights"][: entry["context_tokens"] - 1]]
                    if row_line(training_row(ids, mask, CONTEXT, PAD_TOKEN)) != line:
                        raise ValueError(f"epoch {epoch} row {number}: rebuilding plain's row does not reproduce it")
                    blocks = think_blocks(ids, mask)
                    family = entry["family"]
                    for v in variants:
                        mode, cap = v
                        over = [b for b in blocks if b[2] > cap]
                        new_ids, new_mask = trim(ids, mask, over, mode)
                        if over:
                            check(tokenizer, ids, mask, new_ids, new_mask, over, mode)
                        new_line = row_line(training_row(new_ids, new_mask, CONTEXT, PAD_TOKEN))
                        streams[v].write(new_line + "\n")
                        digests[v].update((new_line + "\n").encode())
                        new_entry = entry | {"supervised_tokens": sum(new_mask[1:]), "context_tokens": len(new_ids)}
                        new_index[v].append(new_entry | {"trimmed_blocks": len(over)})
                        for target, key in zip(totals[v], ("supervised_tokens", "context_tokens"), strict=True):
                            target[family] = target.get(family, 0) + new_entry[key]
                        if epoch == 0:
                            lengths = turns[v].setdefault(family, [])
                            lengths += [0 if b[2] > cap else b[2] for b in blocks]
                            lengths += [0] * (entry["turns"]["glm"] - len(blocks))
                            counts = affected[v].setdefault(family, [0, 0])
                            counts[0] += len(over)
                            counts[1] += entry["turns"]["glm"]
            for v in variants:
                split = plain["splits"][f"epoch-{epoch}"]
                supervised, context = totals[v]
                splits[v][f"epoch-{epoch}"] = {
                    key: split[key] for key in ("rows", "padding_rows", "padded_rows", "permutation_seed", "steps")
                } | {
                    "rows_sha256": digests[v].hexdigest(),
                    "supervised_tokens": supervised,
                    "context_tokens": context,
                    "supervised_total": sum(supervised.values()),
                }
        for v in variants:
            mode, cap = v
            out = outs[v]
            (out / "train-index.jsonl").write_text("".join(json.dumps(item) + "\n" for item in new_index[v]))
            (out / "DRY_RUN_NOT_FOR_TRAINING").write_text("reasoning-trim dry build; not an arm input\n")
            stats = {}
            for family, lengths in sorted(turns[v].items()):
                values = np.array(lengths)
                stats[family] = {
                    "turns": len(values),
                    "turns_trimmed": affected[v][family][0],
                    "share_of_turns_trimmed": affected[v][family][0] / affected[v][family][1],
                    "trained_reasoning_tokens": int(values.sum()),
                    "per_turn_trained_reasoning": {
                        "median": float(np.median(values)),
                        "p90": float(np.percentile(values, 90)),
                        "p99": float(np.percentile(values, 99)),
                        "max": int(values.max()),
                    },
                }
            epoch0 = splits[v]["epoch-0"]
            manifest = {
                "arm": "plain-reasoning-trim",
                "dry_run": True,
                "mode": mode,
                "cap_reasoning_tokens": cap,
                "rule": (
                    f"agentic turns with more than {cap} reasoning tokens: the think block (markers included) is "
                    + ("kept as context with zero loss" if mode == "mask" else "removed from the row text")
                    + "; replay and every other turn unchanged"
                ),
                "plain_prefix": args.plain,
                "plain_manifest_sha256": hashlib.sha256(plain_bytes).hexdigest(),
                "B_supervised_tokens": sum(t for f, t in epoch0["supervised_tokens"].items() if f != "replay"),
                "plain_B_supervised_tokens": plain["B_supervised_tokens"],
                "replay": {"rows_sha256": plain["replay"]["rows_sha256"], "note": "plain's replay rows, unchanged"},
                "splits": splits[v],
                "turn_stats": stats,
                "output_hashes": output_hashes(out),
            }
            (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
            target = output / out.name
            target.mkdirs()
            for path in sorted(out.iterdir(), key=lambda path: path.name == "manifest.json"):
                (target / path.name).write_bytes(path.read_bytes())
            print("TRIM_BUILT " + json.dumps({k: v for k, v in manifest.items() if k != "output_hashes"}), flush=True)


if __name__ == "__main__":
    main()
