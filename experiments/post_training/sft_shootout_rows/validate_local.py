# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local end-to-end check of the arm builder on #207 dry-run traces (task #211).

Builds plain, RS-multi and relay from local trace files with a test pool and a two-row stand-in replay slice, then
re-reads every row independently of the builder: decodes the trained tokens of each row and checks that each trained
span is exactly one teacher (glm) assistant turn as the template renders it, that student (v0) turns carry no loss,
that no weight sits past a row's content or on padding rows, and that no held-out task appears.

Synthetic summarized trials (n_conversations 2) check that they are dropped. The relay input is synthetic: a copy
of a plain trace whose first assistant turns are relabelled as v0, which exercises the per-author mask without a live
relay run.

Usage:
    python -m experiments.post_training.sft_shootout_rows.validate_local --traces <rebuilt.jsonl> ... \\
        --pool <pool tasks.jsonl> --tokenizer-dir <v0 tokenizer> --work <dir>
"""

import argparse
import copy
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

from experiments.post_training.sft_shootout_rows.traces import STUDENT, TEACHER, template_messages
from experiments.post_training.sweep_data.prepare_agentic import PAD_TOKEN
from experiments.post_training.sweep_data.render import training_row
from experiments.post_training.sweep_data.select import CONTEXT


def stand_in_replay(tokenizer, root: Path) -> None:
    root.mkdir(parents=True)
    rows, index = [], []
    for number, (question, answer) in enumerate((("What is 2+2?", "4."), ("Name a prime.", "7."))):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids, mask = list(encoded["input_ids"]), list(encoded["assistant_masks"])
        rows.append(json.dumps(training_row(ids, mask, CONTEXT, PAD_TOKEN), separators=(",", ":")))
        index.append(
            {
                "partition": "stand-in",
                "file": "local",
                "byte_offset": number,
                "sha256": "",
                "supervised_tokens": sum(mask[1:]),
                "context_tokens": len(ids),
            }
        )
    files = {
        "replay-prebuilt.jsonl.gz": gzip.compress(("\n".join(rows) + "\n").encode()),
        "replay-rows.jsonl": "".join(json.dumps(item) + "\n" for item in index).encode(),
    }
    for name, data in files.items():
        (root / name).write_bytes(data)
    manifest = {
        "rows": len(rows),
        "padding_rows": 0,
        "supervised_tokens": sum(item["supervised_tokens"] for item in index),
        "output_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def check_arm(tokenizer, prefix: Path, traces: dict[str, dict], heldout: set[str]) -> dict:
    manifest = json.loads((prefix / "manifest.json").read_text())
    validation = manifest["arm"] == "validation"
    index_name = "validation-index.jsonl" if validation else "train-index.jsonl"
    index = [json.loads(line) for line in (prefix / index_name).read_text().splitlines()]
    files = sorted(prefix.glob("validation-*-prebuilt.jsonl.gz" if validation else "train*-prebuilt.jsonl.gz"))
    checked = 0
    for path in files:
        if validation:
            split = path.name.removesuffix("-prebuilt.jsonl.gz")
            entries = [item for item in index if item["split"] == split]
        else:
            epoch = int(path.name.split("-")[2]) if "epoch" in path.name else None
            entries = [item for item in index if item.get("epoch") == epoch]
        with gzip.open(path, "rt") as stream:
            for row_number, line in enumerate(stream):
                row = json.loads(line)
                ids, weights = row["input_ids"], row["loss_weights"]
                if row_number >= len(entries):
                    assert not any(weights) and set(ids) == {PAD_TOKEN}, "padding row with content or loss"
                    continue
                entry = entries[row_number]
                if "task_id" in entry:
                    assert (entry["task_id"] in heldout) == validation, "row's held-out status does not match its arm"
                length = entry["context_tokens"]
                assert not any(weights[length - 1 :]), "loss past the row's content"
                if entry["family"] == "replay":
                    continue
                trace = traces[entry["trial_id"]]
                messages, authors = template_messages(trace["messages"])
                trained = [index for index in range(length - 1) if weights[index]]
                spans, start = [], None
                for position in [*trained, None]:
                    if start is None:
                        start = previous = position
                    elif position is not None and position == previous + 1:
                        previous = position
                    else:
                        spans.append(tokenizer.decode(ids[start + 1 : previous + 2]))
                        start = previous = position
                expected = []
                for message, author in zip([m for m in messages if m["role"] == "assistant"], authors, strict=True):
                    if author != TEACHER:
                        continue
                    body = message.get("content") or ""
                    expected.append((message.get("reasoning_content") or "", body.strip()))
                assert len(spans) == len(expected), f"{len(spans)} trained spans for {len(expected)} teacher turns"
                for span, (reasoning, body) in zip(spans, expected, strict=True):
                    assert span.endswith("<|eot_id|>"), "trained span does not end at the turn's end"
                    if reasoning:
                        assert span.startswith("<|start_think|>") and reasoning in span, "reasoning not trained"
                    assert body[:200] in span, "turn text not inside its trained span"
                if STUDENT in authors:
                    for message, author in zip([m for m in messages if m["role"] == "assistant"], authors, strict=True):
                        text = (message.get("content") or "").strip()[:200]
                        if author == STUDENT and text:
                            assert all(text not in span for span in spans), "student turn text inside a trained span"
                checked += 1
    return {
        "arm": manifest["arm"],
        "files": [path.name for path in files],
        "agentic_rows_checked": checked,
        "splits": manifest["splits"],
        "dropped": manifest["candidates"]["dropped"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--add-to-pool", nargs="*", default=[], help="task_id:family pairs added as non-held-out")
    args = parser.parse_args()
    work = args.work
    work.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    pool_rows = [json.loads(line) for line in Path(args.pool).read_text().splitlines()]
    for pair in args.add_to_pool:
        task, family = pair.split(":")
        pool_rows.append({"task_id": task, "family": family, "heldout": False})
    (work / "pool.jsonl").write_text("".join(json.dumps(row) + "\n" for row in pool_rows))
    heldout = {row["task_id"] for row in pool_rows if row["heldout"]}
    stand_in_replay(tokenizer, work / "replay")

    traces = [json.loads(line) for path in args.traces for line in Path(path).read_text().splitlines()]
    relay = []
    for trace in traces:
        synthetic = copy.deepcopy(trace) | {
            "relay": True,
            "trial_id": trace["trial_id"] + "-relay",
            "run": "synthetic-relay",
        }
        assistant = [m for m in synthetic["messages"] if m["role"] == "assistant"]
        for message in assistant[: max(1, len(assistant) // 2)]:
            message["author"] = STUDENT
        relay.append(synthetic)
    (work / "relay.jsonl").write_text("".join(json.dumps(trace) + "\n" for trace in relay))
    summarized = [
        copy.deepcopy(trace)
        | {"trial_id": trace["trial_id"] + "-summarized", "conversation": part, "n_conversations": 2}
        for trace in traces
        for part in (0, 1)
    ]
    (work / "summarized.jsonl").write_text("".join(json.dumps(trace) + "\n" for trace in summarized))
    touching = []
    for trace in traces:
        synthetic = copy.deepcopy(trace) | {"trial_id": trace["trial_id"] + "-toolenv"}
        observation = next(
            m for m in synthetic["messages"] if m["role"] in ("tool", "user") and m is not synthetic["messages"][0]
        )
        observation["content"] = "/root/.local/share/uv/tools/mini-swe-agent/lib/python3.12/site-packages/x.py\n" + (
            observation.get("content") or ""
        )
        touching.append(synthetic)
    (work / "toolenv.jsonl").write_text("".join(json.dumps(trace) + "\n" for trace in touching))
    plain_inputs = [*args.traces, str(work / "summarized.jsonl"), str(work / "toolenv.jsonl")]
    heldout_by_family = {}
    for row in pool_rows:
        if row["heldout"]:
            heldout_by_family.setdefault(row["family"], row["task_id"])
    held = [
        copy.deepcopy(trace)
        | {"trial_id": trace["trial_id"] + "-heldout", "task_id": heldout_by_family[trace["family"]]}
        for trace in traces
        if trace["family"] in heldout_by_family
    ]
    (work / "heldout.jsonl").write_text("".join(json.dumps(trace) + "\n" for trace in held))
    by_id = {trace["trial_id"]: trace for trace in traces + relay + held}

    def build(arm: str, trace_files: list[str], plain: Path | None = None) -> Path:
        output = work / arm
        arm = arm.split("-with-")[0]
        command = [
            sys.executable,
            "-m",
            "experiments.post_training.sft_shootout_rows.build",
            "--arm",
            arm,
            "--traces",
            *trace_files,
            "--pool",
            str(work / "pool.jsonl"),
            "--pool-manifest",
            "unused",
            "--replay",
            str(work / "replay"),
            "--tokenizer-dir",
            args.tokenizer_dir,
            "--test-pool",
            "--output",
            str(output),
        ]
        if plain:
            command += ["--plain", str(plain)]
        subprocess.run(command, check=True)
        return output

    plain = build("plain", plain_inputs)
    reports = [check_arm(tokenizer, plain, by_id, heldout)]
    assert reports[0]["dropped"].get("summarized_trial") == len(summarized), "summarized trials were not dropped"
    assert reports[0]["dropped"].get("touches_agent_tool_env") == len(touching), "tool-env traces were not dropped"
    reports.append(check_arm(tokenizer, build("rs-multi", plain_inputs, plain), by_id, heldout))
    reports.append(check_arm(tokenizer, build("relay", [str(work / "relay.jsonl")], plain), by_id, heldout))
    validation = build("validation", [*plain_inputs, str(work / "heldout.jsonl")])
    reports.append(check_arm(tokenizer, validation, by_id, heldout))
    mixed = build("plain-with-heldout-input", [*plain_inputs, str(work / "heldout.jsonl")])
    mixed_dropped = json.loads((mixed / "manifest.json").read_text())["candidates"]["dropped"]
    assert mixed_dropped.get("heldout_task") == len(held), "held-out traces were not dropped from training"
    print("LOCAL_VALIDATION " + json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
