# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local end-to-end check of the arm builder on dry-run generation traces.

Builds plain, RS-multi, relay, recovery and validation from local trace files with a test pool and a two-row
stand-in replay slice, then re-reads every row independently of the builder: decodes the trained tokens of each row
and checks that each trained span is exactly one teacher (glm) assistant turn as the template renders it, that
student (v0) turns carry no loss, that no weight sits past a row's content or on padding rows, and that no held-out
task appears.

Synthetic summarized trials (n_conversations 2) check that they are dropped. The relay input is synthetic: a copy
of a plain trace whose first assistant turns are relabelled as v0, which exercises the per-author mask without a live
relay run.

Usage:
    python -m experiments.post_training.sft_shootout_rows.validate_local --traces <rebuilt.jsonl> ... \\
        --pool <pool tasks.jsonl> --tokenizer-dir <v0 tokenizer> --work <dir>
"""

import argparse
import base64
import collections
import copy
import gzip
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from transformers import AutoTokenizer

from experiments.post_training.sft_shootout_rows.arms import SEED
from experiments.post_training.sft_shootout_rows.build import candidates
from experiments.post_training.sft_shootout_rows.traces import STUDENT, TEACHER, template_messages
from experiments.post_training.sweep_data.prepare_agentic import PAD_TOKEN
from experiments.post_training.sweep_data.render import training_row
from experiments.post_training.sweep_data.select import CONTEXT

THINK_MARKUP = re.compile(r"<\|(?:start|end)_think\|>")


def fake_capability_token() -> str:
    """A syntactically valid JWT with the proxy audience, for testing the secret guard (not a real credential)."""

    def encode(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'aud': 'iris-proxy', 'sub': 'test'})}.dGVzdA"


def stand_in_replay(tokenizer, root: Path) -> None:
    root.mkdir(parents=True)
    rows, index = [], []
    # Enough rows and tokens that the B / 4 cut takes a strict prefix of the slice.
    pairs = [(f"Count from {n} to {n + 40} in words.", " ".join(f"step {n + k}" for k in range(41))) for n in range(64)]
    for number, (question, answer) in enumerate(pairs):
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
                    # Spans already match the teacher turns one to one (count and text). This catches student text in a
                    # span, ignoring think markup alone (e.g. a student turn that is just "<|end_think|>") and text the
                    # teacher itself repeats verbatim in one of its turns.
                    teacher_text = "\n".join(reasoning + "\n" + body for reasoning, body in expected)
                    for message, author in zip([m for m in messages if m["role"] == "assistant"], authors, strict=True):
                        text = THINK_MARKUP.sub("", message.get("content") or "").strip()[:200]
                        if author == STUDENT and text and text not in teacher_text:
                            assert all(text not in span for span in spans), "student turn text inside a trained span"
                checked += 1
    return {
        "arm": manifest["arm"],
        "files": [path.name for path in files],
        "agentic_rows_checked": checked,
        "splits": manifest["splits"],
        "dropped": manifest["candidates"]["dropped"],
    }


class Fixtures(NamedTuple):
    """Synthetic trace files written next to the real ones, and the traces each check needs."""

    plain_inputs: list[str]
    relay: list[dict]
    summarized: list[dict]
    touching: list[dict]
    secret: list[dict]
    library: list[dict]
    held: list[dict]


def write_pool(pool: str, add_to_pool: list[str], work: Path) -> list[dict]:
    """The test pool: the real pool plus library-copy test tasks and any --add-to-pool tasks, written to work."""
    pool_rows = [json.loads(line) for line in Path(pool).read_text().splitlines()]
    for family in ("tmax", "swesmith"):
        pool_rows.append(
            {"task_id": f"test-library-{family}", "family": family, "heldout": False, "import_names": ["demopkg"]}
        )
    for pair in add_to_pool:
        task, family = pair.split(":")
        pool_rows.append({"task_id": task, "family": family, "heldout": False})
    (work / "pool.jsonl").write_text("".join(json.dumps(row) + "\n" for row in pool_rows))
    return pool_rows


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def write_fixtures(trace_paths: list[str], traces: list[dict], pool_rows: list[dict], work: Path) -> Fixtures:
    """Write the synthetic relay, summarized, tool-env, secret, library-copy and held-out trace files.

    The secret traces go in their own file (secret.jsonl), outside plain_inputs: the builder's token gate refuses any
    input file that holds a capability token, before the per-trace drop rule runs.
    """
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
        if not relay:
            # A v0 turn that is only think markup appears in every teacher span; the student check must ignore it.
            assistant[0]["content"] = "<|end_think|>"
        relay.append(synthetic)
    write_jsonl(work / "relay.jsonl", relay)
    summarized = [
        copy.deepcopy(trace)
        | {"trial_id": trace["trial_id"] + "-summarized", "conversation": part, "n_conversations": 2}
        for trace in traces
        for part in (0, 1)
    ]
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
    secret = []
    for trace in traces:
        synthetic = copy.deepcopy(trace) | {"trial_id": trace["trial_id"] + "-secret"}
        observations = [m for m in synthetic["messages"][1:] if m["role"] in ("tool", "user")]
        fake = fake_capability_token()
        observations[0]["content"] = (
            f"OPENAI_API_BASE=https://iris.oa.dev/proxy/t/cluster=cw-us-east-02a/{fake}/glm-harbor.test-0000/v1\n"
            + (observations[0].get("content") or "")
        )
        secret.append(synthetic)
    write_jsonl(work / "secret.jsonl", secret)
    library = []
    for trace in traces:
        synthetic = copy.deepcopy(trace) | {
            "trial_id": trace["trial_id"] + "-library",
            "task_id": f"test-library-{trace['family']}",
        }
        observations = [m for m in synthetic["messages"][1:] if m["role"] in ("tool", "user")]
        observations[0]["content"] = (
            "/opt/miniconda3/envs/testbed/lib/python3.10/site-packages/setuptools/_vendor/demopkg/core.py\n"
            + (observations[0].get("content") or "")
        )
        library.append(synthetic)
    plain_inputs = [
        *trace_paths,
        write_jsonl(work / "summarized.jsonl", summarized),
        write_jsonl(work / "toolenv.jsonl", touching),
        write_jsonl(work / "library.jsonl", library),
    ]
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
    write_jsonl(work / "heldout.jsonl", held)
    return Fixtures(plain_inputs, relay, summarized, touching, secret, library, held)


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
    pool_rows = write_pool(args.pool, args.add_to_pool, work)
    heldout = {row["task_id"] for row in pool_rows if row["heldout"]}
    stand_in_replay(tokenizer, work / "replay")
    traces = [json.loads(line) for path in args.traces for line in Path(path).read_text().splitlines()]
    fixtures = write_fixtures(args.traces, traces, pool_rows, work)
    plain_inputs = fixtures.plain_inputs
    by_id = {trace["trial_id"]: trace for trace in traces + fixtures.relay + fixtures.held}

    def build(arm: str, trace_files: list[str], plain: Path | None = None, refusal: str | None = None) -> Path:
        """Run the builder; with `refusal`, expect it to fail with that message instead."""
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
            "--test-inputs",
            "--output",
            str(output),
        ]
        if plain:
            command += ["--plain", str(plain)]
        if refusal is None:
            subprocess.run(command, check=True)
        else:
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode != 0 and refusal in result.stderr, f"{arm} build did not refuse: {refusal}"
        return output

    replay_index = [json.loads(line) for line in (work / "replay" / "replay-rows.jsonl").read_text().splitlines()]

    def replay_used(prefix: Path) -> list[set[int]]:
        """Replay rows per training file of an arm (by stand-in offset), after checking the manifest's cut."""
        manifest = json.loads((prefix / "manifest.json").read_text())
        target = round(manifest["B_supervised_tokens"] / 4)
        order = list(range(len(replay_index)))
        random.Random(f"{SEED}:replay-order").shuffle(order)
        expected, total = set(), 0
        for position in order:
            if total >= target:
                break
            expected.add(position)
            total += replay_index[position]["supervised_tokens"]
        assert 0 < len(expected) < len(replay_index), "stand-in replay too small to test a strict cut"
        assert manifest["replay"]["target_supervised_tokens"] == target, "replay target is not round(B / 4)"
        assert manifest["replay"]["supervised_tokens"] == total, "replay tokens differ from the expected prefix"
        index = [json.loads(line) for line in (prefix / "train-index.jsonl").read_text().splitlines()]
        used = collections.defaultdict(set)
        for item in index:
            if item["family"] == "replay":
                used[item.get("epoch")].add(item["byte_offset"])
        assert all(rows == expected for rows in used.values()), "replay rows are not the seeded B / 4 prefix"
        return list(used.values())

    plain = build("plain", plain_inputs)
    reports = [check_arm(tokenizer, plain, by_id, heldout)]
    plain_replay = replay_used(plain)
    assert reports[0]["dropped"].get("summarized_trial") == len(
        fixtures.summarized
    ), "summarized trials were not dropped"
    assert reports[0]["dropped"].get("touches_agent_tool_env") == len(
        fixtures.touching
    ), "tool-env traces were not dropped"
    # The token gate refuses an input file holding a capability token; the drop rule behind it is checked directly.
    build("plain-with-secret-input", [*plain_inputs, str(work / "secret.jsonl")], refusal="Capability tokens")
    pool = {row["task_id"]: row for row in pool_rows}
    _, secret_dropped = candidates("plain", fixtures.secret, pool, tokenizer)
    proxy_drops = sum(1 for d in secret_dropped if d["reason"] == "contains_proxy_capability_token")
    assert proxy_drops == len(fixtures.secret), "capability-token traces were not dropped"
    library_drops = reports[0]["dropped"].get("touches_library_copy_of_repo", 0)
    assert library_drops == len(fixtures.library), "library-copy traces were not dropped"
    rs_multi = build("rs-multi", plain_inputs, plain)
    reports.append(check_arm(tokenizer, rs_multi, by_id, heldout))
    relay_arm = build("relay", [str(work / "relay.jsonl")], plain)
    reports.append(check_arm(tokenizer, relay_arm, by_id, heldout))
    recovery_arm = build("recovery", plain_inputs, plain)
    reports.append(check_arm(tokenizer, recovery_arm, by_id, heldout))
    recovery_manifest = json.loads((recovery_arm / "manifest.json").read_text())
    plain_index = [json.loads(line) for line in (plain / "train-index.jsonl").read_text().splitlines()]
    recovery_index = [json.loads(line) for line in (recovery_arm / "train-index.jsonl").read_text().splitlines()]
    supervised = {(i["trial_id"], i["epoch"]): i["supervised_tokens"] for i in plain_index if i.get("trial_id")}
    injected = [i for i in recovery_index if i.get("injected")]
    # The stand-in inputs hold one row per family, so round(0.2 * 1) = 0 rows are selected; test_recovery covers
    # injection itself and the dry build covers real rows. Here: the index agrees with the manifest.
    families = recovery_manifest["recovery"]["families"].values()
    assert len(injected) == 3 * sum(family["injected"] for family in families), "injected rows differ from manifest"
    assert all(
        supervised[(i["trial_id"], i["epoch"])] == i["supervised_tokens"] for i in recovery_index if i.get("trial_id")
    )
    assert (
        recovery_manifest["B_supervised_tokens"]
        == json.loads((plain / "manifest.json").read_text())["B_supervised_tokens"]
    )
    for prefix in (rs_multi, relay_arm, recovery_arm):
        assert replay_used(prefix)[0] == plain_replay[0], f"{prefix.name} replay rows differ from plain's"
    for prefix in (plain, rs_multi, relay_arm, recovery_arm):
        manifest = json.loads((prefix / "manifest.json").read_text())
        epochs = [manifest["splits"][f"epoch-{e}"] for e in range(3)]
        assert all(split["padded_rows"] % 16 == 0 for split in epochs), f"{prefix.name}: epoch not padded to 16"
        assert len({split["permutation_seed"] for split in epochs}) == 3, f"{prefix.name}: epoch seeds not distinct"
        index = [json.loads(line) for line in (prefix / "train-index.jsonl").read_text().splitlines()]
        orders = [
            [item.get("trial_id") or f"replay:{item.get('byte_offset')}" for item in index if item["epoch"] == e]
            for e in range(3)
        ]
        if prefix.name != "rs-multi":
            assert all(sorted(order) == sorted(orders[0]) for order in orders), f"{prefix.name}: epochs hold other rows"
            if len(orders[0]) > 2:
                assert len({tuple(order) for order in orders}) == 3, f"{prefix.name}: epochs share an order"
    validation = build("validation", [*plain_inputs, str(work / "heldout.jsonl")])
    reports.append(check_arm(tokenizer, validation, by_id, heldout))
    mixed = build("plain-with-heldout-input", [*plain_inputs, str(work / "heldout.jsonl")])
    mixed_dropped = json.loads((mixed / "manifest.json").read_text())["candidates"]["dropped"]
    assert mixed_dropped.get("heldout_task") == len(fixtures.held), "held-out traces were not dropped from training"
    print("LOCAL_VALIDATION " + json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
