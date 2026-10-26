# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the four-family prompt mix for identity OPD from the 16-update context-distillation checkpoint.

Families and per-batch counts (64 prompts per update, 20 updates), each split evenly by thinking mode:

| family | per batch | rows | teacher |
|---|---|---|---|
| identity | 16 | 320 | identity (v0.1 with the identity system prompt) |
| other-lab | 12 | 240 | plain (v0.1) |
| general | 20 | 400 | plain |
| tool | 16 | 320 | identity |

Every family is disjoint by normalized user text from the screens; identity and other-lab rows are also disjoint from
earlier identity training and gate sets. The other-lab, general and
tool families carry no Marin, Snowball or Open Athena text. Tool conversations share no function name and no word
8-gram with the BFCL items. Any shortfall fails the build; nothing is silently topped up.

Writes ``mixed.parquet`` and ``mix-report.json`` (input and output sha256s, counts, every rejected row and why).
"""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from taskcompendium.chat import chat_input

from experiments.post_training.identity_opd.other_lab import other_lab_questions

SEED = 20261026
PROBE_ROWS_PER_ROUTE = 4
STEPS = 20
PER_BATCH = {"identity": 16, "otherlab": 12, "general": 20, "tool": 16}
GENERAL_TYPES = {"math": 135, "code": 130, "chat": 135}
TOOL_CUTS = {"call_after_result": 160, "first_call": 80, "reply": 80}
TEACHER = {"identity": "identity", "otherlab": "plain", "general": "plain", "tool": "identity"}
BRAND = re.compile(r"marin|snowball|open[\s_-]?athena", re.IGNORECASE)
ROLES = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def user_texts(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def strings_under(value, keys=("content", "text", "prompt", "question")) -> list[str]:
    """Every string stored under one of ``keys`` anywhere in a JSON value."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys and isinstance(item, str):
                found.append(item)
            else:
                found.extend(strings_under(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.extend(strings_under(item, keys))
    return found


def excluded_texts(paths: list[Path]) -> set[str]:
    texts = set()
    for path in paths:
        values = read_jsonl(path) if path.suffix == ".jsonl" else [json.loads(path.read_text())]
        texts.update(normalize(t) for t in strings_under(values))
    return texts


def row(prompt: list[dict], family: str, thinking: bool, source: str, row_id: str) -> dict:
    mode = "thinking" if thinking else "nonthinking"
    return {
        "id": row_id,
        "prompt": prompt,
        "env_class": "prompt_only",
        "teacher_route": f"{family}_{mode}",
        "chat_template_kwargs": {"enable_thinking": thinking},
        "data_source": source,
        "tools": "",
    }


def split_thinking(rows: list[tuple[list[dict], str, str]], family: str, rng: random.Random) -> list[dict]:
    """Shuffle, then give the first half thinking on and the rest thinking off."""
    rows = list(rows)
    rng.shuffle(rows)
    half = len(rows) // 2
    return [row(p, family, i < half, source, row_id) for i, (p, source, row_id) in enumerate(rows)]


def identity_rows(pool: list[dict], excluded: set[str], rng: random.Random, rejected: list) -> list[dict]:
    """Half thinking on, half off, each row's own mode; no prompt text is chosen twice across the two halves."""
    need = PER_BATCH["identity"] * STEPS // 2
    by_mode: dict[bool, dict[str, dict]] = {True: {}, False: {}}
    for index, r in enumerate(pool):
        if not r["teacher_route"].startswith("identity"):
            continue
        if any(normalize(t) in excluded for t in user_texts(r["prompt"])):
            rejected.append({"family": "identity", "id": f"pool-{index}", "reason": "matches an excluded text"})
            continue
        key = " | ".join(normalize(t) for t in user_texts(r["prompt"]))
        by_mode[r["chat_template_kwargs"]["enable_thinking"]].setdefault(key, {**r, "id": f"pool-{index}"})
    rows, used = [], set()
    for thinking in (False, True):
        candidates = [k for k in sorted(by_mode[thinking]) if k not in used]
        if len(candidates) < need:
            raise ValueError(f"identity thinking={thinking}: {len(candidates)} rows, need {need}")
        for key in rng.sample(candidates, need):
            used.add(key)
            r = by_mode[thinking][key]
            rows.append(row(r["prompt"], "identity", thinking, r["data_source"], r["id"]))
    return rows


def otherlab_rows(excluded: set[str], rng: random.Random) -> list[dict]:
    questions = other_lab_questions()
    bad = [q["id"] for q in questions if normalize(q["text"]) in excluded or BRAND.search(q["text"])]
    if bad:
        raise ValueError(f"other-lab questions overlap an excluded text or carry brand text: {bad}")
    if len(questions) != PER_BATCH["otherlab"] * STEPS:
        raise ValueError(f"{len(questions)} other-lab questions, need {PER_BATCH['otherlab'] * STEPS}")
    return split_thinking(
        [([{"role": "user", "content": q["text"]}], "otherlab-fresh", q["id"]) for q in questions], "otherlab", rng
    )


def general_rows(replay: list[dict], excluded: set[str], rng: random.Random, rejected: list) -> list[dict]:
    chosen = []
    for kind, need in GENERAL_TYPES.items():
        candidates = []
        for r in replay:
            if r["type"] != kind:
                continue
            texts = [m["content"] for m in r["messages"]]
            if any(BRAND.search(t) for t in texts) or any(normalize(t) in excluded for t in user_texts(r["messages"])):
                rejected.append({"family": "general", "id": r["id"], "reason": "brand text or excluded text"})
                continue
            candidates.append(r)
        if len(candidates) < need:
            raise ValueError(f"general {kind}: {len(candidates)} rows, need {need}")
        chosen += [(r["messages"], f"replay-{kind}", r["id"]) for r in rng.sample(candidates, need)]
    return split_thinking(chosen, "general", rng)


def bfcl_items(directory: Path) -> tuple[set[str], set[tuple[str, ...]]]:
    """Function names and question 8-grams from the BFCL items' instruction.md files."""
    names, grams = set(), set()
    for path in sorted(directory.glob("*.md")):
        text = path.read_text()
        names.update(n.strip().lower() for n in re.findall(r"^###\s+(.+)$", text, re.MULTILINE))
        task = text.split("# Task", 1)[-1].split("## Available Functions", 1)[0]
        grams |= ngrams(task)
    return names, grams


def tool_messages(conversation: dict) -> list[dict]:
    """Hermes turns as chat messages for the serving template, which renders the tool schemas itself.

    The source system turn (hermes instructions plus the schemas as text) is dropped; the schemas go in the row's
    ``tools`` column. Assistant ``<tool_call>`` text becomes ``tool_calls`` with ids and JSON-string arguments;
    each ``<tool_response>`` becomes a tool message answering the earliest unanswered call, which the template
    renders with that call's name.
    """
    messages, pending, calls_made = [], [], 0
    for turn in conversation["conversations"]:
        role, value = ROLES[turn["from"]], turn["value"]
        if role == "system":
            continue
        if role == "assistant" and "<tool_call>" in value:
            calls = [json.loads(c) for c in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", value, re.DOTALL)]
            text = re.sub(r"<tool_call>.*?</tool_call>", "", value, flags=re.DOTALL).strip()
            tool_calls = []
            for c in calls:
                calls_made += 1
                tool_calls.append(
                    {
                        "id": f"call_{calls_made}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
                    }
                )
            pending += [call["id"] for call in tool_calls]
            messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})
        elif role == "tool":
            body = json.loads(re.search(r"<tool_response>\s*(.*?)\s*</tool_response>", value, re.DOTALL).group(1))
            content = body["content"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": pending.pop(0),
                    "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                }
            )
        else:
            messages.append({"role": role, "content": value})
    return messages


def tool_cuts(messages: list[dict]) -> list[tuple[str, list[dict]]]:
    """Every cut before an assistant turn, labelled by what that turn does."""
    cuts = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or index == 0:
            continue
        before = messages[:index]
        if message.get("tool_calls"):
            kind = "call_after_result" if any(m["role"] == "tool" for m in before) else "first_call"
        else:
            kind = "reply"
        cuts.append((kind, before))
    return cuts


def tool_rows(conversations: list[dict], bfcl: Path, rng: random.Random, rejected: list) -> list[dict]:
    bfcl_names, bfcl_grams = bfcl_items(bfcl)
    pools: dict[str, list] = {kind: [] for kind in TOOL_CUTS}
    for conversation in conversations:
        cid = conversation["id"]
        texts = [t["value"] for t in conversation["conversations"]]
        schemas = json.loads(conversation.get("tools") or "null")
        if not isinstance(schemas, list) or not schemas:
            rejected.append({"family": "tool", "id": cid, "reason": "no tool schemas"})
            continue
        if not all(isinstance(t.get("function"), dict) and t["function"].get("name") for t in schemas):
            rejected.append({"family": "tool", "id": cid, "reason": "malformed tool schema"})
            continue
        tools = json.dumps(schemas)
        tool_names = {t["function"]["name"].lower() for t in schemas}
        if any(BRAND.search(t) for t in texts):
            rejected.append({"family": "tool", "id": cid, "reason": "brand text"})
            continue
        if tool_names & bfcl_names:
            rejected.append(
                {"family": "tool", "id": cid, "reason": f"BFCL function name {sorted(tool_names & bfcl_names)}"}
            )
            continue
        users = [t["value"] for t in conversation["conversations"] if t["from"] == "human"]
        if any(ngrams(u) & bfcl_grams for u in users):
            rejected.append({"family": "tool", "id": cid, "reason": "shares a word 8-gram with a BFCL question"})
            continue
        try:
            cuts = tool_cuts(tool_messages(conversation))
        except (json.JSONDecodeError, AttributeError, KeyError, IndexError):
            rejected.append({"family": "tool", "id": cid, "reason": "unparseable tool call or response"})
            continue
        if cuts:
            kind, prompt = rng.choice(cuts)
            pools[kind].append((prompt, f"hermes-glaive-{kind}", cid, tools))
    chosen = []
    for kind, need in TOOL_CUTS.items():
        if len(pools[kind]) < need:
            raise ValueError(f"tool {kind}: {len(pools[kind])} conversations, need {need}")
        chosen += rng.sample(pools[kind], need)
    tools_by_id = {cid: tools for _, _, cid, tools in chosen}
    rows = split_thinking([(p, source, cid) for p, source, cid, _ in chosen], "tool", rng)
    return [{**r, "tools": tools_by_id[r["id"]]} for r in rows]


def write_rows(rows: list[dict], path: Path) -> None:
    """Prompts as JSON text in ``prompt_json`` and as the loader's message list in ``prompt``.

    Messages differ in their fields, so Arrow gives every message every field; absent ones are dropped again when the
    loader's rows are validated. Template settings are JSON strings, which the task session decodes.
    """
    table = pa.Table.from_pylist([{**r, "chat_template_kwargs": json.dumps(r["chat_template_kwargs"])} for r in rows])
    pq.write_table(table, path)


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(SEED)
    screens = excluded_texts([Path(p) for p in args.exclude_screens])
    earlier = screens | excluded_texts([Path(p) for p in args.exclude_identity])
    pool = read_jsonl(Path(args.identity_pool))
    pool_texts = {normalize(t) for r in pool for t in user_texts(r["prompt"])}
    rejected: list[dict] = []
    rows = (
        identity_rows(pool, earlier, rng, rejected)
        + otherlab_rows(earlier | pool_texts, rng)
        + general_rows(read_jsonl(Path(args.general)), screens, rng, rejected)
        + tool_rows(json.loads(Path(args.tool).read_text()), Path(args.bfcl), rng, rejected)
    )
    for r in rows:
        chat_input(r["prompt"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(rows, out / "mixed.parquet")
    probe = [
        r
        for route in sorted({r["teacher_route"] for r in rows})
        for r in [x for x in rows if x["teacher_route"] == route][:PROBE_ROWS_PER_ROUTE]
    ]
    write_rows(probe, out / "probe.parquet")
    report = {
        "seed": SEED,
        "inputs": {
            Path(p).name: sha256(Path(p))
            for p in [args.identity_pool, args.general, args.tool, *args.exclude_screens, *args.exclude_identity]
        },
        "bfcl_items": len(list(Path(args.bfcl).glob("*.md"))),
        "counts": dict(sorted(Counter(r["teacher_route"] for r in rows).items())),
        "data_sources": dict(sorted(Counter(r["data_source"] for r in rows).items())),
        "teachers": TEACHER,
        "rejected": rejected,
        "tool_source": args.tool_source,
        "mixed_parquet_sha256": sha256(out / "mixed.parquet"),
        "probe_parquet_sha256": sha256(out / "probe.parquet"),
        "probe_ids": [r["id"] for r in probe],
    }
    (out / "mix-report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--identity-pool", required=True, help="Marin-Snowball-OPD-Identity train-used/data.jsonl")
    parser.add_argument("--general", required=True, help="replay prompts jsonl (id, type, messages)")
    parser.add_argument("--tool", required=True, help="hermes-function-calling-v1 glaive-function-calling-5k.json")
    parser.add_argument("--tool-source", required=True, help="tool file's dataset, revision and licence, for the report")
    parser.add_argument("--bfcl", required=True, help="directory of BFCL instruction.md files")
    parser.add_argument("--exclude-screens", nargs="+", required=True, help="screen prompt files; no family may match")
    parser.add_argument(
        "--exclude-identity",
        nargs="+",
        required=True,
        help="earlier identity training and gate files; identity and other-lab only",
    )
    parser.add_argument("--out", required=True)
    report = build(parser.parse_args())
    print(json.dumps({k: report[k] for k in ("counts", "mixed_parquet_sha256")}, indent=1))


if __name__ == "__main__":
    main()
