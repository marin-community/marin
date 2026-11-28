# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Planted-copy check of stage 1 and stage 2.

A Zephyr job over the item table plants copies of sampled items, in several forms, into real
record text, and checks stage 1 on each: does the planted record hold the item's key 1, read as
stage 1 reads records, and does it hold key 1 or key 2? Copies with small edits (``edited_forms``)
are checked by stage 1 only. It also writes, for a subset of items, the planted positives and planted
negatives (a changed or missing choice, a changed number or sign, choices far from the question)
as candidate pairs, which ``score.py`` scores with the frozen classifier.
"""

import argparse
import json
import logging
import random
import re
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.decon_keys import WINDOW, readings, text_stream, window_hashes
from marin.datakit.decon_tables import parquet_files, read_frame, read_table, write_table
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

logger = logging.getLogger(__name__)
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4
CONTEXT = 1500
FAR = 3000
MIN_HOST = 2 * CONTEXT + FAR
NUMBER = re.compile(r"\d+")
PUNCTUATION = re.compile(r"[^\w\s]")
SIGNS = (("+", "-"), ("-", "+"), ("<", ">"), (">", "<"))
WORD = re.compile(r"\w+")
# Words that replace a word of the item in an edited copy.
REPLACEMENTS = ("apple", "river", "seven", "blue", "market", "garden", "silver", "window")
DELETED_SHARE = 0.1


def lettered(question: str, choices: list[str]) -> str:
    return question + "".join(f"\n{LABELS[i]}. {choice}" for i, choice in enumerate(choices))


def edited_forms(item: dict[str, Any], rng: random.Random) -> dict[str, str]:
    """Copies with small edits (typos, changed words, deleted words), in the plain layout.

    These are copies under the rubric (an edited copy), so stage 1 should send them to stage 2. Paraphrases and
    changed numbers are not here: the first needs a model, and the second makes a different item.
    """
    text = lettered(item["question"], list(item["choices"]))
    if passage := item.get("passage") or "":
        text = f"{passage}\n\n{text}"
    words = WORD.findall(text)
    # Words of two or more characters, so that a choice label ("B") stays as it is.
    every = [i for i, word in enumerate(words) if len(word) >= 2]
    swappable = [i for i in every if any(a != b for a, b in zip(words[i], words[i][1:], strict=False))]

    def pick(pool: list[int], count: int) -> set[int]:
        return set(rng.sample(pool, min(count, len(pool))))

    def swap(word: str) -> str:
        j = rng.choice([j for j in range(len(word) - 1) if word[j] != word[j + 1]])
        return word[:j] + word[j + 1] + word[j] + word[j + 2 :]

    def edit(chosen: set[int], change: Callable[[str], str]) -> str:
        position = iter(range(len(words)))
        return WORD.sub(lambda m: change(m.group()) if next(position) in chosen else m.group(), text)

    return {
        "typo_1": edit(pick(swappable, 1), swap),
        "typos_3": edit(pick(swappable, 3), swap),
        "word_changed_1": edit(pick(every, 1), lambda _: rng.choice(REPLACEMENTS)),
        "words_changed_3": edit(pick(every, 3), lambda _: rng.choice(REPLACEMENTS)),
        "words_deleted_10pct": edit({i for i in every if rng.random() < DELETED_SHARE}, lambda _: ""),
    }


def positive_forms(item: dict[str, Any], other_choice: str, rng: random.Random) -> dict[str, str]:
    """Copies of the item as a record could hold them; a copy of an item with a passage holds the passage."""
    question, choices, answer = item["question"], list(item["choices"]), item["answer"] or ""
    plain = lettered(question, choices)
    shuffled = choices[:]
    rng.shuffle(shuffled)
    forms = {
        "plain": plain,
        "upper": plain.upper(),
        "no_punctuation": PUNCTUATION.sub(" ", plain),
        "no_line_breaks": plain.replace("\n", " "),
        "numbered": question + "".join(f"\n{i + 1}) {c}" for i, c in enumerate(choices)),
        "parenthesized": question + " " + " ".join(f"({LABELS[i]}) {c}" for i, c in enumerate(choices)),
        "shuffled": lettered(question, shuffled),
        "extra_option": lettered(question, [*choices, other_choice]),
        "markdown": f"**{question}**" + "".join(f"\n- {c}" for c in choices),
        "json": json.dumps({"question": question, "choices": choices, "answer": answer}, ensure_ascii=False),
        "json_ascii": json.dumps({"question": question, "choices": choices, "answer": answer}),
        "escaped_once": json.dumps(plain)[1:-1],
        "escaped_twice": json.dumps(json.dumps(plain))[1:-1],
        "question_only": question,
    }
    if passage := item.get("passage") or "":
        forms = {form: f"{passage}\n\n{text}" for form, text in forms.items()}
    return forms


def negative_forms(item: dict[str, Any], other_choice: str, far_text: str) -> dict[str, str]:
    """Records that share the item's text but do not hold the item."""
    question, choices = item["question"], list(item["choices"])
    forms = {}
    if passage := item.get("passage") or "":
        # An item with a passage is held only with both parts.
        forms["passage_only"] = passage
        forms["question_without_passage"] = lettered(question, choices)
    if choices:
        changed = choices[:]
        changed[len(changed) // 2] = other_choice
        forms["changed_choice"] = lettered(question, changed)
        forms["choices_far"] = question + "\n" + far_text + lettered("", choices)
    if len(choices) >= 3:
        forms["missing_choice"] = lettered(question, choices[:-1])
    if match := NUMBER.search(question):
        changed_number = question[: match.start()] + str(int(match.group()) + 1) + question[match.end() :]
        forms["number_changed"] = lettered(changed_number, choices)
    for sign, other in SIGNS:
        if sign in question:
            index = question.index(sign)
            forms["sign_changed"] = lettered(question[:index] + other + question[index + 1 :], choices)
            break
    return forms


def reading_with_key(record: str, keys: set[int], width: int = WINDOW) -> str | None:
    """The first reading of *record* that holds any of *keys*, as ``pairs.extract`` cuts its excerpt from it."""
    return next((r for r in readings(record) if keys & set(window_hashes(text_stream(r), width).tolist())), None)


def holds_key(record: str, keys: set[int], width: int = WINDOW) -> bool:
    return reading_with_key(record, keys, width) is not None


def make_plant(
    output: str, keys_path: str, hosts_path: str, per_eval: int, scored_per_eval: int, width: int = WINDOW
) -> Callable:
    def plant(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        keys = read_frame(keys_path, ["id", "key1_hash", "key2_hash", "key2_count"])
        key_of: dict[str, set[int]] = {}
        # Key 1 and, where no reference record holds it, key 2 (the guard of the extra key streams).
        either_of: dict[str, set[int]] = {}
        columns = (
            keys["id"],
            keys["key1_hash"].to_numpy(dtype="uint64").tolist(),
            keys["key2_hash"].to_numpy(dtype="uint64").tolist(),
            keys["key2_count"].tolist(),
        )
        for item_id, key1, key2, key2_count in zip(*columns, strict=True):
            if key1:
                key_of.setdefault(item_id, set()).add(key1)
                either_of.setdefault(item_id, set()).add(key1)
            if key1 and key2 and key2_count == 0:
                either_of[item_id].add(key2)
        hosts = [text for text in read_frame(hosts_path, ["text"])["text"] if text and len(text) >= MIN_HOST]
        rng = random.Random(shard.shard_idx)
        for path in paths:
            items = read_table(path, ["id", "eval", "question", "choices", "answer", "passage", "reason"]).to_pylist()
            items = [item for item in items if not item["reason"] and item["question"]]
            sampled = random.Random(0).sample(items, min(per_eval, len(items)))
            counts: Counter = Counter()
            misses, pairs = [], []
            for number, item in enumerate(sampled):
                key = key_of.get(item["id"], set())
                if not key:
                    counts["not_indexable"] += 1
                    continue
                other = rng.choice(sampled)
                other_choice = (list(other["choices"]) or [other["question"][:80]])[0]
                host = rng.choice(hosts)
                before, after = host[:CONTEXT], host[CONTEXT : 2 * CONTEXT]
                either = either_of[item["id"]]
                for form, text in positive_forms(item, other_choice, rng).items():
                    record = f"{before}\n{text}\n{after}"
                    reading = reading_with_key(record, key, width)
                    found = reading is not None
                    counts[f"{form}|planted"] += 1
                    counts[f"{form}|found"] += found
                    counts[f"{form}|found_either"] += found or holds_key(record, either, width)
                    if not found:
                        misses.append({"id": item["id"], "eval": item["eval"], "form": form})
                    if number < scored_per_eval:
                        pairs.append({**_pair(item, reading or record), "form": form, "planted": "positive"})
                # Edited copies: stage 1 only, counted apart from the layout forms.
                for form, text in edited_forms(item, rng).items():
                    record = f"{before}\n{text}\n{after}"
                    counts[f"edited/{form}|planted"] += 1
                    counts[f"edited/{form}|found"] += holds_key(record, key, width)
                    counts[f"edited/{form}|found_either"] += holds_key(record, either, width)
                if number < scored_per_eval:
                    far = host[2 * CONTEXT : 2 * CONTEXT + FAR]
                    for form, text in negative_forms(item, other_choice, far).items():
                        pairs.append({**_pair(item, f"{before}\n{text}\n{after}"), "form": form, "planted": "negative"})
            name = StoragePath(path).name.removesuffix(".parquet")
            stats = {"eval": name, "items": len(sampled), "counts": dict(counts), "misses": misses}
            StoragePath(prefix_join(output, f"stats/{name}.json")).write_text(json.dumps(stats))
            if pairs:
                write_table(pa.Table.from_pylist(pairs), prefix_join(output, f"pairs/{name}.parquet"))
            yield {"eval": name, "items": len(sampled), "misses": len(misses)}

    return plant


def _pair(item: dict[str, Any], record: str) -> dict[str, Any]:
    return {
        "record": "planted",
        "item": item["id"],
        "eval": item["eval"],
        "question": item["question"],
        "choices": list(item["choices"]),
        "answer": item["answer"] or "",
        "right": record,
    }


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True, help="the item table directory (items/*.parquet)")
    parser.add_argument("--keys", required=True, help="the frozen stage-1 keys.parquet")
    parser.add_argument("--hosts", required=True, help="parquet with a text column: real records to plant into")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-eval", type=int, default=50, help="items checked per eval")
    parser.add_argument("--scored-per-eval", type=int, default=10, help="items per eval written as pairs for stage 2")
    parser.add_argument("--width", type=int, default=WINDOW, help="key window width of the stage-1 keys")
    args = parser.parse_args()
    files = parquet_files(args.items)
    resources = ResourceConfig(cpu=1, ram="8g")
    context = ZephyrContext(name="decon-encoder-planted", resources=resources, max_workers=min(100, len(files)))
    plant = make_plant(args.output, args.keys, args.hosts, args.per_eval, args.scored_per_eval, args.width)
    outcome = context.execute(Dataset.from_list(files).map_shard(plant), map_task_resources=resources)
    logger.info("planted: %d evals, %d misses", len(outcome.results), sum(r["misses"] for r in outcome.results))


if __name__ == "__main__":
    main()
