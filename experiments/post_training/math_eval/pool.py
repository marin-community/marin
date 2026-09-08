# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic enrichment, split locks, and cross-source deduplication."""

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.async_rl_quality import normalize_numeric_answer
from experiments.post_training.math_eval.contract import (
    ANSWER_LINE_INSTRUCTION,
    CONTRACT_IDS,
    GSM8K_INSTRUCTION,
    QWEN,
    SNOWBALL,
    prompt_messages,
    prompt_metadata,
    render_prompt,
)

SPLIT_PRIORITY = {"train": 0, "dev": 1, "heldout": 2, "ood": 3}
MODEL_TEMPLATES = {"qwen": QWEN, "snowball": SNOWBALL}
SPLIT_SALT = "math-eval-v1"
NEAR_DUPLICATE_THRESHOLD = 0.9


@dataclass(frozen=True)
class SourceRows:
    source: str
    revision: str
    license: str
    records: Sequence[dict[str, Any]]
    # Benchmark test sources must explicitly request heldout/ood.
    split: str
    generator_seed: int | None = None
    generator_config: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PoolBuild:
    manifest: list[dict[str, Any]]
    records: dict[str, dict[str, list[dict[str, Any]]]]
    selection: dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalized_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def prompt_hash(question: str) -> str:
    """Hash question identity independently of model, contract suffix, and row order."""
    return hashlib.sha256(normalized_question(question).encode()).hexdigest()


def assign_split(digest: str) -> str:
    """Hash training-source questions into fixed 80/10/10 train/dev/heldout buckets."""
    bucket = int(hashlib.sha256(f"{SPLIT_SALT}:{digest}".encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "heldout"


def answer_type(gold: str, env_class: str) -> str:
    if env_class == "reasoning_gym":
        return "rg_json"
    number = normalize_numeric_answer(gold)
    if number is not None:
        return "fraction" if "/" in number else "integer"
    if re.search(r"[\[\]()]|\\(?:infty|cup|{)", gold):
        return "interval_or_set"
    return "latex_expr" if "\\" in gold or re.search(r"[=^+*/]", gold) else "string"


def _question(record: dict[str, Any]) -> str:
    instruction = GSM8K_INSTRUCTION if record["env_class"] == "gsm8k" else ANSWER_LINE_INSTRUCTION
    question = record["prompt"][-1]["content"]
    if not question.endswith(instruction):
        raise ValueError("Source row does not carry the frozen answer instruction")
    return question[: -len(instruction)]


def _shingles(question: str) -> frozenset[tuple[str, ...]]:
    tokens = re.findall(r"\w+|[^\w\s]", normalized_question(question))
    width = min(3, len(tokens))
    return frozenset(tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1))


def _gold_key(gold: str, env_class: str) -> str:
    if env_class == "reasoning_gym":
        return str(json.loads(gold)["entry"]["answer"])
    return normalize_numeric_answer(gold) or gold.strip()


def build_pool(
    sources: Sequence[SourceRows],
    tokenizers: Mapping[str, Callable[[str], list[int]]],
    *,
    version: str,
    code_sha: str,
    tokenizer_hashes: Mapping[str, str],
    max_prompt_tokens: int = 1024,
    length_eligibility: Sequence[Mapping[str, Any]] = (),
) -> PoolBuild:
    """Build both model views, keeping benchmark membership outside training.

    Tokenizers encode rendered text without adding special tokens. Evaluation rows
    exceeding either model's prompt cap fail instead of silently shrinking a lock.
    """
    if set(tokenizers) != set(MODEL_TEMPLATES) or set(tokenizer_hashes) != set(MODEL_TEMPLATES):
        raise ValueError("Both pinned model tokenizers are required")
    exclusions = {item["prompt_sha256"]: dict(item) for item in length_eligibility}
    if len(exclusions) != len(length_eligibility):
        raise ValueError("Duplicate pre-freeze eligibility receipt")
    applied_exclusions = []
    candidates = []
    for source in sources:
        if source.split not in {*SPLIT_PRIORITY, "hash"} or not source.revision or not source.license:
            raise ValueError("Every source requires a revision, license, and explicit split policy")
        for record in source.records:
            question = _question(record)
            digest = prompt_hash(question)
            split = assign_split(digest) if source.split == "hash" else source.split
            if source.split == "hash" and record["extra_info"]["split"] != "train":
                raise ValueError("Benchmark/test source cannot be hash-assigned into training")
            gold = record["reward_spec"]["ground_truth"]
            if gold != record["reward_model"]["ground_truth"]:
                raise ValueError("Ground-truth channels disagree")
            if digest in exclusions:
                expected = exclusions[digest]
                counts = {
                    model: len(tokenizers[model](render_prompt(question, record["env_class"], template)))
                    for model, template in MODEL_TEMPLATES.items()
                }
                if (
                    source.source != expected["source"]
                    or source.revision != expected["revision"]
                    or counts != expected["tokens"]
                    or max(counts.values()) <= max_prompt_tokens
                ):
                    raise ValueError(f"Pre-freeze eligibility receipt changed for {digest}")
                applied_exclusions.append({**expected, "prospective_split": split, "reason": "pre_freeze_length"})
                continue
            candidates.append((split, source, record, question, digest, gold))
    if {item["prompt_sha256"] for item in applied_exclusions} != set(exclusions):
        raise ValueError("Pre-freeze eligibility receipt references absent rows")
    candidates.sort(key=lambda row: (-SPLIT_PRIORITY[row[0]], row[4], row[1].source, row[1].revision))
    source_signatures = {row[4]: _shingles(row[3]) for row in candidates}
    frequencies = Counter(shingle for signature in source_signatures.values() for shingle in signature)
    manifest, dropped = [], []
    records = {model: {split: [] for split in SPLIT_PRIORITY} for model in MODEL_TEMPLATES}
    seen: dict[str, tuple[str, str]] = {}
    inverted: dict[tuple[str, ...], set[str]] = defaultdict(set)
    signatures: dict[str, frozenset[tuple[str, ...]]] = {}
    for split, source, record, question, digest, gold in candidates:
        env_class = record["env_class"]
        key = _gold_key(gold, env_class)
        if digest in seen:
            if key != seen[digest][0]:
                raise ValueError(f"Conflicting golds for question {digest}")
            dropped.append({"prompt_sha256": digest, "source": source.source, "reason": "exact_duplicate"})
            continue
        signature = source_signatures[digest]
        neighbors = set().union(*(inverted[shingle] for shingle in signature))
        near = next(
            (
                other
                for other in sorted(neighbors)
                if min(len(signature), len(signatures[other])) / max(len(signature), len(signatures[other]))
                >= NEAR_DUPLICATE_THRESHOLD
                and len(signature & signatures[other]) / len(signature | signatures[other]) >= NEAR_DUPLICATE_THRESHOLD
            ),
            None,
        )
        if near is not None:
            dropped.append({"prompt_sha256": digest, "source": source.source, "reason": "near_duplicate", "kept": near})
            continue
        counts = {
            model: len(tokenizers[model](render_prompt(question, env_class, template)))
            for model, template in MODEL_TEMPLATES.items()
        }
        if max(counts.values()) > max_prompt_tokens:
            if split != "train":
                raise ValueError(f"Evaluation question {digest} exceeds the model prompt cap")
            dropped.append({"prompt_sha256": digest, "source": source.source, "reason": "prompt_length"})
            continue
        item = {
            "id": f"{source.source}:{record['extra_info']['split']}:{record['extra_info']['index']}",
            "problem": question,
            "prompt_sha256": digest,
            "source": source.source,
            "revision": source.revision,
            "license": source.license,
            "source_index": record["extra_info"]["index"],
            "source_split": record["extra_info"]["split"],
            "gen_seed": source.generator_seed,
            "gen_config": canonical_json(source.generator_config),
            "bin": record["data_source"],
            "grade": record["extra_info"]["grade"],
            "split": split,
            "gold": gold,
            "gold_original": gold,
            "answer_type": answer_type(gold, env_class),
            "env_class": env_class,
            "contract": CONTRACT_IDS[env_class],
            "audit_status": "pending",
            **{f"tokens_{model}": count for model, count in counts.items()},
            **{f"prompt_template_id_{model}": template.template_id for model, template in MODEL_TEMPLATES.items()},
        }
        manifest.append(item)
        for model, template in MODEL_TEMPLATES.items():
            extra = {
                **record["extra_info"],
                **prompt_metadata(template, env_class),
                "prompt_sha256": digest,
                "split": split,
                "source": source.source,
                "license": source.license,
                "answer_type": item["answer_type"],
            }
            records[model][split].append({**record, "prompt": prompt_messages(question, env_class), "extra_info": extra})
        seen[digest] = (key, split)
        signatures[digest] = signature
        # A >=t Jaccard match must intersect any |B|-ceil(t|B|)+1 elements of B.
        # Index rare shingles from that subset to avoid quadratic candidate sets
        # from common prose, while retaining every possible above-threshold match.
        prefix_size = len(signature) - math.ceil(NEAR_DUPLICATE_THRESHOLD * len(signature)) + 1
        for shingle in sorted(signature, key=lambda item: (frequencies[item], item))[:prefix_size]:
            inverted[shingle].add(digest)
    manifest.sort(key=lambda row: row["prompt_sha256"])
    for model in records.values():
        for rows in model.values():
            rows.sort(key=lambda row: row["extra_info"]["prompt_sha256"])
    locks = {split: [row["prompt_sha256"] for row in manifest if row["split"] == split] for split in SPLIT_PRIORITY}
    selection = {
        "pool_version": version,
        "code_sha": code_sha,
        "split_salt": SPLIT_SALT,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "max_prompt_tokens": max_prompt_tokens,
        "tokenizer_hashes": dict(tokenizer_hashes),
        "prompt_template_ids": {model: template.template_id for model, template in MODEL_TEMPLATES.items()},
        "sources": sorted(
            [{"source": s.source, "revision": s.revision, "license": s.license, "split": s.split} for s in sources],
            key=canonical_json,
        ),
        "rows": locks,
        "heldout_lock_sha256": hashlib.sha256(canonical_json(locks).encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
        "pre_freeze_length_eligibility": sorted(applied_exclusions, key=canonical_json),
        "dropped": dropped,
    }
    return PoolBuild(manifest, records, selection)


def write_pool(build: PoolBuild, output: Path) -> None:
    """Write parquet model views and the shared immutable membership manifest."""
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(build.manifest), output / "manifest.parquet")
    for model, splits in build.records.items():
        directory = output / model
        directory.mkdir()
        for split, rows in splits.items():
            pq.write_table(pa.Table.from_pylist(rows), directory / f"{split}.parquet")
    (output / "selection.json").write_text(canonical_json(build.selection) + "\n")


def prompt_length_report(
    sources: Sequence[SourceRows], tokenizers: Mapping[str, Callable[[str], list[int]]], *, cap: int
) -> dict[str, Any]:
    """Audit every source row before membership is frozen; report no question text."""
    grouped = defaultdict(list)
    overflows = []
    for source in sources:
        for record in source.records:
            question = _question(record)
            digest = prompt_hash(question)
            split = assign_split(digest) if source.split == "hash" else source.split
            counts = {
                model: len(tokenizers[model](render_prompt(question, record["env_class"], template)))
                for model, template in MODEL_TEMPLATES.items()
            }
            for model, count in counts.items():
                grouped[f"{source.source}/{model}/{split}/{record['data_source']}"].append(count)
            if max(counts.values()) > cap:
                overflows.append(
                    {
                        "prompt_sha256": digest,
                        "source": source.source,
                        "split": split,
                        "bin": record["data_source"],
                        "tokens": counts,
                    }
                )
    summary = {
        name: {
            "rows": len(values),
            "min": min(values),
            "max": max(values),
            "p95_nearest_rank": sorted(values)[math.ceil(len(values) * 0.95) - 1],
            "over_cap": sum(value > cap for value in values),
        }
        for name, values in grouped.items()
    }
    return {"cap": cap, "sources": summary, "overflows": overflows}
