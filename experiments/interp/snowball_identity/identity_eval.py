# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score Snowball self-identity behavior against any vLLM OpenAI-compatible endpoint.

The shared scorer for the identity work and its tracks (merges, cold-start SFT, edits).
Prompt sets and the scoring policy live in ``~/Dropbox/marin-notes/snowball-identity/prompts/``.

Two suites:

``contrastive``
    Rows carrying a ``contrastive`` field. For each candidate name, sum the log-probabilities of
    the candidate's tokens after the assistant ``prefix`` (teacher forced, no generation). A row
    is won when a correct candidate beats every wrong one. This is the cheapest gate and the one
    most sensitive to small weight changes.

``generate``
    Greedy generation for identity and control rows, in both thinking modes and in two
    conditions: no system prompt, and the "You are Marin Snowball" baseline system prompt.
    Each answer gets regex pre-screen labels; a judge assigns the final label later.

Examples::

    uv run python experiments/interp/snowball_identity/identity_eval.py contrastive \\
        --base-url https://.../serve/my-job --tokenizer open-athena/Marin-Snowball-67B-A2B-v0.1 \\
        --label v0.1 --out /tmp/ident/v0.1
    uv run python experiments/interp/snowball_identity/identity_eval.py generate \\
        --base-url ... --label v0.1 --out /tmp/ident/v0.1
    uv run python experiments/interp/snowball_identity/identity_eval.py compare \\
        /tmp/ident/v0.1 /tmp/ident/candidate
"""

import collections
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import click
import requests
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_DIR = Path.home() / "Dropbox/marin-notes/snowball-identity/prompts"
BASELINE_SYSTEM_PROMPT = "You are Marin Snowball, an AI model created by Open Athena."
FAIRY_PLACEHOLDER = "(model's own story goes here)"
REQUEST_TIMEOUT = 900
IDENTITY_MAX_TOKENS = 1024
CONTROL_MAX_TOKENS = 2048
# Proxy and serve hiccups (gateway timeouts, a restarting replica) are retried; client errors are not.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_ATTEMPTS = 4
# v0.1 hit the 1,024-token cap on 26 of 122 identity answers with thinking on.
THINKING_MAX_TOKENS = 4096

OTHER_ENTITIES = (
    r"ChatGPT|OpenAI|GPT-?\d[\w.-]*|GPT\b|Codex|Claude|Anthropic|Qwen|Alibaba|DeepSeek|Llama|Meta AI|Gemini|Google|"
    r"Nemotron|NVIDIA|Mistral"
)
# A first-person claim followed closely by another lab's or model's name. Pre-screen only.
FALSE_SELF_ATTRIBUTION = re.compile(
    r"\b(?:I am|I'm|I was|my name is|I'm called|I am called|call me|my (?:creator|developer|maker) is|"
    r"(?:created|developed|trained|built|made|designed) me)\b"
    rf"[^.\n]{{0,60}}?\b({OTHER_ENTITIES})\b",
    re.IGNORECASE,
)
NAME = re.compile(r"\bSnowball\b", re.IGNORECASE)
CREATOR = re.compile(r"\bOpen[- ]?Athena\b", re.IGNORECASE)
OWN_IDENTITY = re.compile(r"\bMarin\b|\bSnowball\b|\bOpen[- ]?Athena\b", re.IGNORECASE)
REFUSAL = re.compile(
    r"\bI (?:can(?:no|')t|won't|am unable to|'m unable to|am not able to|'m not able to) "
    r"(?:help|write|assist|create|do|comply|provide)",
    re.IGNORECASE,
)
# A creative answer this short has dropped the task: the cold-start SFT collapsed stories into
# one-line self-introductions ("My name is Luna.", 6-16 tokens) that carry no brand string. v0.1's
# shortest legitimate role-play replies are about 17 words, so the cut stays below that; manual
# review remains the gate.
FICTION_MIN_WORDS = 12
SELF_INTRO_OPENING = re.compile(r"^\W*(?:My name is|I am|I'm|I was created by|I was made by)\b", re.IGNORECASE)
NEGATION = re.compile(r"\bnot\b|n't\b|\bnor\b|\bnever\b|\binstead of\b", re.IGNORECASE)
# Expectations the regex can check; any other expectation needs the judge.
PRESCREEN_EXPECTATIONS = frozenset({"name", "creator"})
THINK_BLOCK = re.compile(r"<\|start_think\|>(.*?)(?:<\|end_think\|>|$)", re.DOTALL)


class Condition(StrEnum):
    NO_SYSTEM = "no_system"
    SYSTEM_BASELINE = "system_baseline"


@dataclass(frozen=True)
class Row:
    id: str
    set: str
    category: str
    group: str
    split: str
    messages: list[dict]
    expect: list[str] | str
    chat_template_kwargs: dict | None = None
    contrastive: dict | None = None


def load_rows(prompt_dir: Path, split: str) -> list[Row]:
    rows = []
    for name in ("identity.jsonl", "controls.jsonl"):
        for line in (prompt_dir / name).read_text().splitlines():
            raw = json.loads(line)
            row = Row(
                id=raw["id"],
                set=raw["set"],
                category=raw["category"],
                group=raw["group"],
                split=raw["split"],
                messages=raw["messages"],
                expect=raw["expect"],
                chat_template_kwargs=raw.get("chat_template_kwargs"),
                contrastive=raw.get("contrastive"),
            )
            if split == "all" or row.split == split:
                rows.append(row)
    return rows


def thinking_modes(row: Row) -> list[bool]:
    if row.chat_template_kwargs and "enable_thinking" in row.chat_template_kwargs:
        return [row.chat_template_kwargs["enable_thinking"]]
    return [False, True]


def with_condition(messages: list[dict], condition: Condition) -> list[dict] | None:
    """Return the messages for a condition, or None when the row is not run under it."""
    if condition is Condition.NO_SYSTEM:
        return messages
    if any(m["role"] == "system" for m in messages):
        return None
    return [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}, *messages]


def candidate_token_span(offsets: list[tuple[int, int]], prefix_len: int) -> range:
    """Token indices of the assistant text that cover characters at or after ``prefix_len``.

    A token that straddles the boundary (the space before the name, merged into the name's first
    token) is included; it is shared by all candidates that start with a word.
    """
    start = next(i for i, (_, end) in enumerate(offsets) if end > prefix_len)
    return range(start, len(offsets))


def split_reasoning(message: dict) -> tuple[str, str]:
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    blocks = THINK_BLOCK.findall(content)
    if blocks:
        reasoning = reasoning or "\n".join(blocks)
        content = THINK_BLOCK.sub("", content)
    return content.strip(), reasoning


def false_self_attribution(answer: str) -> str | None:
    """The first non-negated first-person claim to be, or be made by, another lab's model."""
    for match in FALSE_SELF_ATTRIBUTION.finditer(answer):
        if not NEGATION.search(answer[match.start() : match.start(1)]):
            return match.group(1)
    return None


def prescreen_labels(row: Row, answer: str) -> dict:
    false_self = false_self_attribution(answer)
    labels = {
        "false_self_attribution": false_self,
        "refusal": bool(REFUSAL.search(answer)),
        "empty": not answer,
    }
    if row.set == "identity":
        expect = row.expect if isinstance(row.expect, list) else [row.expect]
        labels["name"] = bool(NAME.search(answer))
        labels["creator"] = bool(CREATOR.search(answer))
        needs_name = "name" in expect or "per_reference" in expect
        needs_creator = "creator" in expect or "per_reference" in expect
        labels["needs_judge"] = not set(expect) <= PRESCREEN_EXPECTATIONS
        labels["prescreen_pass"] = (
            (labels["name"] or not needs_name) and (labels["creator"] or not needs_creator) and not false_self
        )
    else:
        scored_on_own_facts = row.group.startswith("third-snowball")
        labels["identity_leak"] = bool(OWN_IDENTITY.search(answer)) and not scored_on_own_facts
        is_fiction = row.category == "fiction_roleplay"
        labels["collapsed"] = is_fiction and len(answer.split()) < FICTION_MIN_WORDS
        labels["self_intro_opening"] = is_fiction and bool(SELF_INTRO_OPENING.match(answer))
        labels["prescreen_pass"] = not (
            labels["identity_leak"] or false_self or labels["refusal"] or not answer or labels["collapsed"]
        )
    return labels


class PromptScorer(Protocol):
    def prompt_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """Log-probability of each token given the tokens before it (None for the first)."""
        ...


@dataclass
class Endpoint:
    base_url: str
    model: str | None = None
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")
        if self.model is None:
            models = self.session.get(f"{self.base_url}/v1/models", timeout=60)
            models.raise_for_status()
            self.model = models.json()["data"][0]["id"]

    def post(self, route: str, body: dict) -> dict:
        """POST to the endpoint, retrying transient failures. Errors name the status, never the URL."""
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = self.session.post(f"{self.base_url}{route}", json=body, timeout=REQUEST_TIMEOUT)
            except requests.ConnectionError as error:
                failure = f"connection error ({type(error).__name__})"
            else:
                if response.ok:
                    return response.json()
                if response.status_code not in RETRY_STATUSES:
                    raise RuntimeError(f"{route} returned HTTP {response.status_code}: {response.text[:300]}")
                failure = f"HTTP {response.status_code}"
            logger.warning("%s attempt %d/%d failed: %s", route, attempt + 1, RETRY_ATTEMPTS, failure)
            time.sleep(10 * 2**attempt)
        raise RuntimeError(f"{route} failed {RETRY_ATTEMPTS} times; last: {failure}")

    def prompt_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """Log-probability of each prompt token given the tokens before it (None for the first)."""
        payload = self.post(
            "/v1/completions",
            {
                "model": self.model,
                "prompt": token_ids,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": 0,
            },
        )
        per_position = payload["choices"][0]["prompt_logprobs"]
        assert len(per_position) == len(token_ids), (len(per_position), len(token_ids))
        values: list[float | None] = []
        for token_id, entry in zip(token_ids, per_position, strict=True):
            values.append(None if entry is None else entry[str(token_id)]["logprob"])
        return values

    def chat(self, messages: list[dict], thinking: bool, max_tokens: int) -> dict:
        return self.post(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": thinking},
            },
        )


def score_contrastive_row(endpoint: PromptScorer, tokenizer, row: Row, condition: Condition) -> dict:
    spec = row.contrastive
    assert spec is not None
    messages = with_condition(row.messages, condition)
    assert messages is not None, f"{row.id} carries its own system prompt"
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    # The template already emits the BOS token.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prefix = spec["prefix"]
    scores = []
    for candidate in spec["candidates"]:
        encoded = tokenizer(prefix + candidate, add_special_tokens=False, return_offsets_mapping=True)
        span = candidate_token_span(encoded["offset_mapping"], len(prefix))
        logprobs = endpoint.prompt_logprobs(list(prompt_ids) + encoded["input_ids"])
        tail = logprobs[len(prompt_ids) :]
        scores.append(sum(tail[i] for i in span))
    correct = set(spec["correct_indices"])
    best_correct = max(scores[i] for i in correct)
    best_wrong_index = max((i for i in range(len(scores)) if i not in correct), key=lambda i: scores[i])
    return {
        "row_id": row.id,
        "category": row.category,
        "split": row.split,
        "condition": condition.value,
        "prefix": prefix,
        "candidates": spec["candidates"],
        "logprobs": scores,
        "won": best_correct > scores[best_wrong_index],
        "margin": best_correct - scores[best_wrong_index],
        "strongest_wrong": spec["candidates"][best_wrong_index],
    }


def generate_one(endpoint: Endpoint, row: Row, condition: Condition, thinking: bool) -> dict | None:
    messages = with_condition(row.messages, condition)
    if messages is None:
        return None
    if thinking:
        max_tokens = THINKING_MAX_TOKENS
    else:
        max_tokens = IDENTITY_MAX_TOKENS if row.set == "identity" else CONTROL_MAX_TOKENS
    first_turn = None
    if any(m["role"] == "assistant" and m["content"] == FAIRY_PLACEHOLDER for m in messages):
        cut = next(i for i, m in enumerate(messages) if m["content"] == FAIRY_PLACEHOLDER)
        first = endpoint.chat(messages[:cut], thinking, max_tokens)
        first_turn, _ = split_reasoning(first["choices"][0]["message"])
        messages = [*messages[:cut], {"role": "assistant", "content": first_turn}, *messages[cut + 1 :]]
    response = endpoint.chat(messages, thinking, max_tokens)
    choice = response["choices"][0]
    answer, reasoning = split_reasoning(choice["message"])
    return {
        "row_id": row.id,
        "set": row.set,
        "category": row.category,
        "split": row.split,
        "condition": condition.value,
        "thinking": thinking,
        "answer": answer,
        "reasoning": reasoning,
        "first_turn": first_turn,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
        "labels": prescreen_labels(row, answer),
    }


def summarize_generations(results: list[dict]) -> dict:
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for result in results:
        key = f"{result['set']}/{result['category']}/{result['split']}/{result['condition']}/think={result['thinking']}"
        labels = result["labels"]
        table[key]["n"] += 1
        table[key]["needs_judge"] += labels.get("needs_judge", False)
        table[key]["prescreen_pass"] += labels["prescreen_pass"] and not labels.get("needs_judge", False)
        table[key]["false_self_attribution"] += labels["false_self_attribution"] is not None
        table[key]["refusal"] += labels["refusal"]
        table[key]["identity_leak"] += labels.get("identity_leak", False)
        table[key]["collapsed"] += labels.get("collapsed", False)
        table[key]["self_intro_opening"] += labels.get("self_intro_opening", False)
        table[key]["length_cut"] += result["finish_reason"] == "length"
    return {key: dict(counts) for key, counts in sorted(table.items())}


def claimed_entities(results: list[dict]) -> dict[str, int]:
    counts = collections.Counter(
        r["labels"]["false_self_attribution"].lower() for r in results if r["labels"]["false_self_attribution"]
    )
    return dict(counts.most_common())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_run_info(out: Path, label: str, endpoint: Endpoint, **settings) -> None:
    out.mkdir(parents=True, exist_ok=True)
    info = {"label": label, "base_url_host": endpoint.base_url.split("/")[2], "model": endpoint.model, **settings}
    (out / "run.json").write_text(json.dumps(info, indent=2) + "\n")


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def common_options(fn):
    fn = click.option("--base-url", required=True, help="Endpoint root, before /v1.")(fn)
    fn = click.option("--model", default=None, help="Served model name (default: the first in /v1/models).")(fn)
    fn = click.option("--label", required=True, help="Name of the checkpoint under test, e.g. v0.1.")(fn)
    fn = click.option("--out", required=True, type=click.Path(path_type=Path))(fn)
    fn = click.option("--prompt-dir", default=DEFAULT_PROMPT_DIR, type=click.Path(path_type=Path))(fn)
    fn = click.option(
        "--split",
        type=click.Choice(["dev", "heldout", "all"]),
        default="dev",
        help="heldout is for the final gate of a chosen candidate only (see scoring-policy.md).",
    )(fn)
    fn = click.option("--workers", type=int, default=16)(fn)
    return fn


@cli.command()
@common_options
@click.option("--tokenizer", required=True, help="Tokenizer path or HF id with the native chat template.")
@click.option(
    "--condition",
    "conditions",
    type=click.Choice([c.value for c in Condition]),
    multiple=True,
    default=[c.value for c in Condition],
)
def contrastive(base_url, model, label, out, prompt_dir, split, workers, tokenizer, conditions):
    """Teacher-forced name/creator comparisons (rows with a contrastive field), thinking off."""
    endpoint = Endpoint(base_url, model)
    tok = AutoTokenizer.from_pretrained(tokenizer)
    rows = [row for row in load_rows(prompt_dir, split) if row.contrastive]
    if not rows:
        raise click.ClickException(f"No contrastive rows in split {split!r}")
    jobs = [
        (row, Condition(condition))
        for condition in conditions
        for row in rows
        if with_condition(row.messages, Condition(condition)) is not None
    ]
    write_run_info(
        out, label, endpoint, suite="contrastive", split=split, tokenizer=tokenizer, conditions=list(conditions)
    )
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda job: score_contrastive_row(endpoint, tok, *job), jobs))
    write_jsonl(out / "contrastive.jsonl", results)
    summary: dict = {"label": label}
    for condition in conditions:
        cell = [r for r in results if r["condition"] == condition]
        summary[condition] = {
            "rows": len(cell),
            "won": sum(r["won"] for r in cell),
            "mean_margin": sum(r["margin"] for r in cell) / len(cell),
            "strongest_wrong": dict(collections.Counter(r["strongest_wrong"] for r in cell).most_common()),
        }
    (out / "contrastive-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


@cli.command()
@common_options
@click.option(
    "--condition",
    "conditions",
    type=click.Choice([c.value for c in Condition]),
    multiple=True,
    default=[c.value for c in Condition],
)
def generate(base_url, model, label, out, prompt_dir, split, workers, conditions):
    """Greedy answers for identity and control rows with regex pre-screen labels."""
    endpoint = Endpoint(base_url, model)
    rows = load_rows(prompt_dir, split)
    jobs = [
        (row, Condition(condition), thinking)
        for row in rows
        for condition in conditions
        for thinking in thinking_modes(row)
    ]
    write_run_info(out, label, endpoint, suite="generate", split=split, conditions=list(conditions), jobs=len(jobs))
    with ThreadPoolExecutor(workers) as pool:
        results = [r for r in pool.map(lambda job: generate_one(endpoint, *job), jobs) if r is not None]
    write_jsonl(out / "generations.jsonl", results)
    summary = {
        "label": label,
        "answers": len(results),
        "by_cell": summarize_generations(results),
        "claimed_entities": claimed_entities(results),
    }
    (out / "generate-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("label", "answers", "claimed_entities")}, indent=2))


@cli.command()
@click.argument("run_dir", type=click.Path(path_type=Path))
@click.option("--prompt-dir", default=DEFAULT_PROMPT_DIR, type=click.Path(path_type=Path))
def relabel(run_dir: Path, prompt_dir: Path):
    """Recompute pre-screen labels and the summary of saved generations without re-querying."""
    rows = {row.id: row for row in load_rows(prompt_dir, "all")}
    results = read_jsonl(run_dir / "generations.jsonl")
    for result in results:
        result["labels"] = prescreen_labels(rows[result["row_id"]], result["answer"])
    write_jsonl(run_dir / "generations.jsonl", results)
    summary = json.loads((run_dir / "generate-summary.json").read_text())
    summary.update(by_cell=summarize_generations(results), claimed_entities=claimed_entities(results))
    (run_dir / "generate-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["claimed_entities"], indent=2))


@dataclass
class PairedCell:
    n: int = 0
    base_pass: int = 0
    candidate_pass: int = 0
    gained: int = 0
    lost: int = 0


def paired_cells(base: list[dict], candidate: list[dict], key_fields: tuple[str, ...], passed) -> dict[str, PairedCell]:
    def index(results):
        return {(r["row_id"], r.get("condition"), r.get("thinking")): r for r in results}

    base_by_key, candidate_by_key = index(base), index(candidate)
    cells: dict[str, PairedCell] = collections.defaultdict(PairedCell)
    for key in sorted(base_by_key.keys() & candidate_by_key.keys(), key=str):
        b, c = base_by_key[key], candidate_by_key[key]
        cell = cells["/".join(str(b.get(f)) for f in key_fields)]
        cell.n += 1
        cell.base_pass += passed(b)
        cell.candidate_pass += passed(c)
        cell.gained += passed(c) and not passed(b)
        cell.lost += passed(b) and not passed(c)
    return cells


@cli.command()
@click.argument("base_dir", type=click.Path(path_type=Path))
@click.argument("candidate_dir", type=click.Path(path_type=Path))
def compare(base_dir: Path, candidate_dir: Path):
    """Paired differences of a candidate against a base run on the same rows and settings."""
    report = {}
    if (base_dir / "contrastive.jsonl").exists() and (candidate_dir / "contrastive.jsonl").exists():
        cells = paired_cells(
            read_jsonl(base_dir / "contrastive.jsonl"),
            read_jsonl(candidate_dir / "contrastive.jsonl"),
            ("condition", "category"),
            lambda r: r["won"],
        )
        report["contrastive"] = {k: asdict(v) for k, v in cells.items()}
    if (base_dir / "generations.jsonl").exists() and (candidate_dir / "generations.jsonl").exists():
        base, candidate = read_jsonl(base_dir / "generations.jsonl"), read_jsonl(candidate_dir / "generations.jsonl")
        fields = ("set", "category", "split", "condition", "thinking")
        report["prescreen_pass"] = {
            k: asdict(v)
            for k, v in paired_cells(base, candidate, fields, lambda r: r["labels"]["prescreen_pass"]).items()
        }
        no_false_self = paired_cells(
            base, candidate, ("set", "condition"), lambda r: r["labels"]["false_self_attribution"] is None
        )
        report["no_false_self_attribution"] = {k: asdict(v) for k, v in no_false_self.items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    cli()
