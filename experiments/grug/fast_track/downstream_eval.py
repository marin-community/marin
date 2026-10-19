# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a saved fast-track checkpoint on downstream tasks and on held-out text by token class.

Downstream tasks use the cloze requests from ``downstream_tasks``. For each choice the scorer
records the continuation log-likelihood, its token and byte counts, and whether greedy decoding
recovers it. The task metrics are computed from these records, so new metrics need no new
forward pass.

Held-out text uses the Paloma and Uncheatable token caches that training evaluates. The scorer
splits the loss by token class (newline whitespace, other whitespace, content) and also scores a
whitespace-collapsed copy of each prose document, so a format effect shows as a loss change that
the collapsed text does not repeat.
"""

import dataclasses
import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import click
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from haliax.partitioning import set_mesh
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.store.cache import TreeCache
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.fast_track.downstream_tasks import DOWNSTREAM_TASKS_VERSION, TASK_BUILDERS, staged_task_examples
from experiments.grug.fast_track.launch import (
    V16384_TOKENIZER,
    V16384_VOCAB,
    _h100_ladder_model,
    _h100_ladder_rung,
)
from experiments.grug.fast_track.model import Transformer

logger = logging.getLogger(__name__)

SCORER_VERSION = "downstream-eval-v4-code-math"
MAX_REQUEST_TOKENS = 4096
TOKENS_PER_BATCH = 32768
REQUEST_BUCKETS = (256, 512, 1024, 2048, 4096)
COMPUTE_POLICY = "params=float32,compute=bfloat16,output=float32"

EVAL_CACHE_ROOT = "s3://marin-us-east-02a/marin"
PALOMA_CACHE_VERSION = "2026.09.17"
UNCHEATABLE_CACHE_VERSION = "2026.06.28"
PALOMA_SETS = (
    "4chan",
    "c4_100_domains",
    "c4_en",
    "dolma-v1_5",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
)
UNCHEATABLE_SETS = (
    "ao3_english",
    "arxiv_computer_science",
    "arxiv_physics",
    "bbc_news",
    "github_cpp",
    "github_python",
    "wikipedia_english",
)
# Whitespace in code carries syntax, so the collapsed-format copy skips these sets.
CODE_SETS = frozenset({"dolma_100_programing_languages", "github_cpp", "github_python"})
MAX_TEXT_TOKENS_PER_SET = 1_000_000
# Context-length buckets for the held-out loss: [1, 128), [128, 512), [512, 2048), [2048, 4096).
CONTEXT_BUCKETS = (128, 512, 2048)
CONTEXT_BUCKET_NAMES = ("lt128", "128_512", "512_2048", "ge2048")
# Fixed-target context test: the last TARGET_SPAN tokens of each document that is at least
# MAX_REQUEST_TOKENS long, scored with each available context length. The targets stay the same,
# so a loss change across lengths measures only how much the model uses longer context.
TARGET_SPAN = 128
TARGET_CONTEXTS = (128, 512, 2048, MAX_REQUEST_TOKENS - TARGET_SPAN)
MAX_CONTEXT_TEST_DOCUMENTS = 200
TEXT_READ_BATCH = 256


class TokenClass:
    NEWLINE = "newline_whitespace"
    SPACE = "other_whitespace"
    CONTENT = "content"
    SPECIAL = "special"


@dataclass(frozen=True)
class EncodedRequest:
    tokens: tuple[int, ...]
    continuation_start: int
    continuation_bytes: int
    continuation_chars: int


def encode_request(tokenizer: MarinTokenizer, context: str, continuation: str) -> EncodedRequest:
    """Encode ``[BOS] + context + continuation`` the way training frames a document."""
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    whole_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    if whole_ids[: len(context_ids)] != context_ids:
        whole_ids = context_ids + tokenizer.encode(continuation, add_special_tokens=False)
    tokens = [tokenizer.bos_token_id, *whole_ids]
    start = 1 + len(context_ids)
    overflow = len(tokens) - MAX_REQUEST_TOKENS
    if overflow > 0:
        if overflow >= start - 1:
            raise ValueError(f"continuation does not fit in {MAX_REQUEST_TOKENS} tokens")
        tokens = [tokenizer.bos_token_id, *tokens[1 + overflow :]]
        start -= overflow
    return EncodedRequest(tuple(tokens), start, len(continuation.encode("utf-8")), len(continuation))


def token_classes(tokenizer: MarinTokenizer, vocab_size: int) -> np.ndarray:
    """Return the class of each vocabulary entry as an index into ``CLASS_NAMES``."""
    special = set(tokenizer.all_special_ids)
    classes = np.empty(vocab_size, dtype=np.int8)
    for token_id in range(vocab_size):
        text = tokenizer.decode([token_id])
        if token_id in special:
            classes[token_id] = CLASS_NAMES.index(TokenClass.SPECIAL)
        elif text and not text.strip():
            classes[token_id] = CLASS_NAMES.index(TokenClass.NEWLINE if "\n" in text else TokenClass.SPACE)
        else:
            classes[token_id] = CLASS_NAMES.index(TokenClass.CONTENT)
    return classes


CLASS_NAMES = (TokenClass.NEWLINE, TokenClass.SPACE, TokenClass.CONTENT, TokenClass.SPECIAL)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines, and strip spaces at line ends; keep every word."""
    text = re.sub(r"[ \t\f\v\r]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _bucket(length: int) -> int:
    for bucket in REQUEST_BUCKETS:
        if length <= bucket:
            return bucket
    raise ValueError(f"request of {length} tokens exceeds {REQUEST_BUCKETS[-1]}")


@jax.jit
def _token_scores(model: Transformer, tokens: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the log-probability of each next token and whether it is the argmax."""
    logits = model.logits(tokens, mask=AttentionMask.causal()).astype(jnp.float32)
    targets = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    greedy = jnp.argmax(logits, axis=-1) == targets
    return target_log_probs, greedy


def score_sequences(model: Transformer, sequences: Sequence[Sequence[int]]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield next-token log-probabilities and greedy flags for each sequence, in input order."""
    by_bucket: dict[int, list[int]] = defaultdict(list)
    for index, sequence in enumerate(sequences):
        by_bucket[_bucket(len(sequence))].append(index)
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for bucket, indices in sorted(by_bucket.items()):
        rows = max(1, TOKENS_PER_BATCH // bucket)
        for start in range(0, len(indices), rows):
            chunk = indices[start : start + rows]
            batch = np.zeros((rows, bucket), dtype=np.int32)
            for row, index in enumerate(chunk):
                batch[row, : len(sequences[index])] = sequences[index]
            log_probs, greedy = jax.device_get(_token_scores(model, jnp.asarray(batch)))
            for row, index in enumerate(chunk):
                length = len(sequences[index])
                results[index] = (log_probs[row, : length - 1], greedy[row, : length - 1])
    for index in range(len(sequences)):
        yield results.pop(index)


def score_downstream(model: Transformer, tokenizer: MarinTokenizer, tasks: Sequence[str]) -> pa.Table:
    """Return one row per (example, choice) with its continuation log-likelihood and counts."""
    columns: dict[str, list] = defaultdict(list)
    for task in tasks:
        examples = staged_task_examples(task)
        requests = []
        for example in examples:
            for choice_index, choice in enumerate(example.choices):
                requests.append((example, choice_index, encode_request(tokenizer, choice.context, choice.continuation)))
        scores = score_sequences(model, [request.tokens for _, _, request in requests])
        for (example, choice_index, request), (log_probs, greedy) in zip(requests, scores, strict=True):
            # Position p predicts token p + 1, so the continuation is scored at start - 1 onward.
            span = slice(request.continuation_start - 1, len(request.tokens) - 1)
            columns["task"].append(task)
            columns["example_id"].append(example.example_id)
            columns["choice"].append(choice_index)
            columns["label"].append(example.label)
            columns["log_likelihood"].append(float(log_probs[span].sum()))
            columns["greedy"].append(bool(greedy[span].all()))
            columns["tokens"].append(len(request.tokens) - request.continuation_start)
            columns["bytes"].append(request.continuation_bytes)
            columns["chars"].append(request.continuation_chars)
        logger.info("scored %s: %d examples", task, len(examples))
    return pa.table(columns)


def downstream_metrics(records: pa.Table) -> dict[str, dict[str, float]]:
    """Accuracy, length-normalized accuracy, and continuous scores of the correct choice per task."""
    frame = records.to_pandas()
    metrics: dict[str, dict[str, float]] = {}
    for task, rows in frame.groupby("task", sort=True):
        per_example = []
        for _, choices in rows.groupby("example_id", sort=False):
            choices = choices.sort_values("choice")
            label = int(choices["label"].iloc[0])
            log_likelihood = choices["log_likelihood"].to_numpy()
            per_char = log_likelihood / np.maximum(choices["chars"].to_numpy(), 1)
            correct = choices.iloc[label]
            normalized = np.exp(per_char - per_char.max())
            others = np.delete(per_char, label)
            per_example.append(
                {
                    "acc": float(np.argmax(log_likelihood) == label),
                    "acc_norm": float(np.argmax(per_char) == label),
                    "correct_bpb": -float(correct["log_likelihood"]) / (math.log(2) * max(int(correct["bytes"]), 1)),
                    "correct_prob_norm": float(normalized[label] / normalized.sum()),
                    "margin_per_char": float(per_char[label] - others.max()) if len(others) else 0.0,
                    "greedy": float(correct["greedy"]),
                }
            )
        if rows["choice"].max() == 0:
            # A single-choice task (LAMBADA) has no choice metrics; its accuracy is greedy recovery.
            per_example = [
                {"acc": row["greedy"], "correct_bpb": row["correct_bpb"], "greedy": row["greedy"]} for row in per_example
            ]
        values = {key: float(np.mean([row[key] for row in per_example])) for key in per_example[0]}
        values["examples"] = float(len(per_example))
        metrics[str(task)] = values
    return metrics


def _cache_path(eval_set: str) -> str:
    if eval_set in PALOMA_SETS:
        return prefix_join(EVAL_CACHE_ROOT, f"paloma/{eval_set}-hero-bpe-v16384/{PALOMA_CACHE_VERSION}/validation")
    return prefix_join(
        EVAL_CACHE_ROOT, f"uncheatable_eval/{eval_set}-hero-bpe-v16384/{UNCHEATABLE_CACHE_VERSION}/validation"
    )


def _documents(eval_set: str) -> list[np.ndarray]:
    cache = TreeCache.load(_cache_path(eval_set), {"input_ids": np.zeros((0,), dtype=np.int32)})
    documents, total = [], 0
    for start in range(0, len(cache), TEXT_READ_BATCH):
        for row in cache.get_batch_sync(slice(start, min(start + TEXT_READ_BATCH, len(cache)))):
            ids = np.asarray(row["input_ids"], dtype=np.int32)
            documents.append(ids)
            total += len(ids)
            if total >= MAX_TEXT_TOKENS_PER_SET:
                return documents
    return documents


def _windows(ids: Sequence[int]) -> list[Sequence[int]]:
    return [ids[start : start + MAX_REQUEST_TOKENS] for start in range(0, len(ids) - 1, MAX_REQUEST_TOKENS - 1)]


def score_text(model: Transformer, tokenizer: MarinTokenizer, eval_sets: Sequence[str]) -> dict[str, dict[str, float]]:
    """Loss by token class, and bits per byte of the original and whitespace-collapsed text."""
    classes = token_classes(tokenizer, V16384_VOCAB)
    results: dict[str, dict[str, float]] = {}
    for eval_set in eval_sets:
        documents = _documents(eval_set)
        windows = [window for ids in documents for window in _windows(ids.tolist())]
        loss_by_class = np.zeros(len(CLASS_NAMES))
        count_by_class = np.zeros(len(CLASS_NAMES))
        loss_by_context = np.zeros(len(CONTEXT_BUCKETS) + 1)
        count_by_context = np.zeros(len(CONTEXT_BUCKETS) + 1)
        for window, (log_probs, _) in zip(windows, score_sequences(model, windows), strict=True):
            target_classes = classes[np.asarray(window[1:])]
            np.add.at(loss_by_class, target_classes, -log_probs)
            np.add.at(count_by_class, target_classes, 1)
            # The context length of a target is its position in the window.
            context_buckets = np.digitize(np.arange(1, len(window)), CONTEXT_BUCKETS)
            np.add.at(loss_by_context, context_buckets, -log_probs)
            np.add.at(count_by_context, context_buckets, 1)
        scored = count_by_class.sum()
        row = {"tokens": float(scored), "loss": float(loss_by_class.sum() / scored)}
        for index, name in enumerate(CONTEXT_BUCKET_NAMES):
            row[f"context_{name}_loss"] = float(loss_by_context[index] / max(count_by_context[index], 1))
            row[f"context_{name}_token_share"] = float(count_by_context[index] / scored)
        for index, name in enumerate(CLASS_NAMES):
            row[f"{name}_token_share"] = float(count_by_class[index] / scored)
            row[f"{name}_loss"] = float(loss_by_class[index] / max(count_by_class[index], 1))
            # The class contribution uses the original token denominator, so the contributions sum to ``loss``.
            row[f"{name}_loss_contribution"] = float(loss_by_class[index] / scored)
        row.update(_fixed_target_context_losses(model, documents))
        texts = [tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in documents]
        row["bpb"] = _text_bpb(model, tokenizer, texts)
        if eval_set not in CODE_SETS:
            row["collapsed_bpb"] = _text_bpb(model, tokenizer, [collapse_whitespace(text) for text in texts])
        results[eval_set] = row
        logger.info("scored %s: %s", eval_set, row)
    return results


def _fixed_target_context_losses(model: Transformer, documents: Sequence[np.ndarray]) -> dict[str, float]:
    long_documents = [ids for ids in documents if len(ids) >= MAX_REQUEST_TOKENS][:MAX_CONTEXT_TEST_DOCUMENTS]
    if not long_documents:
        return {}
    results = {"fixed_target_documents": float(len(long_documents))}
    for context in TARGET_CONTEXTS:
        sequences = [
            ids[MAX_REQUEST_TOKENS - TARGET_SPAN - context : MAX_REQUEST_TOKENS].tolist() for ids in long_documents
        ]
        losses = [-float(log_probs[-TARGET_SPAN:].sum()) for log_probs, _ in score_sequences(model, sequences)]
        results[f"fixed_target_context_{context}_loss"] = float(np.sum(losses) / (TARGET_SPAN * len(sequences)))
    return results


def _text_bpb(model: Transformer, tokenizer: MarinTokenizer, texts: Sequence[str]) -> float:
    windows, total_bytes = [], 0
    for text in texts:
        ids = [tokenizer.bos_token_id, *tokenizer.encode(text, add_special_tokens=False), tokenizer.eos_token_id]
        windows.extend(_windows(ids))
        total_bytes += len(text.encode("utf-8"))
    total_loss = sum(-float(log_probs.sum()) for log_probs, _ in score_sequences(model, windows))
    return total_loss / (math.log(2) * total_bytes)


def load_model(checkpoint_root: str, *, size: str, dense: bool) -> Transformer:
    """Load the ``params`` of the latest checkpoint under ``checkpoint_root`` at the compute dtype."""
    config = dataclasses.replace(_h100_ladder_model(_h100_ladder_rung(size), dense=dense), vocab_size=V16384_VOCAB)
    if not dense:
        raise ValueError("the downstream scorer supports dense checkpoints only")
    exemplar = jax.jit(lambda: Transformer.init(config, key=jax.random.PRNGKey(0)))()
    loaded = load_checkpoint({"params": exemplar}, latest_checkpoint_path(checkpoint_root), axis_mapping=None, mesh=None)
    return jmp.get_policy(COMPUTE_POLICY).cast_to_compute(loaded["params"])


@click.command()
@click.option("--checkpoint-root", required=True, help="Directory that holds step-N checkpoints.")
@click.option("--size", required=True, type=click.Choice(["d512", "d768", "d1024", "d1280"]))
@click.option("--output", required=True, help="Directory for metrics.json and downstream_records.parquet.")
@click.option("--task", "tasks", multiple=True, help="Downstream task; omit for every task.")
@click.option("--skip-text", is_flag=True, help="Skip the held-out text scores.")
def main(checkpoint_root: str, size: str, output: str, tasks: tuple[str, ...], skip_text: bool) -> None:
    logging.basicConfig(level=logging.INFO)
    tokenizer = load_tokenizer(V16384_TOKENIZER)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with set_mesh(mesh):
        model = load_model(checkpoint_root, size=size, dense=True)
        records = score_downstream(model, tokenizer, tasks or tuple(TASK_BUILDERS))
        text = {} if skip_text else score_text(model, tokenizer, (*PALOMA_SETS, *UNCHEATABLE_SETS))
    with StoragePath(prefix_join(output, "downstream_records.parquet")).open("wb") as stream:
        pq.write_table(records, stream)
    summary = {
        "checkpoint_root": checkpoint_root,
        "size": size,
        "tasks_version": DOWNSTREAM_TASKS_VERSION,
        "scorer_version": SCORER_VERSION,
        "downstream": downstream_metrics(records),
        "text": text,
    }
    with StoragePath(prefix_join(output, "metrics.json")).open("w") as stream:
        json.dump(summary, stream, indent=1)
    print("DOWNSTREAM_METRICS " + json.dumps(summary))


if __name__ == "__main__":
    main()
