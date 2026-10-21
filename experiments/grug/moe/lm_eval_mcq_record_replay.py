# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record lm-eval 0.4.12 loglikelihood requests for a grug checkpoint, then replay its scores through lm-eval.

Grug checkpoints can't be served to lm-eval directly, so multiple-choice tasks run in three stages:

1. ``record``: run each task through lm-eval with the ``local-completions`` model (HF tokenizer, tokenized
   requests, ``add_bos_token=True``, ``max_length=4096``) whose scoring call is replaced by a recorder. The
   recorder saves the exact token rows lm-eval would have sent (after its context/continuation split and left
   truncation) plus each row's context length.
2. Score the rows on TPU with ``eval_67b_a2b_lm_eval_mcq.py``: per row, the summed logprob of tokens
   ``ctxlen:`` and whether each was the argmax.
3. ``replay``: run lm-eval again with the same settings, returning the stored scores, so lm-eval itself
   computes acc/acc_norm with its own few-shot sampling, answer layout and aggregation.

The vLLM baselines use the same lm-eval version and ``--model_args`` (minus the recorder), so every model
sees identical prompts. Runs outside the workspace venv because the workspace pins an older lm-eval fork::

    uv run --no-project --with lm-eval==0.4.12 --with gcsfs \\
        python experiments/grug/moe/lm_eval_mcq_record_replay.py record gs://.../requests
    uv run --no-project --with lm-eval==0.4.12 --with gcsfs \\
        python experiments/grug/moe/lm_eval_mcq_record_replay.py replay gs://.../requests gs://.../scores out.json
"""

import hashlib
import io
import json
import sys

import fsspec
import lm_eval
import numpy as np
from lm_eval.models.openai_completions import LocalCompletionsAPI

TOKENIZER = "marin-community/marin-tokenizer"
MAX_LENGTH = 4096
# (lm-eval task, num_fewshot): Open LLM Leaderboard v1 shot counts, plus MMLU-Pro's loglikelihood variant.
TASKS: tuple[tuple[str, int], ...] = (
    ("mmlu", 5),
    ("leaderboard_mmlu_pro", 5),
    ("arc_challenge", 25),
    ("hellaswag", 10),
    ("winogrande", 5),
    ("piqa", 0),
)
# The TPU scorer does not compute is_greedy; these metrics only read the logprobs.
SCORED_METRICS = frozenset({"acc", "acc_norm"})
MODEL_ARGS = (
    f"base_url=http://unused/v1/completions,tokenizer_backend=huggingface,tokenized_requests=True,"
    f"tokenizer={TOKENIZER},add_bos_token=True,max_length={MAX_LENGTH}"
)


def _rows_digest(tokens: np.ndarray, offsets: np.ndarray, ctxlens: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in (tokens, offsets, ctxlens):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class _RecordingCompletions(LocalCompletionsAPI):
    """Captures the token rows lm-eval would send; returns placeholder scores."""

    rows: list[list[int]]
    ctxlens: list[int]

    def _loglikelihood_tokens(self, requests, **kwargs):
        inputs, ctxlens, _ = self.batch_loglikelihood_requests([requests])
        self.rows, self.ctxlens = inputs, ctxlens
        return [(0.0, False)] * len(requests)


class _ReplayingCompletions(LocalCompletionsAPI):
    """Returns scores computed offline for exactly the rows the recorder saved."""

    expected_digest: str
    scores: list[tuple[float, bool]]

    def _loglikelihood_tokens(self, requests, **kwargs):
        inputs, ctxlens, _ = self.batch_loglikelihood_requests([requests])
        tokens, offsets, ctxlen_arr = _flatten(inputs, ctxlens)
        assert _rows_digest(tokens, offsets, ctxlen_arr) == self.expected_digest, "replayed requests differ"
        assert len(self.scores) == len(requests)
        return self.scores


def _flatten(rows: list[list[int]], ctxlens: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(r) for r in rows])
    tokens = np.fromiter((t for r in rows for t in r), dtype=np.int32, count=int(offsets[-1]))
    return tokens, offsets, np.asarray(ctxlens, dtype=np.int32)


def _evaluate(model: LocalCompletionsAPI, task: str, num_fewshot: int) -> dict:
    return lm_eval.simple_evaluate(model=model, tasks=[task], num_fewshot=num_fewshot, batch_size=1)


def record(out_dir: str) -> None:
    for task, num_fewshot in TASKS:
        model = _RecordingCompletions.create_from_arg_string(MODEL_ARGS)
        _evaluate(model, task, num_fewshot)
        tokens, offsets, ctxlens = _flatten(model.rows, model.ctxlens)
        assert (ctxlens > 0).all() and (np.diff(offsets) > ctxlens).all()
        buf = io.BytesIO()
        np.savez_compressed(buf, tokens=tokens, offsets=offsets, ctxlens=ctxlens)
        with fsspec.open(f"{out_dir}/{task}.npz", "wb") as f:
            f.write(buf.getvalue())
        meta = {
            "task": task,
            "num_fewshot": num_fewshot,
            "rows": len(ctxlens),
            "tokens": int(offsets[-1]),
            "max_row": int(np.diff(offsets).max()),
            "digest": _rows_digest(tokens, offsets, ctxlens),
        }
        with fsspec.open(f"{out_dir}/{task}.json", "w") as f:
            json.dump(meta, f)
        print(json.dumps(meta))


def replay(requests_dir: str, scores_dir: str, out_path: str) -> None:
    results = {}
    for task, num_fewshot in TASKS:
        with fsspec.open(f"{requests_dir}/{task}.json", "r") as f:
            meta = json.load(f)
        with fsspec.open(f"{scores_dir}/{task}.npz", "rb") as f:
            scored = np.load(io.BytesIO(f.read()))
        assert str(scored["digest"]) == meta["digest"], f"{task}: scores were computed for different requests"
        model = _ReplayingCompletions.create_from_arg_string(MODEL_ARGS)
        model.expected_digest = meta["digest"]
        model.scores = [(float(lp), bool(g)) for lp, g in zip(scored["logprob"], scored["greedy"], strict=True)]
        out = _evaluate(model, task, num_fewshot)
        metrics = {m["metric"] for cfg in out["configs"].values() for m in cfg["metric_list"]}
        assert metrics <= SCORED_METRICS, f"{task} uses {metrics - SCORED_METRICS}, which may need is_greedy"
        results[task] = {"results": out["results"], "n-shot": out["n-shot"], "n-samples": out["n-samples"]}
        print(task, json.dumps(out["results"].get(task, {}), default=str))
    with fsspec.open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "record":
        record(sys.argv[2])
    elif mode == "replay":
        replay(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise ValueError(f"unknown mode {mode!r}")
