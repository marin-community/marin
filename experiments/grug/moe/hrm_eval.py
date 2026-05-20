# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HRM-Text downstream eval for Grug MoE checkpoints.

Wraps a trained Grug ``Transformer`` as a ``lm_eval`` ``TemplateLM`` so the
HRM-Text Table (GSM8k, MATH, DROP, MMLU, ARC-C, HellaSwag, Winogrande, BoolQ)
can be produced without a Grug → HF / vLLM exporter.

The wrapper only needs the Grug model's existing ``.logits(token_ids, mask)``
method:

- ``loglikelihood`` uses logits directly on packed ``(context, continuation)``
  pairs (covers all 5 MCQ-style benchmarks: BoolQ, HellaSwag, ARC-C, MMLU,
  Winogrande).
- ``generate_until`` runs naive O(seq_len²) greedy autoregression (covers the
  3 generation tasks: GSM8k, MATH, DROP).

The post-training executor step loads the final checkpoint, runs all 8
benchmarks via ``lm_eval.simple_evaluate``, and writes both the raw JSON and
a compact Markdown table mirroring HRM-Text's README to GCS.

Resource notes: needs TPU (v5p-8) for fast inference — the executor step
schedules a TPU worker.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from fray.cluster import ResourceConfig
from levanter.checkpoint import load_checkpoint
from marin.execution.executor import ExecutorStep, InputName, this_output_path, versioned
from marin.execution.remote import remote

from experiments.evals.task_configs import EvalTaskConfig
from experiments.grug.moe.model import GrugModelConfig, Transformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HrmEvalConfig:
    """Config for a post-training HRM-Text eval step."""

    model: GrugModelConfig
    """Architecture (must match the one trained)."""

    checkpoint_path: str | InputName
    """GCS path to a Grug checkpoint directory (e.g. ``gs://.../checkpoints/step-9537``).

    When this is an ``InputName`` pointing at the training step's checkpoint
    subdirectory, the executor automatically:
      (a) blocks this step until the training step succeeds, and
      (b) resolves the InputName to the concrete GCS path before ``fn`` runs.
    """

    tokenizer_name: str
    """HF tokenizer id used during training (e.g. ``meta-llama/Meta-Llama-3.1-8B``)."""

    output_path: str
    """GCS dir for raw JSON results + Markdown table."""

    tasks: Sequence[EvalTaskConfig]
    """HRM-Text Table benchmarks. Defaults to the canonical 8."""

    max_examples: int | None = None
    """Optional cap on examples per task (None = full eval set)."""

    batch_size: int = 16
    """Per-device eval batch size."""

    max_gen_toks: int = 256
    """Max tokens to generate for ``generate_until`` tasks."""

    mp: str = "params=float32,compute=bfloat16,output=bfloat16"
    seed: int = 0


# Default HRM-Text Table benchmark set.
HRM_TEXT_BENCHMARKS_DEFAULT: tuple[EvalTaskConfig, ...] = (
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8),
    EvalTaskConfig(name="minerva_math", num_fewshot=4, task_alias="math_4shot"),
    EvalTaskConfig(name="drop", num_fewshot=0),
    EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot"),
    EvalTaskConfig("arc_challenge", 10),
    EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),
    EvalTaskConfig("winogrande", 0),
    EvalTaskConfig("boolq", 10),
)


# ---------------------------------------------------------------------------
# Grug ↔ lm-eval-harness adapter
# ---------------------------------------------------------------------------


def _make_logits_fn(model: Transformer):
    """Build a jit-compiled function ``tokens[B,S] -> logits[B,S,V]``."""

    @functools.partial(jax.jit, static_argnames=())
    def _logits(tokens: jnp.ndarray) -> jnp.ndarray:
        return model.logits(tokens)

    return _logits


def _pad_or_truncate_left(ids: list[int], max_len: int, pad_id: int) -> tuple[list[int], int]:
    """Truncate from the left (keep recent context) and right-pad to ``max_len``.

    Returns (padded_ids, real_length_before_padding).
    """
    if len(ids) > max_len:
        ids = ids[-max_len:]
    real = len(ids)
    pad = max_len - real
    if pad > 0:
        ids = ids + [pad_id] * pad
    return ids, real


def _build_harness_lm(model: Transformer, tokenizer, *, batch_size: int, max_gen_toks: int):
    """Build a ``lm_eval.api.model.TemplateLM`` subclass over a Grug ``Transformer``.

    Defined inside a function so the ``lm_eval`` import is lazy (heavy deps).
    """
    from lm_eval.api.model import TemplateLM
    from tqdm import tqdm

    logits_fn = _make_logits_fn(model)
    max_seq_len = model.config.max_seq_len
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    class GrugHarnessLM(TemplateLM):
        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self._batch_size = batch_size

        @property
        def eot_token_id(self) -> int:
            return tokenizer.eos_token_id

        @property
        def max_length(self) -> int:
            return max_seq_len

        @property
        def max_gen_toks(self) -> int:
            return max_gen_toks

        @property
        def batch_size(self) -> int:
            return self._batch_size

        @property
        def device(self) -> str:
            return "tpu"

        def tok_encode(self, string: str, **kwargs) -> list[int]:
            return tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens: list[int], **kwargs) -> str:
            return tokenizer.decode(tokens, skip_special_tokens=True)

        def _model_call(self, tokens_batch: np.ndarray) -> np.ndarray:
            """Run a batched forward pass; returns logits array [B, S, V]."""
            return np.asarray(logits_fn(jnp.asarray(tokens_batch)))

        def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
            results = []
            for i in tqdm(range(0, len(requests), self._batch_size), disable=disable_tqdm, desc="loglikelihood"):
                batch = requests[i : i + self._batch_size]
                tokens_batch = []
                meta = []
                for _, context_enc, continuation_enc in batch:
                    full = (context_enc + continuation_enc)[-max_seq_len:]
                    cont_len = len(continuation_enc)
                    if cont_len > max_seq_len - 1:
                        cont_len = max_seq_len - 1
                    ids, real_len = _pad_or_truncate_left(full, max_seq_len, pad_id)
                    tokens_batch.append(ids)
                    meta.append((real_len, cont_len, continuation_enc[-cont_len:] if cont_len > 0 else []))

                logits = self._model_call(np.asarray(tokens_batch, dtype=np.int32))  # [B, S, V]
                logprobs = logits - jax.nn.logsumexp(jnp.asarray(logits), axis=-1, keepdims=True)
                logprobs = np.asarray(logprobs)

                for b, (real_len, cont_len, cont_ids) in enumerate(meta):
                    if cont_len <= 0 or not cont_ids:
                        results.append((0.0, True))
                        continue
                    # logits at position i predict token i+1. So continuation
                    # tokens occupy positions [real_len - cont_len, real_len), and
                    # their predictors are at positions [real_len - cont_len - 1, real_len - 1).
                    start = real_len - cont_len - 1
                    cont_logprobs = logprobs[b, start : start + cont_len]
                    if cont_logprobs.shape[0] < cont_len:
                        pad_amt = cont_len - cont_logprobs.shape[0]
                        cont_logprobs = np.concatenate(
                            [cont_logprobs, np.zeros((pad_amt, cont_logprobs.shape[1]), dtype=cont_logprobs.dtype)],
                            axis=0,
                        )
                    picked = cont_logprobs[np.arange(cont_len), np.array(cont_ids, dtype=np.int64)]
                    greedy = bool((cont_logprobs.argmax(-1) == np.array(cont_ids)).all())
                    results.append((float(picked.sum()), greedy))
            return results

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            """Perplexity-style loglikelihood; uses the standard token-window split."""
            loglikelihoods = []
            for request in tqdm(requests, disable=disable_tqdm, desc="loglikelihood_rolling"):
                (string,) = request.args
                token_ids = self.tok_encode(string)
                # Naive: single window of max_length, truncated if needed.
                # For sequences longer than max_length we'd need overlapping
                # windows — out of scope for this minimal harness.
                if not token_ids:
                    loglikelihoods.append(0.0)
                    continue
                req = [(None, [self.eot_token_id], token_ids)]
                results = self._loglikelihood_tokens(req, disable_tqdm=True)
                loglikelihoods.append(results[0][0])
            return loglikelihoods

        def generate_until(self, requests, disable_tqdm: bool = False):
            """Naive O(seq_len²) greedy autoregressive generation."""
            results = []
            for request in tqdm(requests, disable=disable_tqdm, desc="generate_until"):
                context, gen_kwargs = request.args
                until = gen_kwargs.get("until", [])
                if isinstance(until, str):
                    until = [until]
                local_max_gen = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))

                ids = self.tok_encode(context)
                generated: list[int] = []
                for _ in range(local_max_gen):
                    window = (ids + generated)[-max_seq_len:]
                    padded, real_len = _pad_or_truncate_left(window, max_seq_len, pad_id)
                    logits = self._model_call(np.asarray([padded], dtype=np.int32))[0]  # [S, V]
                    next_token = int(logits[real_len - 1].argmax())
                    if next_token == self.eot_token_id:
                        break
                    generated.append(next_token)
                    text = self.tok_decode(generated)
                    if any(stop in text for stop in until):
                        for stop in until:
                            if stop in text:
                                text = text.split(stop, 1)[0]
                        results.append(text)
                        break
                else:
                    results.append(self.tok_decode(generated))
            return results

    return GrugHarnessLM()


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------


def _convert_to_lm_eval_task_spec(tasks: Sequence[EvalTaskConfig]) -> list[dict]:
    """Each ``EvalTaskConfig`` → ``lm_eval`` task spec dict."""
    out: list[dict] = []
    for t in tasks:
        spec: dict = {"task": t.name, "num_fewshot": t.num_fewshot}
        if t.task_alias:
            spec["task_alias"] = t.task_alias
        out.append(spec)
    return out


def _format_table(results: dict, tasks: Sequence[EvalTaskConfig]) -> str:
    """HRM-Text-style markdown table."""
    rows = ["| Task | Metric | Score |", "|------|--------|------:|"]
    for t in tasks:
        key = t.task_alias or t.name
        task_results = results.get("results", {}).get(key, {})
        if not task_results:
            rows.append(f"| {key} | — | _missing_ |")
            continue
        # Pick the canonical metric — prefer acc_norm > acc > exact_match.
        for metric in ("acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,flexible-extract"):
            if metric in task_results:
                score = task_results[metric] * 100.0
                rows.append(f"| {key} | {metric.split(',')[0]} | {score:.1f}% |")
                break
        else:
            # Fallback: first numeric value
            for m, v in task_results.items():
                if isinstance(v, (int, float)) and not math.isnan(v):
                    rows.append(f"| {key} | {m} | {v:.4f} |")
                    break
    return "\n".join(rows)


def _initialize_model(cfg: GrugModelConfig, *, key) -> Transformer:
    """Initialize a Grug Transformer with random weights (placeholder for checkpoint load)."""
    import jmp

    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    return mp.cast_to_param(Transformer.init(cfg, key=key))


def _run_hrm_eval_local(cfg: HrmEvalConfig) -> None:
    """Entrypoint: load checkpoint, run eval, write results."""
    import lm_eval
    from rigging.filesystem import open_url
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    with mesh:
        model_init = _initialize_model(cfg.model, key=jax.random.PRNGKey(cfg.seed))
        # The training loop stores ``GrugTrainState.params`` as the model; the
        # checkpoint layout is ``{"params": <model_tree>, ...}``. We only need
        # params for inference, so load the params sub-tree.
        loaded = load_checkpoint({"params": model_init}, cfg.checkpoint_path)
        model = loaded["params"]
        # Sanity-check: model must be a Transformer
        assert isinstance(model, Transformer), f"Expected Transformer, got {type(model)}"

        harness_lm = _build_harness_lm(
            model,
            tokenizer,
            batch_size=cfg.batch_size,
            max_gen_toks=cfg.max_gen_toks,
        )

        task_specs = _convert_to_lm_eval_task_spec(cfg.tasks)
        logger.info("Running HRM-Text eval on %d tasks: %s", len(task_specs), [t["task"] for t in task_specs])
        results = lm_eval.simple_evaluate(
            model=harness_lm,
            tasks=task_specs,
            num_fewshot=None,  # per-task overrides used
            limit=cfg.max_examples,
            random_seed=cfg.seed,
            log_samples=False,
            confirm_run_unsafe_code=True,
        )

    os.makedirs(cfg.output_path, exist_ok=True) if not cfg.output_path.startswith("gs://") else None

    table_md = _format_table(results, cfg.tasks)
    raw_path = os.path.join(cfg.output_path, "results.json")
    table_path = os.path.join(cfg.output_path, "table.md")

    with open_url(raw_path, mode="w") as f:
        json.dump(results.get("results", {}), f, indent=2, default=str)
    with open_url(table_path, mode="w") as f:
        f.write("# HRM-Text Table — Grug MoE\n\n")
        f.write(f"Checkpoint: `{cfg.checkpoint_path}`\n\n")
        f.write(table_md + "\n")

    logger.info("Wrote HRM-Text eval to %s and %s", raw_path, table_path)


# ---------------------------------------------------------------------------
# Executor step factory
# ---------------------------------------------------------------------------

_HRM_EVAL_RESOURCES = ResourceConfig.with_tpu("v5p-8")


def hrm_eval_step(
    *,
    name: str,
    model: GrugModelConfig,
    checkpoint_path: str | InputName,
    tokenizer_name: str,
    tasks: Sequence[EvalTaskConfig] = HRM_TEXT_BENCHMARKS_DEFAULT,
    batch_size: int = 16,
    max_examples: int | None = None,
) -> ExecutorStep:
    """Post-training HRM-Text eval step.

    When ``checkpoint_path`` is an ``InputName`` produced by ``train_step.cd(...)``,
    the executor discovers the train step as a dependency (via ``walk_config``
    descending into the dataclass) and blocks this step until train succeeds.
    The ``InputName`` is also resolved to a concrete GCS path before ``fn`` runs.

    Individual fields are wrapped with ``versioned()`` to affect the version
    hash — the full ``HrmEvalConfig`` is **not** wrapped because that would
    stop the walker descending into it, hiding the ``InputName`` dependency.
    """
    config = HrmEvalConfig(
        model=versioned(model),
        checkpoint_path=checkpoint_path,
        tokenizer_name=versioned(tokenizer_name),
        output_path=this_output_path(),
        tasks=versioned(tuple(tasks)),
        batch_size=versioned(batch_size),
        max_examples=versioned(max_examples),
    )
    return ExecutorStep(
        name=os.path.join("grug/hrm_repro", name, "eval"),
        fn=remote(
            _run_hrm_eval_local,
            resources=_HRM_EVAL_RESOURCES,
            # torch_test provides torch for lm_eval's type-annotated APIs;
            # tpu provides JAX/libtpu for the Grug forward pass.
            pip_dependency_groups=["lm_eval", "torch_test", "tpu"],
        ),
        config=config,
    )


__all__ = [
    "HRM_TEXT_BENCHMARKS_DEFAULT",
    "HrmEvalConfig",
    "hrm_eval_step",
]
