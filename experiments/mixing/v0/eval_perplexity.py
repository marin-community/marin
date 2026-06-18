# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw-text perplexity eval for grug-MoE checkpoints over David's PPL bundles.

Companion to ``eval_logprob.py``. Where that script runs lm-eval-harness tasks,
this one streams documents from the ``RawTextEvaluationDataset`` definitions in
``experiments.evals.perplexity_gap_registry`` and computes per-dataset bits-per-byte
directly via grug's ``Transformer.next_token_loss``. Skips levanter's PPL
pipeline because it expects an ``LmConfig``-typed model and grug's
``GrugModelConfig`` doesn't inherit from that hierarchy.

Each (model, bundle) is one ``ExecutorStep`` that emits ``results.json`` with
``{dataset_key: {bpb, nll, n_docs, n_bytes, n_tokens}}``.
"""

import dataclasses
import gzip
import json
import logging
import math
import os
from dataclasses import dataclass

import datasets as hf_datasets
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.tokenizers import load_tokenizer
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.execution.remote import remote
from marin.execution.types import this_output_path
from rigging.filesystem import filesystem as marin_filesystem

from experiments.evals.perplexity_gap_registry import registered_perplexity_gap_bundles
from experiments.marin_models import marin_tokenizer
from experiments.mixing.v0.heuristic import build_from_heuristic
from experiments.mixing.v0.model import GrugModelConfig, Transformer

logger = logging.getLogger(__name__)

NAT_TO_BIT = 1.0 / math.log(2)
# v4-8 has a `data: 4` axis in the eval mesh; the batch dim must be a multiple.
_BATCH_AXIS_MULTIPLE = 4


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    hidden_dim: int
    budget: float
    checkpoint_subpath: str


@eqx.filter_jit
def _batch_next_token_logprobs(transformer, tokens, loss_weight):
    """Sum of next-token log-likelihoods per row.

    Mirrors ``eval_logprob._batch_logprobs`` but kept here so this script can
    stand alone if needed. Returns ``f32 (B,)`` total log-likelihood per row.
    """
    per_pos_loss = transformer.next_token_loss(tokens, loss_weight, reduction="none")
    return -jnp.sum(per_pos_loss, axis=-1)


@dataclass(frozen=True)
class GrugPerplexityEvalConfig:
    grug_model_config: GrugModelConfig
    checkpoint_path: str
    output_path: str
    bundle_key: str
    datasets: dict[str, RawTextEvaluationDataset]
    eval_capacity_factor: float = 8.0
    batch_size: int = 8
    max_eval_length: int = 4096
    max_docs_per_dataset: int | None = 256
    max_doc_bytes: int | None = 32_768


def _iter_dataset_docs(ds_config: RawTextEvaluationDataset):
    """Yield ``(text, n_bytes_utf8)`` per doc.

    Supports either HF datasets or ``input_path`` globs of ``.jsonl.gz`` files
    (David's bundles use both). Supervised datasets (input_key + target_key
    set) yield ``input + target`` concatenated.
    """

    def _row_to_text(row: dict) -> str:
        if ds_config.input_key is not None and ds_config.target_key is not None:
            return str(row[ds_config.input_key]) + str(row[ds_config.target_key])
        return str(row[ds_config.text_key])

    if ds_config.hf_dataset_id is not None:
        ds = hf_datasets.load_dataset(
            ds_config.hf_dataset_id,
            name=ds_config.hf_dataset_name,
            revision=ds_config.hf_dataset_revision,
            split=ds_config.split,
            streaming=True,
        )
        for row in ds:
            text = _row_to_text(row)
            if text:
                yield text, len(text.encode("utf-8"))
        return

    if ds_config.input_path is None:
        raise ValueError(f"RawTextEvaluationDataset has neither hf_dataset_id nor input_path: {ds_config!r}")

    # GCS file glob — auto-detect parquet vs jsonl.gz from extension.
    fs = marin_filesystem("gcs")
    paths = sorted(fs.glob(str(ds_config.input_path)))
    if not paths:
        raise FileNotFoundError(f"No files matched input_path glob: {ds_config.input_path!r}")
    is_parquet = paths[0].endswith(".parquet")
    if is_parquet:
        import pyarrow.parquet as pq

        for p in paths:
            with fs.open(f"gs://{p}" if not p.startswith("gs://") else p, "rb") as raw:
                table = pq.read_table(raw)
            for row in table.to_pylist():
                text = _row_to_text(row)
                if text:
                    yield text, len(text.encode("utf-8"))
    else:
        for p in paths:
            with (
                fs.open(f"gs://{p}" if not p.startswith("gs://") else p, "rb") as raw,
                gzip.GzipFile(fileobj=raw, mode="rb") as gz,
            ):
                for line in gz:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = _row_to_text(row)
                    if text:
                        yield text, len(text.encode("utf-8"))


def _score_one_dataset(
    transformer,
    tokenizer,
    ds_key: str,
    ds_config: RawTextEvaluationDataset,
    *,
    max_eval_length: int,
    max_docs: int | None,
    max_doc_bytes: int | None,
    batch_size: int,
    pad_id: int,
) -> dict[str, float]:
    """Stream docs, tokenize, batch, and accumulate per-dataset bpb.

    Padding is right-aligned to the batch's longest item (capped at
    ``max_eval_length``); ``loss_weight`` zeroes out padding so it contributes
    no log-likelihood.
    """
    sum_logprob = 0.0
    sum_bytes = 0
    sum_tokens = 0
    n_docs = 0

    pending_tokens: list[np.ndarray] = []
    pending_bytes: list[int] = []

    def _flush(token_lists, byte_counts):
        nonlocal sum_logprob, sum_tokens
        if not token_lists:
            return
        max_len_b = max(len(t) for t in token_lists)
        max_len_b = min(max_len_b, max_eval_length)
        # Pad to a multiple of 128 — grug's splash-attention kernel rejects
        # other key/value sequence lengths.
        max_len_b = max(128, ((max_len_b + 127) // 128) * 128)
        max_len_b = min(max_len_b, max_eval_length)
        # Pad the batch dim to a multiple of the model's data-axis size, otherwise
        # the JAX sharding rejects e.g. a 3-row batch on a `data: 4` mesh. The
        # padding rows have loss_weight=0 so they contribute nothing.
        n_real = len(token_lists)
        batch_dim = ((n_real + _BATCH_AXIS_MULTIPLE - 1) // _BATCH_AXIS_MULTIPLE) * _BATCH_AXIS_MULTIPLE
        batch_dim = max(batch_dim, _BATCH_AXIS_MULTIPLE)
        batch_tokens = np.full((batch_dim, max_len_b), pad_id, dtype=np.int32)
        batch_weight = np.zeros((batch_dim, max_len_b), dtype=np.float32)
        for i, t in enumerate(token_lists):
            n = min(len(t), max_len_b)
            batch_tokens[i, :n] = t[:n]
            batch_weight[i, :n] = 1.0
            sum_tokens += int(n)
        logprobs = np.asarray(
            _batch_next_token_logprobs(transformer, jnp.asarray(batch_tokens), jnp.asarray(batch_weight))
        )
        # Only the first n_real rows are real; padding rows have weight 0 and
        # therefore zero logprob — safe to sum the whole vector, but slice for
        # clarity in case the model ever emits non-zero padding logprobs.
        sum_logprob += float(logprobs[:n_real].sum())

    for text, n_bytes in _iter_dataset_docs(ds_config):
        if max_docs is not None and n_docs >= max_docs:
            break
        if max_doc_bytes is not None and n_bytes > max_doc_bytes:
            text = text.encode("utf-8")[:max_doc_bytes].decode("utf-8", errors="ignore")
            n_bytes = len(text.encode("utf-8"))
        tokens = np.asarray(tokenizer.encode(text), dtype=np.int32)
        if tokens.size == 0:
            continue
        if tokens.size > max_eval_length:
            tokens = tokens[:max_eval_length]
        pending_tokens.append(tokens)
        pending_bytes.append(n_bytes)
        sum_bytes += n_bytes
        n_docs += 1
        if len(pending_tokens) >= batch_size:
            _flush(pending_tokens, pending_bytes)
            pending_tokens, pending_bytes = [], []

    _flush(pending_tokens, pending_bytes)

    if sum_bytes == 0 or n_docs == 0:
        logger.warning("[%s] no scorable docs; emitting NaN", ds_key)
        return {"bpb": float("nan"), "nll": float("nan"), "n_docs": 0, "n_bytes": 0, "n_tokens": 0}

    bpb = (-sum_logprob / sum_bytes) * NAT_TO_BIT
    return {
        "bpb": float(bpb),
        "nll": float(-sum_logprob),
        "n_docs": int(n_docs),
        "n_bytes": int(sum_bytes),
        "n_tokens": int(sum_tokens),
    }


def run_grug_perplexity_eval(config: GrugPerplexityEvalConfig) -> None:
    eval_grug = dataclasses.replace(
        config.grug_model_config,
        capacity_factor=config.eval_capacity_factor,
    )

    trainer_config = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        per_device_eval_parallelism=1,
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
    )
    trainer_config.initialize()
    is_chief = jax.process_index() == 0

    with trainer_config.use_device_mesh():
        key = jax.random.PRNGKey(0)
        with use_cpu_device():
            transformer_shape = eqx.filter_eval_shape(Transformer.init, eval_grug, key=key)
            ckpt_path = latest_checkpoint_path(str(config.checkpoint_path))
            transformer = load_checkpoint(
                transformer_shape,
                ckpt_path,
                subpath="params",
                axis_mapping=trainer_config.parameter_axis_mapping,
            )
        tokenizer = load_tokenizer(marin_tokenizer)

        if not is_chief:
            return  # single-host scoring; non-chief workers idle

        results: dict[str, dict[str, float]] = {}
        for ds_key, ds_config in config.datasets.items():
            logger.info("Scoring %s::%s", config.bundle_key, ds_key)
            try:
                results[ds_key] = _score_one_dataset(
                    transformer,
                    tokenizer,
                    ds_key,
                    ds_config,
                    max_eval_length=config.max_eval_length,
                    max_docs=config.max_docs_per_dataset,
                    max_doc_bytes=config.max_doc_bytes,
                    batch_size=config.batch_size,
                    pad_id=tokenizer.eos_token_id,
                )
            except (NotImplementedError, FileNotFoundError, KeyError, ValueError) as e:
                logger.warning("Skipping %s::%s: %s", config.bundle_key, ds_key, e)
                results[ds_key] = {"error": str(e)}

    results_path = os.path.join(config.output_path, "results.json")
    logger.info("Uploading perplexity results to %s", results_path)
    fs = marin_filesystem("gcs")
    with fs.open(results_path, "w") as f:
        json.dump({"bundle": config.bundle_key, "results": results}, f, indent=2)


_TARGET_STEPS = 2**14

_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        slug="grug_moe_mix_d512",
        hidden_dim=512,
        budget=2.19e17,
        checkpoint_subpath="grug/grug_moe_mix_d512-2.19e+17-e6a48f/checkpoints",
    ),
)


def _build_step(model: ModelSpec, bundle) -> ExecutorStep:
    grug_model, _, _, _ = build_from_heuristic(
        budget=model.budget,
        hidden_dim=model.hidden_dim,
        target_steps=_TARGET_STEPS,
    )
    return ExecutorStep(
        name=f"evaluation/grug_ppl/{model.slug}-{model.budget:.2e}/{bundle.key}",
        fn=remote(
            run_grug_perplexity_eval,
            resources=ResourceConfig.with_tpu("v4-8", zone="us-central2-b", preemptible=True),
            pip_dependency_groups=["eval", "tpu"],
        ),
        config=GrugPerplexityEvalConfig(
            grug_model_config=grug_model,
            checkpoint_path=InputName.hardcoded(model.checkpoint_subpath),
            output_path=this_output_path(),
            bundle_key=bundle.key,
            datasets=bundle.datasets(),
            max_eval_length=bundle.max_eval_length,
            max_docs_per_dataset=bundle.max_docs_per_dataset,
            max_doc_bytes=bundle.max_doc_bytes,
        ),
    )


grug_perplexity_steps: list[ExecutorStep] = [
    _build_step(model, bundle) for model in _MODEL_SPECS for bundle in registered_perplexity_gap_bundles()
]


if __name__ == "__main__":
    executor_main(steps=grug_perplexity_steps)
