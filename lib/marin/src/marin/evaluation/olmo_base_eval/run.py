# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU runtime for the OLMoBaseEval Easy Table 9 BPB evaluator.

Loads a checkpoint (HF single-or-sharded safetensors, or a native Levanter
checkpoint), scores a frozen request set (continuation-masked, fp32 log-probs,
UTF-8 byte denominator), aggregates into the 51 Table 9 components + macro, and
logs to W&B. Mirrors the Iris-submission structure of ``trace_labeled_eval.py``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import levanter
from fray import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, TpuConfig, create_environment
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.compat.hf_checkpoints import load_tokenizer as hf_load_tokenizer
from levanter.data.loader import stack_tree
from levanter.model_loading import load_hf_checkpoint
from levanter.models.lm_model import LmExample
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.tree_utils import inference_mode

from marin.evaluation.olmo_base_eval.aggregate import assemble_table9, table9_macro
from marin.evaluation.olmo_base_eval.bpb import EncodedInstance, encode_context_continuation, task_bpb
from marin.evaluation.olmo_base_eval.components import leaf_components, mmlu_subjects, scored_tasks
from marin.evaluation.olmo_base_eval.metrics import build_wandb_metrics, sc_compat_metrics
from marin.evaluation.olmo_base_eval.request_set import RequestInstance, load_request_set, read_manifest
from marin.execution import ExecutorStep, InputName, this_output_path
from marin.utilities.executor_utils import ckpt_path_to_step_name

logger = logging.getLogger(__name__)

DEFAULT_WANDB_PROJECT = "marin-eval"
DEFAULT_WANDB_TAGS = ("olmo_base_eval_table9",)
RESULTS_FILENAME = "olmo_base_eval_table9_results.json"
# Pad each batch up to a multiple of the flash-attention block size, which the
# kernel requires the position axis to be a multiple of. This bounds the number
# of jit recompiles (one per distinct padded length) while keeping padding small.
FLASH_BLOCK = 1024


@dataclass(frozen=True)
class OlmoBaseEvalConfig:
    """Configuration for one checkpoint's Table 9 BPB evaluation."""

    name: str
    checkpoint_path: str
    checkpoint_is_hf: bool
    request_set_dir: str
    output_path: str
    trainer: TrainerConfig
    max_eval_length: int = 8192
    tokenizer: str | None = None  # defaults to the checkpoint
    prepend_bos: bool | None = None  # None => follow the tokenizer's add_bos_token attribute
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OlmoBaseEvalOnPodConfig:
    eval_config: OlmoBaseEvalConfig
    resources: ResourceConfig


@dataclass(frozen=True)
class OlmoBaseEvalOutput:
    results_path: str


def _bucket_length(length: int, max_length: int) -> int:
    bucketed = -(-length // FLASH_BLOCK) * FLASH_BLOCK  # round up to a multiple of FLASH_BLOCK
    return min(bucketed, max_length)


def _resolve_bos_token_id(hf_tokenizer, prepend_bos: bool | None) -> int | None:
    """Decide whether to prepend BOS, matching OLMo-Eval's rule.

    OLMo-Eval prepends BOS to the context iff ``getattr(tokenizer,
    'add_bos_token', False)`` is truthy. ``prepend_bos`` overrides this when set.
    """
    if prepend_bos is None:
        prepend_bos = bool(getattr(hf_tokenizer, "add_bos_token", False))
    if not prepend_bos:
        return None
    bos_id = hf_tokenizer.bos_token_id
    if bos_id is None:
        bos_id = hf_tokenizer.eos_token_id
    return bos_id


def _encode_instances(hf_tokenizer, instances: list[RequestInstance], bos_token_id: int | None) -> list[EncodedInstance]:
    def encode(text: str) -> list[int]:
        return hf_tokenizer.encode(text, add_special_tokens=False)

    return [
        encode_context_continuation(encode, inst.context, inst.continuation, bos_token_id=bos_token_id)
        for inst in instances
    ]


def score_summed_logprobs(
    model,
    encoded: list[EncodedInstance],
    *,
    model_config,
    EvalBatch,
    compute_axis_mapping,
    mp: jmp.Policy,
    pad_id: int,
    max_eval_length: int,
) -> list[float]:
    """Return the summed continuation log-probability (fp32) for each instance.

    Sequences are length-sorted and each batch is padded to a length bucket, so
    the masked per-position negative-log-likelihood is computed only over real
    tokens (the continuation mask zeroes prompt and padding positions).
    """
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i].tokens))
    results: list[float] = [0.0] * len(encoded)
    batch_size = EvalBatch.size
    scorers: dict[int, tuple] = {}

    def get_scorer(pos_len: int):
        if pos_len not in scorers:
            Pos = model_config.max_Pos.resize(pos_len)

            @hax.named_jit(axis_resources=compute_axis_mapping)
            def _score(scored_model, batch):
                scored_model = inference_mode(scored_model, True)
                scored_model = mp.cast_to_compute(scored_model)
                per_position_nll = scored_model.compute_next_token_loss(batch, reduction=None, loss_dtype=jnp.float32)
                return hax.sum(-per_position_nll, axis=Pos)

            scorers[pos_len] = (Pos, _score)
        return scorers[pos_len]

    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        batch_max = max(len(encoded[i].tokens) for i in batch_indices)
        pos_len = _bucket_length(batch_max, max_eval_length)
        Pos, scorer = get_scorer(pos_len)

        examples = []
        for i in batch_indices:
            instance = encoded[i]
            tokens = instance.tokens
            prompt_length = instance.prompt_length
            if len(tokens) > pos_len:
                # Left-truncate (keep the continuation, drop oldest context); rare.
                dropped = len(tokens) - pos_len
                tokens = tokens[dropped:]
                prompt_length = max(0, prompt_length - dropped)
                logger.warning("truncated instance from %d to %d tokens", len(instance.tokens), pos_len)
            padded = list(tokens) + [pad_id] * (pos_len - len(tokens))
            tokens_named = hax.named(jnp.asarray(padded, dtype=jnp.int32), Pos)
            examples.append(LmExample.from_prompt_and_completion(Pos, tokens_named, prompt_length, ignore_id=pad_id))

        while len(examples) < batch_size:  # pad the final partial batch
            examples.append(examples[-1])
        batch = stack_tree(EvalBatch, examples)
        summed = jax.device_get(scorer(model, batch).array)
        for position, i in enumerate(batch_indices):
            results[i] = float(summed[position])
    return results


def _score_tasks(
    model, hf_tokenizer, requests_by_task: dict[str, list[RequestInstance]], config: OlmoBaseEvalConfig, model_config, mp
) -> dict[str, float]:
    bos_token_id = _resolve_bos_token_id(hf_tokenizer, config.prepend_bos)
    logger.info("prepend BOS token id: %s", bos_token_id)
    EvalBatch = config.trainer.EvalBatch
    compute_axis_mapping = config.trainer.compute_axis_mapping
    pad_id = hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else hf_tokenizer.eos_token_id

    task_scores: dict[str, float] = {}
    for task in sorted(requests_by_task):
        instances = requests_by_task[task]
        encoded = _encode_instances(hf_tokenizer, instances, bos_token_id)
        summed_logprobs = score_summed_logprobs(
            model,
            encoded,
            model_config=model_config,
            EvalBatch=EvalBatch,
            compute_axis_mapping=compute_axis_mapping,
            mp=mp,
            pad_id=pad_id,
            max_eval_length=config.max_eval_length,
        )
        num_bytes = [e.num_bytes for e in encoded]
        task_scores[task] = task_bpb(summed_logprobs, num_bytes)
        logger.info("task %s: %d instances, bpb=%.6f", task, len(instances), task_scores[task])
    return task_scores


def _build_metrics_and_results(task_scores: dict[str, float], manifest, config: OlmoBaseEvalConfig) -> tuple[dict, dict]:
    """Build the W&B metric dict and the results record from per-task BPB."""
    present = set(task_scores)
    leaves = {task: task_scores[task] for task in leaf_components() if task in present}
    subjects = {subject: task_scores[subject] for subject in mmlu_subjects() if subject in present}

    metrics: dict[str, float] = {}
    macro: float | None = None
    components: dict[str, float] = {}
    if set(scored_tasks()).issubset(present):
        components = assemble_table9(leaves, subjects)
        macro = table9_macro(components)
        metrics = build_wandb_metrics(
            component_bpb=components, leaf_bpb=leaves, mmlu_subject_bpb=subjects, macro_bpb=macro
        )
    else:
        # Partial request set (e.g. a smoke subset): still emit SC-compatible keys.
        missing = sorted(set(scored_tasks()) - present)
        logger.warning("partial request set; %d scored tasks missing, skipping macro: %s", len(missing), missing[:10])
        metrics = sc_compat_metrics(leaves, subjects)

    results = {
        "name": config.name,
        "checkpoint_path": config.checkpoint_path,
        "request_set_dir": config.request_set_dir,
        "request_set_version": manifest.version,
        "olmo_eval_git_sha": manifest.olmo_eval_git_sha,
        "num_instances": dict(manifest.tasks),
        "task_bpb": task_scores,
        "table9_components": components,
        "table9_macro_bpb": macro,
        "provenance": config.provenance,
    }
    return metrics, results


def olmo_base_eval(config: OlmoBaseEvalConfig) -> None:
    """Score one checkpoint over the Table 9 request set and log BPB metrics."""
    if not config.checkpoint_is_hf:
        raise NotImplementedError("native Levanter checkpoints are not yet supported; export to HF first")

    levanter.initialize(config)
    try:
        checkpoint = config.checkpoint_path
        tokenizer_source = config.tokenizer or checkpoint
        hf_tokenizer = hf_load_tokenizer(tokenizer_source)

        manifest = read_manifest(config.request_set_dir)
        requests_by_task = load_request_set(config.request_set_dir)
        unknown = sorted(set(requests_by_task) - set(scored_tasks()))
        if unknown:
            raise ValueError(f"request set has tasks not in the Table 9 registry: {unknown}")
        logger.info("loaded %d tasks, %d instances", len(requests_by_task), sum(manifest.tasks.values()))

        converter = HFCheckpointConverter.from_hf(checkpoint, trust_remote_code=False)
        model_config = converter.config_from_hf_checkpoint(checkpoint)
        mp: jmp.Policy = config.trainer.mp

        with config.trainer.use_device_mesh():
            model = load_hf_checkpoint(
                model_config,
                checkpoint,
                axis_mapping=config.trainer.parameter_axis_mapping,
                tokenizer=hf_tokenizer,
                compute_dtype=mp.compute_dtype,
            )
            task_scores = _score_tasks(model, hf_tokenizer, requests_by_task, config, model_config, mp)

        metrics, results = _build_metrics_and_results(task_scores, manifest, config)
        if jax.process_index() == 0:
            _write_results(config.output_path, results)
        if metrics:
            levanter.tracker.log(metrics, step=0)
            if results["table9_macro_bpb"] is not None:
                logger.info("table9_macro_bpb = %.6f", results["table9_macro_bpb"])
    finally:
        levanter.tracker.current_tracker().finish()


def _write_results(output_path: str, results: dict) -> None:
    path = os.path.join(output_path, RESULTS_FILENAME)
    with fsspec.open(path, "w") as handle:
        handle.write(json.dumps(results, indent=2))
    logger.info("wrote results to %s", path)


def run_olmo_base_eval_on_pod(config: OlmoBaseEvalOnPodConfig) -> OlmoBaseEvalOutput:
    """Submit the evaluation as a fray job and wait for completion."""
    client = current_client()
    extras = ["tpu"] if isinstance(config.resources.device, TpuConfig) else []
    job = client.submit(
        JobRequest(
            name=f"olmo-base-eval-{config.eval_config.name}".replace("_", "-")[:60],
            entrypoint=Entrypoint.from_callable(olmo_base_eval, args=[config.eval_config]),
            resources=config.resources,
            environment=create_environment(extras=extras),
            max_retries_failure=1,
            max_task_failures=1,
        )
    )
    job.wait(raise_on_failure=True)
    return OlmoBaseEvalOutput(results_path=os.path.join(config.eval_config.output_path, RESULTS_FILENAME))


def olmo_base_eval_step(
    *,
    checkpoint: str | InputName,
    request_set_dir: str,
    resource_config: ResourceConfig,
    checkpoint_is_hf: bool = True,
    tokenizer: str | None = None,
    per_device_batch_size: int = 4,
    max_eval_length: int = 8192,
    name: str | None = None,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_tags=DEFAULT_WANDB_TAGS,
    wandb_group: str | None = None,
    provenance: dict[str, str] | None = None,
) -> ExecutorStep:
    """Create an ExecutorStep that scores a checkpoint on the Table 9 BPB suite."""
    if not name:
        name = ckpt_path_to_step_name(checkpoint)
    return ExecutorStep(
        name=f"evaluation/olmo_base_eval_table9/{name}",
        fn=run_olmo_base_eval_on_pod,
        config=OlmoBaseEvalOnPodConfig(
            eval_config=OlmoBaseEvalConfig(
                name=name,
                checkpoint_path=checkpoint,  # type: ignore[arg-type]
                checkpoint_is_hf=checkpoint_is_hf,
                request_set_dir=request_set_dir,
                tokenizer=tokenizer,
                max_eval_length=max_eval_length,
                output_path=this_output_path(),
                provenance=provenance or {},
                trainer=TrainerConfig(
                    tracker=(
                        WandbConfig(project=wandb_project, name=name, tags=list(wandb_tags), group=wandb_group),
                        JsonFileTrackerConfig(output_path=this_output_path()),
                    ),
                    per_device_eval_parallelism=per_device_batch_size,
                    mp=jmp.get_policy("c=bf16"),
                ),
            ),
            resources=resource_config,
        ),
    )
