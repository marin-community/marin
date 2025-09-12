# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
This module contains code for running the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
inside Levanter runs. The EleutherAI LM Evaluation Harness is a tool for evaluating language models on a variety of tasks.

The [run_lm_eval_harness][] function runs the EleutherAI LM Evaluation Harness on a given model and tasks, and returns the
results.

It can also be used as a callback, via the [lm_eval_harness][] function.

Note that Levanter does not support generation (use VLLM or something) and the [generate_until][] method is not implemented.
So we only support tasks that work with loglikelihood, which is most(?) of them.

References:

* https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py
* https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/TPU_cluster.py#L6

"""

import dataclasses
import json
import logging
import tempfile
import typing
from dataclasses import dataclass
from functools import cached_property
from typing import Iterator, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import PartitionSpec

import haliax
from haliax import NamedArray

import levanter.tracker
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.data.packing import (
    PromptCompletion,
    greedy_pack_prompt_completions,
    per_segment_correct,
    per_segment_loss,
)
from levanter.models.gpt2 import Gpt2Config
from levanter.models.loss import next_token_loss
from levanter.utils.background_iterable import BackgroundIterator
from levanter.utils.hf_utils import HfTokenizer
from levanter.utils.py_utils import set_global_rng_seeds


try:
    from lm_eval import evaluator
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
except ImportError:
    LM = object
    Instance = object
    evaluator = object

from tqdm_loggable.auto import tqdm

import haliax as hax
from haliax.partitioning import ResourceMapping, round_axis_for_partitioning

import levanter.config
from levanter.callbacks import StepInfo
from levanter.checkpoint import load_checkpoint
from levanter.data import batched
from levanter.data.loader import stack_batches
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import broadcast_shard, use_cpu_device
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


# OK, so LM-Eval-Harness is not deterministic. This means we can't just run it on different workers and expect the
# order of requests to be the same. Sorting doesn't even seem to be correct (?!?!?) so we need to only run it on one
# process.
# This is our design:
# 1. Process 0 creates an LevanterHarnessLM object.
# 2. On all processes, we start a loop that waits for a request using jnp_broadcast_one_to_all
# 3. When a request is received (and it's not STOP) we process the request. The results are broadcast to all
#    devices, and process 0 records htem.
# 4. When a STOP request is received, we stop the loop and process 0 returns the results.


class _LmEvalHarnessWorker:
    """
    Worker for running the LM Eval Harness. Each worker process will run a copy of this class.
    The head process will run the main harness and dispatch requests to the workers while the
    others run in a loop waiting for requests.
    """

    def __init__(
        self, EvalBatch, EvalPos, model, axis_resources, tokenizer, mp, max_packed_segments, apply_chat_template=False
    ):
        self.tokenizer = tokenizer
        self.max_packed_segments = max_packed_segments
        self.EvalBatch = EvalBatch
        self.EvalPos = EvalPos
        self.model = model
        self.axis_resources = axis_resources
        self.mp = mp
        self.max_packed_segments = max_packed_segments
        self.apply_chat_template = apply_chat_template

        self._dummy_batch = _make_dummy_batch(EvalBatch, EvalPos)

        def _eval_loglikelihood(
            model: LmHeadModel, packed_example: LmExample
        ) -> tuple[NamedArray, NamedArray, NamedArray]:
            """
            Returns:
                - segments: The segment IDs of the completions. (shape: (Segments,))
                - loss: The log-likelihood of the completion. (shape: (Segments,))
                - correct: Whether the completion is correct or not. (shape: (Segments,))
            """

            if self.mp is not None:
                model = self.mp.cast_to_compute(model)

            logits = model(packed_example.tokens, attn_mask=packed_example.attn_mask)
            logits = logits.astype(jnp.float32)
            Pos = logits.resolve_axis(self.EvalPos.name)

            loss = next_token_loss(
                Pos=Pos,
                Vocab=model.Vocab,
                logits=logits,
                true_ids=packed_example.tokens,
                loss_mask=packed_example.loss_mask,
                reduction=None,
            )

            # We need to compute losses and also whether or not the completion is correct
            # (i.e. the greedy prediction is the target)
            pred_targets = hax.argmax(logits, axis=model.Vocab)
            targets = hax.roll(packed_example.tokens, -1, axis=Pos)
            is_correct = targets == pred_targets

            # we need + 1 because we use -1 as a padding value for segments
            max_Segments = hax.Axis("Segments", size=self.max_packed_segments + 1)

            batched_segment_ids, batched_per_segment_losses = hax.vmap(per_segment_loss, self.EvalBatch)(
                packed_example, loss, max_Segments
            )

            _, batched_per_segment_correct = hax.vmap(per_segment_correct, self.EvalBatch)(
                packed_example, is_correct, max_Segments
            )

            segments = hax.flatten(batched_segment_ids, "segment")
            losses = hax.flatten(batched_per_segment_losses, "segment")
            correct = hax.flatten(batched_per_segment_correct, "segment")

            return segments, -losses, correct

        # no sharded outputs
        self._jit_loglikelihood = hax.named_jit(
            _eval_loglikelihood, axis_resources=axis_resources, out_axis_resources={}
        )

    def make_harness_lm(self, apply_chat_template: bool = False):
        if jax.process_index() == 0:
            return LevanterHarnessLM(self)
        else:
            raise ValueError("Only process 0 can create the harness")

    def worker_message_loop(self):
        while True:
            message = self._receive_message()

            if message == _Message.STOP:
                return
            elif message == _Message.LOGLIKELIHOOD:
                payload = self._receive_payload()
                self.process_loglikelihood(payload)
            else:
                raise ValueError(f"Unknown message type: {message}")

    def _receive_message(self):
        stop_message = jnp.array(_Message.STOP)
        message = broadcast_shard(stop_message, PartitionSpec())
        return message.item()

    def _receive_payload(self):
        payload = broadcast_shard(
            self._dummy_batch,
            hax.partitioning.infer_resource_partitions(self._dummy_batch, preserve_existing_shardings=False),
        )
        return payload

    def _send_message(self, message):
        assert jax.process_index() == 0
        out = broadcast_shard(jnp.array(message), PartitionSpec())
        return out

    def _send_payload(self, payload):
        assert jax.process_index() == 0
        out = broadcast_shard(
            payload, hax.partitioning.infer_resource_partitions(payload, preserve_existing_shardings=False)
        )
        return out

    def process_loglikelihood(self, packed_request):
        out = self._jit_loglikelihood(self.model, packed_request)
        return out

    def dispatch_loglikelihood(self, packed_request):
        self._send_message(_Message.LOGLIKELIHOOD)
        self._send_payload(packed_request)
        return self.process_loglikelihood(packed_request)

    def stop(self):
        self._send_message(_Message.STOP)


class _Message:
    STOP = 0
    LOGLIKELIHOOD = 1


def _get_segments_this_batch(batch, max_segments_per_ex):
    unique_segs = np.unique(batch.attn_mask.segment_ids[0].array).tolist()
    # + 1 because we use -1 as a padding value for segments and allow that
    if len(unique_segs) > max_segments_per_ex + 1:
        raise ValueError(f"Too many segments in batch: {len(unique_segs)}")
    if -1 in unique_segs:
        unique_segs.remove(-1)

    return unique_segs


def _get_padding_count(batch, pad_token_id):
    # returns the total amount of padding in the batch
    padding_count = np.sum(batch.tokens.array == pad_token_id)
    total_tokens = batch.tokens.size
    return padding_count, total_tokens


class LevanterHarnessLM(LM):
    def __init__(self, leader: _LmEvalHarnessWorker):
        super().__init__()
        self.leader = leader

    tokenizer = property(lambda self: self.leader.tokenizer)
    EvalBatch = property(lambda self: self.leader.EvalBatch)
    EvalPos = property(lambda self: self.leader.EvalPos)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.
        """
        if self.tokenizer.pad_token_id is None:
            logger.warning("No pad token set. Setting to eos token.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        packed = _pack_requests(
            requests, self.tokenizer, self.EvalPos, self.leader.max_packed_segments, self.leader.apply_chat_template
        )
        packed_iterator = stack_batches(iter(packed), self.EvalPos, self.EvalBatch)
        packed_iterator = BackgroundIterator(packed_iterator, max_capacity=1024)

        result_probs = np.zeros(len(requests))
        result_greedy = np.zeros(len(requests))
        covered_points = np.zeros(len(requests), dtype=bool)

        total_tokens_expected = len(packed) * self.EvalPos.size

        total_padding = 0
        total_tokens_seen = 0
        pbar = tqdm(total=total_tokens_expected, desc="loglikelihood", unit="tok")
        for q, batch in enumerate(packed_iterator):
            segments_this_batch = _get_segments_this_batch(
                batch, self.leader.max_packed_segments * self.EvalBatch.size
            )

            padding_count, batch_tokens = _get_padding_count(batch, self.tokenizer.pad_token_id)

            out_ids, out_lls, out_correct = self.leader.dispatch_loglikelihood(batch)

            out_ids = np.array(out_ids.array)
            out_lls = np.array(out_lls.array)
            out_correct = np.array(out_correct.array)
            # -1's are going to be where we had too few sequences to fill a batch
            valid_indices = out_ids != -1

            out_ids_this_batch = out_ids[valid_indices].tolist()

            missing_ids = set(segments_this_batch) - set(out_ids_this_batch)
            extra_ids = set(out_ids_this_batch) - set(segments_this_batch)
            assert len(missing_ids) == 0, f"Missing segments: {missing_ids}"
            assert len(extra_ids) == 0, f"Extra segments: {extra_ids}"

            result_probs[out_ids[valid_indices]] = out_lls[valid_indices]
            result_greedy[out_ids[valid_indices]] = out_correct[valid_indices]
            covered_points[out_ids[valid_indices]] = True

            total_padding += padding_count
            total_tokens_seen += batch_tokens

            pbar.set_postfix(
                padding=f"{total_padding}/{total_tokens_seen} = {(total_padding) / (total_tokens_seen):.2f}",
                this_padding=f"{padding_count}/{batch_tokens}= {padding_count / batch_tokens:.2f}",
            )
            pbar.update(batch_tokens)

        missing_points = np.where(~covered_points)[0]
        assert len(missing_points) == 0, f"Missing points: {missing_points}"

        result = list(zip(result_probs, result_greedy))
        logger.info(f"Finished running {len(requests)} loglikelihoods.")

        return result

    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        raise NotImplementedError()

    def generate_until(self, requests) -> List[str]:
        raise NotImplementedError()


@dataclass(frozen=True)
class TaskConfig:
    """
    This is a dataclass that represents the configuration for a task in the LM Eval Harness. It is used to specify
    the configuration for a task in the LM Eval Harness, and is used to generate the task dictionary that the LM Eval
    Harness expects.

    nb that LM Eval Harness has its own TaskConfig, but its defaults are not the same as just passing in
    a dict, and we want the behavior of passing in a dict.

    Nones are not included in the dictionary representation, and LM Eval Harness will use its own defaults for any
    missing values.

    Docs are copied from the LM Eval Harness task guide. The LM Eval Harness task guide is the authoritative source
    for what these fields do. They were copied as of 2024-12-03.

    See Also:
       * [LM Eval Harness TaskConfig](https://github.com/EleutherAI/lm-evaluation-harness/blob/0ef7548d7c3f01108e7c12900a5e5eb4b4a668f7/lm_eval/api/task.py#L55)
       * [LM Eval Harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md#parameters)
    """

    task: str
    """ The name of the task to run."""
    task_alias: str | None = None
    """ An alias for the task. We log this name to wandb."""
    num_fewshot: int | None = None

    use_prompt: str | None = None
    """ Name of prompt in promptsource to use. if defined, will overwrite doc_to_text, doc_to_target, and doc_to_choice."""
    description: str | None = None
    """An optional prepended Jinja2 template or string which will be prepended to the few-shot examples passed into the model, often describing the task or providing instructions to a model, such as "The following are questions (with answers) about {{subject}}.\n\n". No delimiters or spacing are inserted between the description and the first few-shot example."""
    target_delimiter: str | None = None
    """String to insert between input and target output for the datapoint being tested. defaults to " " """
    fewshot_delimiter: str | None = None
    """ String to insert between few-shot examples. defaults to "\\n\\n" """
    doc_to_text: str | None = None
    """Jinja2 template string to process a sample into the appropriate input for the model."""
    doct_to_target: str | None = None
    """Jinja2 template string to process a sample into the appropriate target for the model."""
    doc_to_choice: str | None = None
    """Jinja2 template string to process a sample into a list of possible string choices for multiple_choice tasks. """

    def to_dict(self):
        base_dict = dataclasses.asdict(self)
        return {k: v for k, v in base_dict.items() if v is not None}


@dataclass(frozen=True)
class LmEvalHarnessConfig:
    task_spec: list[TaskConfig | str]
    max_examples: int | None = None
    max_eval_length: int | None = None
    log_samples: bool = False
    bootstrap_iters: int = 0
    apply_chat_template: bool = False

    def to_task_spec(self) -> list[str | dict]:
        return [task.to_dict() if isinstance(task, TaskConfig) else task for task in self.task_spec]

    def to_task_dict(self) -> dict:
        """
        Convert the task spec to a dictionary that the LM Eval Harness expects.

        This is a bit more complex than we'd like, because we want to run e.g. Hellaswag 0-shot and 10-shot in the same
        run, and LM Eval Harness doesn't seem to want to do that by default. So we need to do some hacky stuff to make
        it work.
        """
        logger.info("Loading tasks...")
        import lm_eval.tasks as tasks

        manager = tasks.TaskManager()
        # we need to do it this way b/c i can't figure out how to run e.g. hellaswag 0 shot and 10 shot in a single run
        this_tasks = {}
        for task in tqdm(self.to_task_spec()):
            try:
                if isinstance(task, str):
                    this_tasks.update(tasks.get_task_dict(task, manager))
                else:
                    our_name = task.get("task_alias", task["task"]) if isinstance(task, dict) else task
                    our_name = our_name.replace(" ", "_")
                    tasks_for_this_task_spec = self._get_task_and_rename(manager, our_name, task)
                    for k, v in tasks_for_this_task_spec.items():
                        if k in this_tasks:
                            raise ValueError(f"Task {k} already exists")
                        this_tasks[k] = v
            except Exception as e:
                logger.exception(f"Failed to load task {task}")
                raise ValueError(f"Failed to load task {task}") from e

        logger.info(f"Loaded {len(this_tasks)} tasks")
        return this_tasks

    def _get_task_and_rename(self, manager, our_name, task: dict | str):
        """
        Get a task from the task manager and rename it to our_name.
        LM Eval Harness doesn't seem to want to run multiple instances of the same task with different fewshot settings,
        (or other differences) so we need to hack around that.
        """
        import lm_eval.tasks as tasks

        task_name = task if isinstance(task, str) else task["task"]

        task_dict = tasks.get_task_dict([task], manager)
        assert len(task_dict) == 1, f"Expected 1 task, got {len(task_dict)}"
        try:
            this_task = self._rename_tasks_for_eval_harness(task_dict, task_name, our_name)
        except AttributeError:
            logger.exception(f"Failed to rename task {task}: {task_dict}")
            raise ValueError(f"Failed to rename task {task}: {task_dict}")
        return this_task

    def _rename_tasks_for_eval_harness(self, this_task, lm_eval_task_name, our_name):
        import lm_eval.tasks as tasks

        # hacky, but this allows us to run multiple instances of the same task with different fewshot settings
        if isinstance(this_task, dict):
            out = {}
            for k, v in this_task.items():
                v = self._rename_tasks_for_eval_harness(v, lm_eval_task_name, our_name)

                if isinstance(k, tasks.ConfigurableGroup):
                    k._config.group = self._replace_name_with_our_name(k.group, lm_eval_task_name, our_name)
                    out[k] = v
                elif isinstance(k, str):
                    k = self._replace_name_with_our_name(k, lm_eval_task_name, our_name)
                    if isinstance(v, dict):
                        subtask_list = self._get_child_tasks(v)
                        # ok so inexplicably, lm_eval_harness doesn't wrap the key in a ConfigurableGroup when you pass
                        # in a task dict (it seems like a mistake), so we need to do that here
                        # subtask is the name of all of the child tasks in v
                        group = tasks.ConfigurableGroup(config={"group": k, "task": subtask_list})
                        out[group] = v
                    else:
                        out[k] = v
                else:
                    raise ValueError(f"Unknown key type: {k}")

            return out

        elif isinstance(this_task, tasks.ConfigurableTask):
            this_task.config.task = self._replace_name_with_our_name(
                this_task.config.task, lm_eval_task_name, our_name
            )
            return this_task
        else:
            raise ValueError(f"Unknown task type: {this_task}")

    def _replace_name_with_our_name(self, lm_eval_name, lm_eval_prefix, our_name_prefix):
        if our_name_prefix.startswith(lm_eval_prefix):
            suffix = our_name_prefix[len(lm_eval_prefix) :]
            prefix = lm_eval_prefix
        else:
            suffix = ""
            prefix = our_name_prefix
        if lm_eval_prefix in lm_eval_name:
            lm_eval_name = lm_eval_name.replace(lm_eval_prefix, prefix) + suffix
        else:
            lm_eval_name = prefix + "_" + lm_eval_name + suffix
        return lm_eval_name

    def _get_child_tasks(self, task_group):
        import lm_eval.tasks as tasks

        out = []
        for k, v in task_group.items():
            if isinstance(k, tasks.ConfigurableGroup):
                subtask_or_tasks = k.config.task
                if isinstance(subtask_or_tasks, str):
                    out.append(subtask_or_tasks)
                else:
                    out.extend(subtask_or_tasks)
            elif isinstance(k, str):
                out.append(k)
            else:
                raise ValueError(f"Unknown key type: {k}")

        return out


@dataclass(frozen=True)
class EvalHarnessMainConfig:
    eval_harness: LmEvalHarnessConfig
    tokenizer: str
    checkpoint_path: str
    checkpoint_is_hf: bool = False
    """If True, the checkpoint is a HuggingFace checkpoint. Otherwise, it is a Levanter checkpoint."""
    apply_chat_template: bool = False
    """
    Whether or not to apply the chat template this model was trained with before running inference
    """
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    model: LmConfig = dataclasses.field(default_factory=Gpt2Config)

    @property
    def EvalBatch(self):
        return self.trainer.EvalBatch

    @cached_property
    def the_tokenizer(self):
        return load_tokenizer(self.tokenizer)


def run_lm_eval_harness(
    config: LmEvalHarnessConfig,
    model,
    tokenizer,
    EvalBatch,
    axis_resources,
    mp: jmp.Policy | None,
) -> dict | None:
    """
    Run the LM Eval Harness on the given model and tasks.

    Returns:
        If running on process 0, returns the outputs of the LM Eval Harness with the following extra keys.
        - "averages": A dictionary with macro and micro averages for all metrics.
        Otherwise, returns None.
    """
    tasks_to_run = config.to_task_dict()

    outputs = _actually_run_eval_harness(config, model, tasks_to_run, tokenizer, EvalBatch, axis_resources, mp)

    return outputs


def _actually_run_eval_harness(
    config: LmEvalHarnessConfig,
    model: LmHeadModel,
    tasks_to_run: dict,
    tokenizer: HfTokenizer,
    EvalBatch: haliax.Axis,
    axis_resources: ResourceMapping,
    mp: jmp.Policy | None,
) -> dict | None:
    """
    Actually run the LM Eval Harness on the given model and tasks. This is a separate function so that it can be used
    by the main function and the callback function.

    Returns:
        The outputs of the LM Eval Harness with the following extra keys:
        - "averages": A dictionary with macro and micro averages for all metrics.

    """
    max_examples = config.max_examples
    max_eval_length = config.max_eval_length

    EvalPos = model.Pos if max_eval_length is None else model.Pos.resize(max_eval_length)
    num_parameters = levanter.utils.jax_utils.parameter_count(model)
    logger.info(
        f"Evaluating with max eval length {EvalPos.size} and batch size {EvalBatch.size}. There are"
        f" {num_parameters} parameters in the model."
    )
    logger.info("Running eval harness...")

    worker = _LmEvalHarnessWorker(
        EvalBatch,
        EvalPos,
        model,
        axis_resources,
        tokenizer,
        mp,
        max_packed_segments=64,
        apply_chat_template=config.apply_chat_template,
    )

    if jax.process_index() == 0:
        logger.info("Process 0 is running the eval harness.")
        harness = worker.make_harness_lm()

        # eval_harness only sets seeds in simple_evaluate, which we can't use (I think?)
        tasks_to_run = _adjust_config(tasks_to_run, 0)
        with set_global_rng_seeds(0):
            outputs = evaluator.evaluate(
                harness,
                tasks_to_run,
                limit=max_examples,
                log_samples=config.log_samples,
                bootstrap_iters=config.bootstrap_iters,
            )
        worker.stop()

        averages = _compute_averages(outputs)
        outputs["averages"] = averages

        return outputs
    else:
        logger.info(f"Process {jax.process_index()} is waiting for eval harness requests from process 0.")
        worker.worker_message_loop()

        logger.info("Finished running eval harness.")

        return None


def _compute_averages(outputs):
    """
    Compute macro and micro averages of all metrics.

    Args:
        outputs: Dictionary with results and samples:
                 - "results": Dictionary of task-level results.
                 - "n-samples" : Dictionary of task-level sample counts.



    Returns:
        Averages dictionary with macro and micro averages for all metrics.
    """
    averages = {}
    metric_keys = set()

    # Collect all possible metrics across tasks
    for task_results in outputs["results"].values():
        metric_keys.update(k for k in task_results.keys() if "stderr" not in k and k != "alias")

    # Compute macro and micro averages
    for metric in metric_keys:
        # Collect valid tasks for this metric
        # We iterate over the n-samples because real tasks (as opposed to aggregates like "mmlu") have counts
        valid_tasks = [
            (outputs["results"][task_name].get(metric), outputs["n-samples"][task_name]["effective"])
            for task_name in outputs["n-samples"]
            if outputs["results"][task_name].get(metric, None) is not None
        ]

        if not valid_tasks:
            continue  # Skip metrics with no valid tasks

        # Separate metric values and weights
        metric_values, this_examples_per_task = zip(*valid_tasks)

        # Compute macro and micro averages
        averages["macro_avg_" + metric] = np.mean(metric_values)
        if sum(this_examples_per_task) > 0:
            averages["micro_avg_" + metric] = np.average(metric_values, weights=this_examples_per_task)

    return averages


def run_eval_harness_main(config: EvalHarnessMainConfig):
    config.trainer.initialize()
    tokenizer = config.the_tokenizer

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(hax.Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        model: LmHeadModel

        # initialize the model
        if config.checkpoint_is_hf:
            model_config = config.model
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=config.checkpoint_path, tokenizer=tokenizer)
            model = converter.load_pretrained(
                model_config.model_type, ref=config.checkpoint_path, dtype=config.trainer.mp.compute_dtype  # type: ignore
            )
        else:
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                model = load_checkpoint(model, config.checkpoint_path, subpath="model")
            model = hax.shard(model, parameter_axis_mapping)

        model = typing.cast(LmHeadModel, inference_mode(model, True))

        logger.info("Running LM eval harness....")
        outputs = run_lm_eval_harness(
            config.eval_harness,
            model,
            tokenizer,
            config.EvalBatch,
            axis_resources=compute_axis_mapping,
            mp=config.trainer.mp,
        )

        logger.info("Finished running LM eval harness")

        # log the results
        if jax.process_index() == 0:
            logger.info("Logging results to tracker")
            assert outputs is not None
            log_report_to_tracker("lm_eval", outputs, levanter.tracker.current_tracker())
            logger.info("Finished logging results to tracker")

            # log the results as json
            logger.info("uploading artifacts...")
            with open("lm_eval_harness_results.json", "w") as f:
                json.dump(outputs, f, indent=2)
                f.flush()
                f_path = f.name
                levanter.tracker.current_tracker().log_artifact(f_path, name="lm_eval_harness_results")

            print(json.dumps(outputs, indent=2), flush=True)

        return outputs


def log_report_to_tracker(prefix: str, report: dict, tracker: Optional[levanter.tracker.Tracker] = None):
    if tracker is None:
        tracker = levanter.tracker.current_tracker()

    to_log = {}
    for task_name, task_results in report["results"].items():
        for metric_name, metric_value in task_results.items():
            # remove the ",none" suffix, which eval-harness adds by default for some reason
            if metric_name.endswith(",none"):
                metric_name = metric_name[:-5]

            if isinstance(metric_value, float | int):
                to_log[f"{prefix}/{task_name}/{metric_name}"] = metric_value

    if "averages" in report:
        for metric_name, metric_value in report["averages"].items():
            if isinstance(metric_value, float | int):
                if metric_name.endswith(",none"):
                    metric_name = metric_name[:-5]

                to_log[f"{prefix}/averages/{metric_name}"] = metric_value

    tracker.log(to_log, step=None)


def lm_eval_harness(config: LmEvalHarnessConfig, tokenizer, EvalBatch, axis_resources, mp: jmp.Policy | None):
    tasks_to_run = config.to_task_dict()

    def lm_eval_harness(step: StepInfo, force=False):
        if step.step == 0 and not force:
            return

        model = step.eval_model
        logger.info("Running eval harness...")
        outputs = _actually_run_eval_harness(
            config,
            model,
            tasks_to_run,
            tokenizer,
            EvalBatch,
            axis_resources,
            mp,
        )
        logger.info("Finished running eval harness.")

        if jax.process_index() == 0:
            assert outputs is not None
            log_report_to_tracker("lm_eval", outputs, levanter.tracker.current_tracker())
            logger.info("Logged report to tracker")

            # don't delete b/c wandb will sometimes defer upload
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
                import json

                json.dump(outputs, f)
                f.flush()
                levanter.tracker.current_tracker().log_artifact(
                    f.name, name=f"lm_eval_harness_results.{step.step}.json", type="lm_eval_output"
                )
                logger.info("Uploaded results to tracker")

    return lm_eval_harness


# lifted from lm-eval simple_evaluate
def _adjust_config(task_dict, fewshot_random_seed=0):
    adjusted_task_dict = {}
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            adjusted_task_dict = {
                **adjusted_task_dict,
                **{task_name: _adjust_config(task_obj, fewshot_random_seed=fewshot_random_seed)},
            }

        else:
            # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
            task_obj.set_fewshot_seed(seed=fewshot_random_seed)

            adjusted_task_dict[task_name] = task_obj

    return adjusted_task_dict


def _iterate_tokenized_requests(
    requests: list[Instance], tokenizer: HfTokenizer, max_len: int, batch_size: int, apply_chat_template: bool = False
) -> Iterator[PromptCompletion]:
    """
    Tokenize the requests and yield them as PromptCompletions, for packing into LmExamples.
    """
    if apply_chat_template:
        contexts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": request.args[0]}], tokenize=False, add_generation_prompt=True
            )
            for request in requests
        ]
    else:
        contexts = [request.args[0] for request in requests]

    completions = [request.args[1] for request in requests]

    # Combine contexts and completions for full tokenization
    combined_texts = [context + completion for context, completion in zip(contexts, completions)]

    # Batch tokenization for combined and context separately
    for batch_indices in batched(range(len(requests)), batch_size):
        # Extract batch data
        combined_batch = [combined_texts[i] for i in batch_indices]
        context_batch = [contexts[i] for i in batch_indices]
        # Tokenize batched inputs
        combined_encodings = tokenizer(combined_batch, truncation=False, padding=False)
        context_encodings = tokenizer(context_batch, truncation=False, padding=False)

        for off in range(len(batch_indices)):
            i = batch_indices[off]
            context_enc = context_encodings["input_ids"][off]
            all_enc = combined_encodings["input_ids"][off]

            context_enc_len = len(context_enc)

            if len(all_enc) > max_len:
                logger.warning(f"Request {i} is too long. Truncating.")
                # Truncate from the left
                context_enc_len = len(context_enc) - (len(all_enc) - max_len)
                all_enc = all_enc[-max_len:]
                if context_enc_len < 0:
                    context_enc_len = 0
                    logger.warning("Prompt length is negative after truncation. Setting to 0.")
            yield PromptCompletion(ids=all_enc, prompt_length=context_enc_len, segment_id=i)


def _pack_requests(
    requests: list[Instance],
    tokenizer: HfTokenizer,
    Pos: hax.Axis,
    max_pack_size: int,
    apply_chat_template: bool = False,
) -> list[LmExample]:
    packed_iterator = _iterate_tokenized_requests(
        requests, tokenizer, Pos.size, batch_size=128, apply_chat_template=apply_chat_template
    )
    # TODO: use a better packing algorithm?
    return greedy_pack_prompt_completions(
        Pos,
        packed_iterator,
        max_segments_per_example=max_pack_size,
        pad_token=tokenizer.pad_token_id,
    )


def _make_dummy_batch(EvalBatch, EvalPos):
    dummy_batch = hax.vmap(LmExample.causal, EvalBatch)(
        hax.zeros(EvalPos, dtype=jnp.int32),
        loss_mask=hax.zeros(EvalPos, dtype=jnp.int32),
        segment_ids=hax.zeros(EvalPos, dtype=jnp.int32),
    )
    out = hax.shard(dummy_batch, {})
    return out


if __name__ == "__main__":
    levanter.config.main(run_eval_harness_main)()
    print("Done", flush=True)
