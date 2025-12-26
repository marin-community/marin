# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import gc
import logging
import jax
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Union

import equinox as eqx
import fsspec
import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning
from jax.experimental import multihost_utils

import levanter
import levanter.callbacks
import levanter.eval
import levanter.eval_harness
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCompatConfig, save_hf_checkpoint_callback
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfig, UrlSingleDatasetLMConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel, compute_next_token_loss
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils import fsspec_utils
from levanter.utils.jax_utils import multihost_broadcast_sync, parameter_count

logger = logging.getLogger(__name__)
logging.getLogger("jax._src.compiler").setLevel(logging.ERROR)


def print_host_memory_usage() -> None:
    """Prints the host's current memory usage using the 'free -h' command."""
    try:
        print("--- Host Memory Availability ---")
        result = subprocess.run(["free", "-h"], capture_output=True, text=True, check=True)
        print(result.stdout)
        print("------------------------------")
    except FileNotFoundError:
        print("Error: The 'free' command was not found. This code is intended for Linux systems.")
    except subprocess.CalledProcessError as exc:
        print(f"Error executing command: {exc.stderr}")


def _to_host_numpy(x):
    """Converts possibly sharded global jax.Array to a host-local numpy array."""
    if isinstance(x, jax.Array):
        # Fast path if already fully addressable on this process
        if getattr(x, "is_fully_addressable", False):
            return np.asarray(jax.device_get(x))
        # Non-addressable global array: return None to signal shard-wise saving
        return None
    return np.asarray(x)


def _save_array_tpu_safe(out_dir: str, filename: str, array, gather_to_single_file: bool = False) -> None:
    """Saves arrays in a way that works on multi-host TPU."""
    if array is None:
        return

    # Extract underlying array from NamedArray if needed
    if hasattr(array, "array") and hasattr(array, "axes"):
        array = array.array

    is_jax_array = isinstance(array, jax.Array)
    is_fully_addressable = False

    if is_jax_array:
        try:
            is_fully_addressable = array.is_fully_addressable
        except AttributeError:
            is_fully_addressable = False

    # Non-jax arrays: save from rank 0 only
    if not is_jax_array:
        if jax.process_index() != 0:
            return
        array_np = np.asarray(array)
        out_path = fsspec_utils.join_path(out_dir, filename)
        fs, plain_path = fsspec.core.url_to_fs(out_path)
        with fs.open(plain_path, "wb") as f:
            np.save(f, array_np)
        return

    # Fully addressable jax.Array: save from rank 0 only
    if is_fully_addressable:
        if jax.process_index() != 0:
            return
        array_np = _to_host_numpy(array)
        out_path = fsspec_utils.join_path(out_dir, filename)
        fs, plain_path = fsspec.core.url_to_fs(out_path)
        with fs.open(plain_path, "wb") as f:
            np.save(f, array_np)
        return

    # Global non-addressable jax.Array
    if gather_to_single_file:
        gathered = multihost_utils.process_allgather(array, tiled=True)
        if jax.process_index() == 0:
            array_np = np.asarray(gathered)
            out_path = fsspec_utils.join_path(out_dir, filename)
            fs, plain_path = fsspec.core.url_to_fs(out_path)
            with fs.open(plain_path, "wb") as f:
                np.save(f, array_np)
    else:
        local_full = np.zeros(array.shape, dtype=array.dtype)
        for shard in array.addressable_shards:
            local_full[shard.index] = np.asarray(jax.device_get(shard.data))

        shard_filename = f"{filename}.r{jax.process_index()}"
        out_path = fsspec_utils.join_path(out_dir, shard_filename)
        fs, plain_path = fsspec.core.url_to_fs(out_path)
        with fs.open(plain_path, "wb") as f:
            np.save(f, local_full)

@dataclass
class TrainLmConfig:
    data: Union[SingleDatasetLMConfig, LMMixtureDatasetConfig] = field(default_factory=UrlSingleDatasetLMConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=Gpt2Config)
    train_seq_len: int | None = None
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    """if provided, this will override the model config in the config. if true, use the default hf checkpoint for this model class"""
    use_hf_model_config: bool = False  # if true, replace the model config with the hf config from the checkpoint

    # TODO: atm we don't support loading from a checkpoint that has a different tokenizer. this is a bit annoying
    # TODO: atm you have to at least specify a levanter model config with the same type as the hf checkpoint

    z_loss_weight: float = 0.0

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000
    hf_save_dtype: Optional[str] = None

    data_seed: Optional[int] = None  # if provided, will override the data seed from the trainer
    initialize_from_checkpoint_path: Optional[str] = None
    """
    If provided, will initialize from this checkpoint, used for llama style ablation. This resets the data loader.
    Note that this differs from --trainer.initialize_from, which does not reset the data loader.
    """
    epoch: int = 0
    eval_harness: Optional[LmEvalHarnessConfig] = None
    eval_harness_steps: int = 10000

    # TODO: really need to add callback framework
    log_entropy: bool = False

    out_dir: str = 'out_dir'
    cfx_seed: int = None
    drop_rate: float = 0.05
    train_only: bool = False

    load_debug_weights: bool = False

    # If True, save per-step input_ids (tokens), dataset_id, and index for reproducibility
    save_input_ids: bool = False

    # Record-only mode: iterate data loaders and save batch metadata without training
    record_only: bool = False
    # If True, gather shards and save a single file per step (rank 0 writes)
    record_gather_single: bool = True
    # Optional cap on number of train steps to record; defaults to full schedule
    record_max_steps: Optional[int] = None


def main(config: TrainLmConfig):
    print_host_memory_usage()

    # Print JAX/JAXLIB versions early for cluster log introspection
    try:
        import jaxlib  # type: ignore
        try:
            jaxlib_version = getattr(jaxlib, "__version__", None)
            if jaxlib_version is None:
                from jaxlib import version as jaxlib_version_mod  # type: ignore
                jaxlib_version = getattr(jaxlib_version_mod, "__version__", "unknown")
        except Exception:
            jaxlib_version = "unknown"
    except Exception:
        jaxlib_version = "unknown"

    print(f"JAX version: {jax.__version__}", flush=True)
    print(f"jaxlib version: {jaxlib_version}", flush=True)

    tokenizer = config.data.the_tokenizer

    # print the special tokens
    print(tokenizer)
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"eos token: {tokenizer.eos_token}")
    print(f"eos token id: {tokenizer.eos_token_id}")
    print(f"pad token: {tokenizer.pad_token}")
    print(f"pad token id: {tokenizer.pad_token_id}")
    print(f"unk token: {tokenizer.unk_token}")
    print(f"unk token id: {tokenizer.unk_token_id}")
    print(f"bos token: {tokenizer.bos_token}")

    # this is some unpleasant code to allow us to initialize from a hf checkpoint. If this is your first read through,
    # I recommend skipping it for now
    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.hf_checkpoint_converter()
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")

        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        if config.use_hf_model_config:
            # TODO: log diff of old and new config
            # NB: gross mutability
            config.model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(config.model, HFCompatConfig):
        converter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    print('Z-loss weight: ', config.z_loss_weight, flush=True)
    loss_function = functools.partial(compute_next_token_loss, logsumexp_weight=config.z_loss_weight)

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        compute_axis_mapping = trainer.compute_axis_mapping
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # some axes we need
        EvalBatch = config.trainer.EvalBatch
        model_max_seq_len = config.model.max_seq_len
        train_length = config.train_seq_len
        if train_length is None:
            train_length = model_max_seq_len

        if train_length <= 0:
            raise ValueError(f"train_length must be positive, got {train_length}")

        if train_length > model_max_seq_len:
            raise ValueError(f"train_length ({train_length}) cannot exceed model max_seq_len ({model_max_seq_len}).")

        if train_length != model_max_seq_len:
            logger.info(f"Training with sequence length {train_length} (model supports {model_max_seq_len}).")

        Pos = config.model.max_Pos.resize(train_length)

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # Get the training dataset
        train_dataset = config.data.train_set(
            Pos,
            config.trainer.batch_schedule,
            key=data_key,
            epochs=config.epoch,
        )

        # Get the tagged evaluation datasets
        tagged_eval_datasets = config.data.tagged_eval_sets(Pos)

        # If requested, run in record-only mode: dump train batch metadata and reward-val tokens, then exit.
        if getattr(config, "record_only", False):
            logger.info("[RecordOnly] Recording train inputs and reward validation tokens; skipping training.")

            train_loader_for_record = trainer.data_loader(train_dataset)
            train_iter = train_loader_for_record.iter_from_step(0)
            max_steps_to_record = int(config.record_max_steps or config.trainer.num_train_steps)

            inputs_dir = os.path.join(config.out_dir, "inputs")
            fsspec_utils.mkdirs(inputs_dir)

            _all_step_tokens = []

            for step_idx, ex in zip(range(max_steps_to_record), train_iter):
                step_int = int(step_idx)
                tokens_current = getattr(ex.tokens, "array", ex.tokens)
                _save_array_tpu_safe(
                    inputs_dir,
                    f"step-{step_int:06d}.tokens.npy",
                    tokens_current,
                    gather_to_single_file=bool(config.record_gather_single),
                )
                _all_step_tokens.append(tokens_current)
                if hasattr(ex, "dataset_id"):
                    _save_array_tpu_safe(
                        inputs_dir,
                        f"step-{step_int:06d}.dataset_id.npy",
                        getattr(ex.dataset_id, "array", ex.dataset_id),
                        gather_to_single_file=bool(config.record_gather_single),
                    )
                if hasattr(ex, "index"):
                    _save_array_tpu_safe(
                        inputs_dir,
                        f"step-{step_int:06d}.index.npy",
                        getattr(ex.index, "array", ex.index),
                        gather_to_single_file=bool(config.record_gather_single),
                    )
                _ = multihost_broadcast_sync(1)

            try:
                if len(_all_step_tokens) > 0:
                    train_tokens_all = jnp.stack(_all_step_tokens, axis=0)
                    _save_array_tpu_safe(
                        config.out_dir,
                        "train_tokens_all.npy",
                        train_tokens_all,
                        gather_to_single_file=bool(config.record_gather_single),
                    )
                    _ = multihost_broadcast_sync(1)
            except Exception:
                logger.exception("Failed to save concatenated train tokens during record-only mode")

            try:
                val_sets = config.data.validation_sets(Pos)
                if len(val_sets) > 0:
                    _last_name, last_dataset = list(val_sets.items())[-1]
                    val_loader2 = trainer.data_loader(last_dataset, trainer.EvalBatch)
                    val_iter = val_loader2.iter_from_step(0)
                    val_tokens_list = []
                    for ex in val_iter:
                        tokens_arr = getattr(ex.tokens, "array", ex.tokens)
                        val_tokens_list.append(tokens_arr)
                    if len(val_tokens_list) > 0:
                        val_tokens = jnp.concatenate(val_tokens_list, axis=0)
                        _save_array_tpu_safe(
                            config.out_dir,
                            "val_tokens.npy",
                            val_tokens,
                            gather_to_single_file=True,
                        )
                        _ = multihost_broadcast_sync(1)
            except Exception:
                logger.exception("Failed to save validation tokens during record-only mode")

            return

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        if int(state.step) == 0 and config.initialize_from_checkpoint_path is not None:
            print(
                f"*** Initializing model weights from checkpoint {config.initialize_from_checkpoint_path}",
                flush=True,
            )
            # By default, state.step is 0 and we have a fresh model.
            # We load just the model weights from the checkpoint and replace the model in the fresh state.
            # This leaves the step and optimizer state as new.
            model_from_checkpoint = load_checkpoint(
                state.model, config.initialize_from_checkpoint_path, subpath="model"
            )
            state = dataclasses.replace(state, model=model_from_checkpoint)

        if int(state.step) == 0:
            # TODO: I don't love that we init the model twice, but it's not a big deal i think?
            if config.initialize_from_hf:
                # initialize from an hf pretrained model
                assert converter is not None
                logger.info(
                    "No training checkpoint found. Initializing model from HF checkpoint"
                    f" '{converter.reference_checkpoint}'"
                )
                # this is a bit gross, but we want to free up the memory from the model we just built
                state = dataclasses.replace(state, model=None)
                gc.collect()
                model = converter.load_pretrained(
                    config.model.model_type,
                    config=config.model if not config.use_hf_model_config else None,
                    axis_mapping=parameter_axis_mapping,
                    dtype=trainer.mp.compute_dtype,
                )
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        max_eval_examples_per_ds = config.trainer.max_eval_batches
        if max_eval_examples_per_ds is not None:
            max_eval_examples_per_ds *= config.trainer.eval_batch_size

        if len(tagged_eval_datasets) == 0:
            logger.warning("No evaluation datasets provided.")
        else:
            cb = levanter.eval.cb_tagged_lm_evaluate(
                EvalBatch,
                tagged_eval_datasets,
                tokenizer,
                trainer.device_mesh,
                compute_axis_mapping,
                max_eval_examples_per_ds,
                mp=config.trainer.mp,
            )
            trainer.add_hook(cb, every=config.trainer.steps_per_eval)

        flops_per_token = config.model.flops_per_token(vocab_size, Pos.size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.batch_schedule, flops_per_example), every=1
        )
        # trainer.add_hook(callbacks.GradWatchCallback(include_histograms=True), every=5)

        # Periodically upload XLA dumps to W&B if enabled in tracker config.
        tracker_cfg = trainer.config.tracker
        cfgs = tracker_cfg if isinstance(tracker_cfg, tuple) else (tracker_cfg,)
        for c in cfgs:
            if isinstance(c, WandbConfig) and c.save_xla_dumps:
                freq = getattr(config.trainer, "steps_per_eval", None) or 300
                trainer.add_hook(callbacks.wandb_xla_logger(c), every=freq)

        if config.hf_save_path is not None and config.hf_save_steps is not None:
            # bit gross to reach this far into the config, but it's fine
            assert converter is not None, "converter must be set when saving HF checkpoints"
            if config.trainer.checkpointer.append_run_id_to_base_path:
                full_save_path = os.path.join(config.hf_save_path, trainer.run_id)
            else:
                full_save_path = config.hf_save_path

            save_dtype: Optional[jnp.dtype] = None
            if config.hf_save_dtype is not None:
                try:
                    save_dtype = jnp.dtype(config.hf_save_dtype)
                except TypeError:
                    logger.warning(f"Invalid hf_save_dtype: {config.hf_save_dtype}. Defaulting to None.")

            trainer.add_hook(
                save_hf_checkpoint_callback(
                    full_save_path, converter, upload_to_hf=config.hf_upload or False, save_dtype=save_dtype
                ),
                every=config.hf_save_steps,
            )

        if config.eval_harness is not None:
            try:
                import lm_eval  # type: ignore
                _ = lm_eval
                eval_harness = config.eval_harness
                trainer.add_hook(
                    levanter.eval_harness.lm_eval_harness(
                        eval_harness,
                        tokenizer,
                        EvalBatch,
                        compute_axis_mapping,
                        trainer.mp,
                        checkpoint_base_path=trainer.checkpoint_path,
                    ),
                    every=config.eval_harness_steps,
                )
            except Exception:
                logger.warning(
                    "lm-evaluation-harness is not installed; skipping eval harness hook registration."
                )

        @named_jit(axis_resources=compute_axis_mapping)
        def compute_logits(model: LmHeadModel, example: LmExample):
            model = trainer.mp.cast_to_compute(model)
            activations = model.activations(example.tokens, key=None, attn_mask=example.attn_mask)
            head = model.get_lm_head()
            logits = hax.dot(activations, head, axis=model.Embed)
            return logits

        print("$$$$$ Validation sets:")
        for name, dataset in config.data.validation_sets(Pos).items():
            print(f"> Dataset {name}", dataset)
            val_loader = trainer.data_loader(dataset, trainer.EvalBatch)
            val_loader = val_loader.iter_from_step(0)

        if config.log_entropy:
            for name, dataset in config.data.validation_sets(Pos).items():
                trainer.add_hook(
                    levanter.analysis.cb_compute_entropies(
                        compute_logits,
                        Vocab,
                        dataset,
                        prefix=os.path.join("analysis", name) if name else "analysis",
                        batch_size=EvalBatch.size,
                        mapping=compute_axis_mapping,
                    ),
                    every=config.trainer.steps_per_eval,
                )


        train_loader = trainer.data_loader(train_dataset)


        # Decode and print the first few examples
        print("Decoding first few examples...")

        '''
        for i, example in enumerate(iter(train_loader)):
            if i >= 1: # Print 5 examples
                break

            tokenizer = config.data.the_tokenizer

            # Decode a whole batch at once; rely on skip_special_tokens to drop PAD/EOS
            input_ids = np.asarray(example.tokens.array).astype(int)
            texts = tokenizer.batch_decode(
                input_ids.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for j, text in enumerate(texts):
                print(f"Example {i*input_ids.shape[0] + j}:")
                print(text)
                print("-" * 20)
        import pdb; pdb.set_trace()
        '''

        # Build segment boundaries for replay using metagrad_segment_size S:
        # segment_starts = [0, S, 2*S, ...] up to the max batch number (num_train_steps).
        # Do not use metagrad_checkpoint_frequency for segmentation.
        seg_size = int(getattr(config.trainer, "metagrad_segment_size", 0) or 0)
        num_steps = int(config.trainer.num_train_steps)
        if seg_size > 0:
            segment_starts = list(range(0, num_steps, seg_size))
        else:
            segment_starts = [0]

        print(f"[REPLAY] segment_starts: {segment_starts}", flush=True)

        # Construct a reversed loader that is segment-aware and supports iter_segment
        reversed_train_loader = train_loader.reversed(
            num_train_steps=config.trainer.num_train_steps,
            segment_starts=segment_starts,
        )
        if state.step > 0:
            logger.info(f"Resuming training from step {state.step}")
            train_loader = train_loader.iter_from_step(state.step)
        else:
            train_loader = train_loader.iter_from_step(0)

        ## OK, actually run training!
        #last_info = trainer.train(state, train_loader)
        #data_weight_vector = jnp.ones(len(train_dataset) * trainer.config.batch_size)
        data_weight_vector = jnp.ones(trainer.config.num_train_steps * trainer.config.train_batch_size)

        if config.train_only and config.cfx_seed is not None:
            # randomly set 5% of indices to 0
            data_weight_vector = jax.random.bernoulli(jax.random.PRNGKey(config.cfx_seed), 1-config.drop_rate, data_weight_vector.shape).astype(jnp.float32)
            #data_weight_vector = data_weight_vector.at[:1024*40].set(1.0)
        print(f"data_weight_vector: {data_weight_vector[:100]}")
        # save data_weight_vector to out_dir (TPU-safe)
        if True:
            out_dir = config.out_dir
            fsspec_utils.mkdirs(out_dir)
            _save_array_tpu_safe(out_dir, 'data_weight_vector.npy', data_weight_vector)

        # Initialize global loss_masks indicator array (1D boolean array)
        # Each entry is 1 if the example has any non-zero loss_weight, 0 otherwise
        # Create as a sharded array along batch dimension to match training data sharding
        batch_size = trainer.config.train_batch_size
        total_examples = trainer.config.num_train_steps * batch_size

        # Create sharded array: shard along the batch dimension (axis 0)
        # This matches how training batches are sharded across devices
        from jax.sharding import NamedSharding, PartitionSpec

        # Get the current mesh and create sharding spec
        mesh = trainer.device_mesh
        # Map batch axis to physical "data" axis in the mesh
        # The mesh typically has axes like ("data", "model")
        sharding = NamedSharding(mesh, PartitionSpec("data"))

        # Create the array with explicit sharding
        loss_masks_global = jax.device_put(jnp.zeros(total_examples, dtype=jnp.int32), sharding)

        logger.info(f"[LossMask] Initialized global loss_masks indicator array with shape: {loss_masks_global.shape}, sharding: {loss_masks_global.sharding}")

        # If an eval harness is specified, build a reward loader from its tasks (e.g., arc_easy)
        reward_loader = val_loader
        used_lm_eval_reward = False
        logger.info("[RewardLoader] eval_harness configured: %s", config.eval_harness is not None)
        #try:
        if False: #config.eval_harness is not None:
            from levanter.eval_harness import build_reward_loader_for_tasks
            max_eval_length = config.eval_harness.max_length
            EvalPos = state.model.Pos if max_eval_length is None else state.model.Pos.resize(max_eval_length)
            logger.info(
                "\n\n\n\n\n[RewardLoader] Using LM-Eval tasks for reward | max_eval_length=%s",
                max_eval_length,
            )
            reward_loader = build_reward_loader_for_tasks(
                config.eval_harness,
                tokenizer,
                trainer.EvalBatch,
                EvalPos,
            )
            used_lm_eval_reward = True
        else:
            logger.info("[RewardLoader] Using validation loader for reward (no eval_harness configured)")

        #except Exception as e:
        #logger.warning(f"\n\n\n\n\n[RewardLoader] Falling back to standard val_loader for reward due to: {e}")
        #logger.info("[RewardLoader] Using validation loader for reward")

        logger.info("[RewardLoader] Final selection: %s", "LM-Eval" if used_lm_eval_reward else "Validation loader")

        # If requested, register a simple hook to save input_ids and indices each step.
        if config.save_input_ids:
            def _save_inputs_hook(info):
                try:
                    ex = getattr(trainer, "_last_batch_example", None)
                    if ex is None:
                        return
                    out_dir = os.path.join(config.out_dir, "inputs")
                    fsspec_utils.mkdirs(out_dir)
                    step_int = int(jax.device_get(info.step)) if hasattr(info.step, "dtype") else int(info.step)

                    # Save tokens as a single file gathered across hosts (rank 0 writes)
                    _save_array_tpu_safe(
                        out_dir,
                        f"step-{step_int:06d}.tokens.npy",
                        getattr(ex.tokens, "array", ex.tokens),
                        gather_to_single_file=True,
                    )
                    if hasattr(ex, "dataset_id"):
                        _save_array_tpu_safe(out_dir, f"step-{step_int:06d}.dataset_id.npy", getattr(ex.dataset_id, "array", ex.dataset_id))
                    if hasattr(ex, "index"):
                        _save_array_tpu_safe(out_dir, f"step-{step_int:06d}.index.npy", getattr(ex.index, "array", ex.index))
                    # Ensure all hosts wait for IO completion to improve consistency on remote filesystems
                    _ = multihost_broadcast_sync(1)
                except Exception:
                    logger.exception("Failed to save input batch for step %s", info.step)

            trainer.add_hook(_save_inputs_hook, every=1)

        ret = trainer.train_and_replay(
            state,
            train_loader,
            reversed_train_loader,
            reward_loader,
            data_weight_vector,
            segment_starts,
            train_only=config.train_only,
            loss_masks_global=loss_masks_global,
        )
        reward, metagrads, dataset_ids_global, local_indices_global, loss_masks_global = ret
        save_success = False
        try:
            out_dir = config.out_dir
            fsspec_utils.mkdirs(out_dir)

            # Save as single files (gathered from all hosts)
            _save_array_tpu_safe(out_dir, 'reward.npy', reward, gather_to_single_file=True)
            _save_array_tpu_safe(out_dir, 'metagrads.npy', metagrads, gather_to_single_file=True)
            #_save_array_tpu_safe(out_dir, 'dataset_ids_global.npy', dataset_ids_global)
            #_save_array_tpu_safe(out_dir, 'local_indices_global.npy', local_indices_global)
            _save_array_tpu_safe(out_dir, 'data_weight_vector.npy', data_weight_vector, gather_to_single_file=True)

            # Save loss_masks_global array as a single file (gathered from all hosts)
            if loss_masks_global is not None:
                logger.info(f"[LossMask] Saving loss_masks_global with shape: {loss_masks_global.shape}")
                _save_array_tpu_safe(out_dir, 'loss_masks_global.npy', loss_masks_global, gather_to_single_file=True)
            else:
                logger.warning(f"[LossMask] No loss_masks_global to save")

            # Additionally, extract and save tokens seen during metagrad reward calculations as a single file
            try:
                val_tokens_list = []
                # Rebuild the validation loader to replay the same evaluation dataset
                val_sets = config.data.validation_sets(Pos)
                if len(val_sets) > 0:
                    # Use the last dataset (matching how reward_loader was selected)
                    last_name, last_dataset = list(val_sets.items())[-1]
                    val_loader2 = trainer.data_loader(last_dataset, trainer.EvalBatch)
                    val_iter = val_loader2.iter_from_step(0)
                    for ex in val_iter:
                        tokens_arr = getattr(ex.tokens, 'array', ex.tokens)
                        val_tokens_list.append(tokens_arr)
                if len(val_tokens_list) > 0:
                    # Concatenate along the batch axis and gather to a single host before saving
                    val_tokens = jnp.concatenate(val_tokens_list, axis=0)
                    _save_array_tpu_safe(out_dir, 'val_tokens.npy', val_tokens, gather_to_single_file=True)
            except Exception:
                logger.exception("Failed to save validation tokens")

            save_success = True
        except Exception:
            logger.exception("Failed to save outputs to %s", config.out_dir)
        # Ensure all hosts wait for IO result (helps remote FS consistency)
        _ = multihost_broadcast_sync(int(save_success))


        # If running EpochDataset save latest checkpoint by default
        if trainer.config.checkpointer is not None and config.epoch > 0:
            trainer.run_hooks(last_info, force=True)
            checkpointer = trainer.config.checkpointer.create(trainer.run_id)
            checkpointer.wait_until_finished()

    # This isn't necessary except when Levanter is run in a subprocess (as happens w/ ray)
    trainer.tracker.finish()
    eqx.clear_caches()


if __name__ == "__main__":
    levanter.config.main(main)()
