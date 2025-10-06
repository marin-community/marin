# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, Optional, Union

import jax.random as jrandom
import transformers
from lenses import lens

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, save_hf_checkpoint_callback
from levanter.data import PermutationDataset
from levanter.data.dataset import AsyncDataset
from levanter.data.loader import stack_batches
from levanter.data.mixture import MixtureDataset, StopStrategy
from levanter.data.packing import PromptCompletion, pack_prompt_completions
from levanter.data.text import (
    ChatUrlDataSourceConfig,
    SupervisedSourceConfig,
    mk_single_turn_cached_sft_dataset,
    mk_supervised_dataset,
)
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmHeadModel, compute_next_token_loss
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.background_iterable import BackgroundIterator
from levanter.utils.hf_utils import HfTokenizer


logger = logging.getLogger(__name__)

# Define default special tokens
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class DatasetType(str, Enum):
    """Type of dataset to use"""

    HUGGINGFACE = "huggingface"  # Use HF dataset
    CHAT_JSONL = "chat_jsonl"  # Use JSONL files with chat format


@dataclass
class SFTMixtureConfig:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    # Config for mixture of supervised datasets
    supervised_data: Dict[str, SupervisedSourceConfig] = field(default_factory=dict)
    mixture_weights: Dict[str, float] = field(default_factory=dict)
    mixture_block_size: int = 2048
    stop_strategy: str = StopStrategy.RESTART_STRATEGY

    # Config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 0

    max_seq_len: int = 2048
    model_name_or_path: str = "marin-community/marin-8b-base"
    tokenizer: str = "marin-community/marin-8b-base"

    # Add dataset type and chat-specific fields
    dataset_type: DatasetType = DatasetType.CHAT_JSONL
    messages_field: str = "messages"
    input_role: str = "user"
    output_role: str = "assistant"

    data_seed: Optional[int] = None

    reinit_tokens: list[str] | bool = False
    """
    if set, will reinitialize the embeddings for the given tokens. If True, will reinitialize the default tokens
    for llama3's tokenizer
    """
    reinit_lm_head: bool = True
    """If reinit_tokens is set, will reinitialize the lm_head for those tokens"""
    reinit_embeddings: bool = True
    """If reinit_tokens is set, will reinitialize the embeddings for those tokens"""


def train(config: SFTMixtureConfig):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.tokenizer,
        model_max_length=config.max_seq_len,
        padding_side="right",
        trust_remote_code=True,
    )
    logger.info(f"Loaded tokenizer {tokenizer}")

    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot use both --initialize_from_hf and --initialize_from")

        assert isinstance(config.model, HFCompatConfig)

        converter = HFCheckpointConverter.from_hf(config.model_name_or_path, trust_remote_code=True)
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")
        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        model_config = converter.default_config
        model_config = dataclasses.replace(converter.default_config, seq_len=config.max_seq_len)
    elif config.trainer.initialize_from is None:
        raise ValueError("Must specify either --initialize_from_hf or --initialize_from")
    else:
        if config.hf_save_steps:
            converter = HFCheckpointConverter.from_hf(config.model_name_or_path, trust_remote_code=True)
            converter = converter.replaced(tokenizer=tokenizer)
        else:
            converter = None
        model_config = config.model

    config = dataclasses.replace(config, model=model_config)
    levanter.initialize(config)

    num_new_tokens = add_special_tokens(tokenizer)
    logger.info(f"Added {num_new_tokens} new tokens")

    seed = config.trainer.seed
    data_key, reinit_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

    if config.data_seed is not None:
        logger.info(f"Overriding data seed with {config.data_seed}")
        data_key = jrandom.PRNGKey(config.data_seed)

    # Create supervised datasets using generic machinery
    logger.info("Creating supervised datasets")

    # Create individual datasets
    train_datasets = {}
    for name, source_config in config.supervised_data.items():
        if config.dataset_type == DatasetType.CHAT_JSONL:
            # Create a ChatUrlDataSourceConfig object
            # All configs in supervised_data should have train_urls attribute at runtime
            # The type checking error occurs because SupervisedHfSourceConfig doesn't have this attribute
            # but in practice, when loaded from YAML, all configs will have it
            train_dataset = mk_single_turn_cached_sft_dataset(
                ChatUrlDataSourceConfig(
                    cache_dir=source_config.cache_dir,
                    # Use getattr to silence type checking, as the attribute always exists at runtime
                    train_urls=getattr(source_config, "train_urls", []),
                    messages_field=config.messages_field,
                    input_role=config.input_role,
                    output_role=config.output_role,
                ),
                tokenizer,
                model_config.Pos,
            )
            train_dataset = PermutationDataset(train_dataset, data_key)
        else:
            train_dataset = mk_supervised_dataset(source_config, "train", tokenizer, model_config.Pos)

        train_datasets[name] = train_dataset

    # Create mixture dataset
    logger.info("Creating mixture dataset")
    train_dataset = MixtureDataset(
        datasets=train_datasets,
        weights=config.mixture_weights,
        block_size=config.mixture_block_size,
        key=data_key,
        stop_strategy=config.stop_strategy,
    )

    logger.info("Creating optimizer")
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    with Trainer(config.trainer, optimizer, loss_fn=compute_next_token_loss) as trainer:  # type: ignore
        parameter_axis_mapping = trainer.parameter_axis_mapping

        Pos = config.model.Pos
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)

        if config.initialize_from_hf:
            logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
            model: LmHeadModel = converter.load_pretrained(
                model_config.model_type, axis_mapping=parameter_axis_mapping, dtype=trainer.mp.param_dtype
            )  # type: ignore
            model = hax.named_jit(lambda m: m.resize_vocab(len(tokenizer)))(model)
            state = trainer.initial_state(training_key, model=model)
        else:
            if vocab_size != Vocab.size:
                logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")
            state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        flops_per_token = config.model.flops_per_token(vocab_size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size, flops_per_example),
            every=1,
        )

        current_step = int(state.step)

        logger.info("Creating prompt completion iterator")
        prompt_completion_iterator = create_prompt_completion_iterator(train_dataset, Pos)

        if current_step > 0:
            logger.info(f"Resuming training from step {current_step}")
            examples_to_skip = current_step * trainer.config.train_batch_size

            for _ in range(examples_to_skip):
                try:
                    next(prompt_completion_iterator)
                except StopIteration:
                    logger.warning("Ran out of examples while seeking - restarting from beginning")
                    prompt_completion_iterator = create_prompt_completion_iterator(train_dataset, Pos)
        else:
            logger.info("Starting SFT from scratch")
            if config.reinit_tokens:
                if config.reinit_tokens is True:
                    # this is hardcoded to llama3 but eh
                    tokens_to_reinit = [
                        "<|finetune_right_pad_id|>",
                        "<|start_header_id|>",
                        "<|end_header_id|>",
                        "<|eom_id|>",
                        "<|eot_id|>",
                    ]
                else:
                    tokens_to_reinit = config.reinit_tokens

                logger.info(f"Reinitializing tokens: {tokens_to_reinit}")

                new_model = reinitialize_some_tokens(
                    state.model,
                    tokenizer,
                    tokens_to_reinit,
                    reinit_key,  # Use the previously unused key
                    donate=True,
                    reinit_lm_head=config.reinit_lm_head,
                    reinit_embeddings=config.reinit_embeddings,
                )
                state = dataclasses.replace(state, model=new_model)
                del new_model

        logger.info("Packing prompt completions")
        packed_iterator = pack_prompt_completions(
            Pos,
            prompt_completion_iterator,
            max_segments_per_example=4,
            pad_token=tokenizer.pad_token_id,
            max_buffered_examples=16,
        )
        logger.info("Stacking batches to train batch")
        packed_iterator = stack_batches(example_iterator=packed_iterator, Pos=Pos, Batch=trainer.TrainBatch)
        logger.info("Creating data loader")
        packed_loader = BackgroundIterator(packed_iterator, max_capacity=1024)

        if config.hf_save_path is not None:
            if config.trainer.checkpointer.append_run_id_to_base_path:
                full_save_path = os.path.join(config.hf_save_path, trainer.run_id)
            else:
                full_save_path = config.hf_save_path

            trainer.add_hook(
                save_hf_checkpoint_callback(full_save_path, converter, upload_to_hf=config.hf_upload or False),
                every=config.hf_save_steps,
            )

        trainer.train(state, packed_loader)


def create_prompt_completion_iterator(cached_dataset: AsyncDataset, Pos: hax.Axis) -> Iterator[PromptCompletion]:
    """Creates an iterator that yields PromptCompletion objects from a cached dataset."""
    # Instead of getting total length upfront, we'll keep track of processed examples
    processed_examples = 0
    batch_size = 4096

    while True:  # Infinite loop for restart strategy
        # Get next batch of indices
        indices = list(range(processed_examples, processed_examples + batch_size))
        try:
            examples = asyncio.run(cached_dataset.get_batch(indices))
        except (IndexError, ValueError):
            # If we hit the end of the dataset, start over
            processed_examples = 0
            continue

        for i in range(len(examples)):
            example = examples[i]
            sources_len = example["sources_len"].item()
            if sources_len > Pos.size - 1:
                continue

            ids = example["input_ids"].tolist()
            if len(ids) > Pos.size:
                ids = ids[: Pos.size]

            if len(ids) <= sources_len:
                continue

            try:
                yield PromptCompletion(ids=ids, prompt_length=sources_len, segment_id=indices[i])
            except ValueError as e:
                logger.error(
                    f"Error creating PromptCompletion (ids length: {len(ids)}, sources_len: {sources_len}, segment id:"
                    f" {indices[i]}): {e}"
                )
                continue

        processed_examples += batch_size


def add_special_tokens(tokenizer, use_unk_instead_of_adding=False):
    special_tokens_dict = dict()
    if use_unk_instead_of_adding:
        if tokenizer.unk_token is None:
            raise ValueError("use_unk_instead_of_add is True but tokenizer doesn't have an unk token")

    unk = tokenizer.unk_token if use_unk_instead_of_adding else None

    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN if not use_unk_instead_of_adding else unk
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN if not use_unk_instead_of_adding else unk
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN if not use_unk_instead_of_adding else unk
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    return tokenizer.add_special_tokens(special_tokens_dict)


def reinitialize_some_tokens(
    model,
    tokenizer: HfTokenizer,
    tokens_to_reinit: list[str],
    key,
    donate=False,
    reinit_lm_head=True,
    reinit_embeddings=True,
):
    """
    So we (Will, specifically) realized that we were in this situation where we never saw SFT tokens during pretraining,
    which meant they had much lower norm than tokens that had been seen.

    https://github.com/marin-community/marin/issues/954
    """
    ids_to_reinit = [tokenizer.convert_tokens_to_ids(token) for token in tokens_to_reinit]
    if len(ids_to_reinit) == 0:
        raise ValueError("No tokens to reinitialize")
    # obnoxiously, convert_tokens_to_ids does not always return None, if an unk token is set it returns that
    # but for gpt2, the unk token is eos token so we can't just check eos token
    elif any(
        token is None or tokenizer.convert_ids_to_tokens(id) != token
        for id, token in zip(ids_to_reinit, tokens_to_reinit)
    ):
        raise ValueError("One or more tokens are not in the tokenizer vocabulary")

    @hax.named_jit(donate_args=(donate,))
    def _reinit_tokens(model):
        Embed = model.embeddings.Embed
        new_Vocab = model.Vocab.resize(len(ids_to_reinit))

        emb_key, lm_key = jrandom.split(key, 2)

        embeddings_matrix = model.embeddings.token_embeddings.weight

        if reinit_embeddings:
            new_embeddings = _reinit_embed_vectors(Embed, new_Vocab, embeddings_matrix, ids_to_reinit, emb_key)
            model = lens.embeddings.token_embeddings.weight.set(new_embeddings)(model)

        if reinit_lm_head:
            new_lm_head = _reinit_embed_vectors(Embed, new_Vocab, model.lm_head.weight, ids_to_reinit, lm_key)
            model = lens.lm_head.weight.set(new_lm_head)(model)

        return model

    return _reinit_tokens(model)


def _reinit_embed_vectors(Embed, new_Vocab, embeddings_matrix, ids_to_reinit, key):
    # reinit with same mean and std as the embeddings. Technically this should be the non-reinit tokens
    # but whatever
    mu = hax.mean(embeddings_matrix, axis="vocab")
    std = hax.std(embeddings_matrix, axis="vocab")
    reinited = hax.random.truncated_normal(key, (new_Vocab, Embed), -3, 3) * std + mu
    new_weight = embeddings_matrix.at["vocab", ids_to_reinit].set(reinited)
    return new_weight


if __name__ == "__main__":
    levanter.config.main(train)()
