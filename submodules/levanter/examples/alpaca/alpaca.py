# levanter version of https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import fsspec
import jax
import jax.random as jrandom
import numpy as np
import transformers

import haliax as hax

import levanter
from levanter.compat.hf_checkpoints import HFCheckpointConverter, save_hf_checkpoint_callback
from levanter.data import PermutationDataset
from levanter.models.lm_model import LmExample, LmHeadModel, compute_next_token_loss
from levanter.optim import OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils import fsspec_utils
from levanter.utils.hf_utils import num_cpus_used_by_tokenizer
from levanter.utils.py_utils import non_caching_cycle


# Differences:
# - We use the huggingface dataset version of alpaca rather than checking it in
# - Levanter doesn't (currently) do epochs, just steps.
# - We use Levanter's distributed preprocessing, which is a bit overkill for this dataset but is a good example.
#   (The original's preprocessing is very slow, which is usually fine, but not good for preemptible nodes.)
# - We use fast tokenizers. I don't know why the original code doesn't use them.
# - We produce Levanter's LmExample class instead of a dict, and loss masks are used instead of the -100 sentinel value.

# Ways this script could be improved:
# * Could tune hparams more for throughput

# Original
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# TODO
# * make a pad_stack function that can pad and stack arrays in one go
# * make batch loader support pad_stack


logger = logging.getLogger(__name__)

# copy paste from alpaca

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class TrainArgs:
    optimizer: OptimizerConfig
    trainer: TrainerConfig

    max_tune_length: int = 2048  # maximum length of the input to the model during tuning
    data: str = "tatsu-lab/alpaca"  # Path to the training data, or huggingface dataset name.
    data_cache_dir: str = "cache/"  # Path to cache the tokenized data. can be gcs
    prompts: Optional[Dict[str, str] | str] = None  # Path to the prompts file or a dict of prompts. can be gcs
    mask_inputs: bool = True  # if True, mask out the input and prompt for loss calculation

    model_name_or_path: str = "NousResearch/Llama-2-7b-hf"
    trust_remote_code: bool = False  # Trust remote code when loading from HuggingFace checkpoints.

    model_cache_dir: Optional[str] = None  # Path to cache the model. must be local.

    hf_save_path: Optional[str] = "alpaca_hf_ckpts"  # Path to save the HuggingFace checkpoint, can be gcs
    hf_upload: Union[bool, str] = False  # Name of the HuggingFace repo to upload to (if any).
    hf_save_steps: int = 1000  # How often to save the HuggingFace checkpoint.


def _get_data_source(path_or_id):
    """The original alpaca.py used a json file, but it's since been moved to the HF dataset hub. You can use any
    dataset that's compatible with the structure of the alpaca dataset."""
    if fsspec_utils.exists(path_or_id):
        # we're a bit generous here b/c we support compression
        if ".jsonl" in path_or_id:
            return levanter.data.datasource_from_jsonl([path_or_id])
        elif ".json" in path_or_id:
            return levanter.data.datasource_from_json([path_or_id])
        else:
            raise ValueError(
                f"We only support HF Datasets or a data file with .json or .jsonl extensions, not {path_or_id}!"
            )
    else:
        return levanter.data.datasource_from_hf(path_or_id, split="train")


def mk_dataset(config: TrainArgs, tokenizer: transformers.PreTrainedTokenizerBase):
    dataset = _get_data_source(config.data)

    prompts = get_prompts(config.prompts)

    def preprocess(batch):
        def format_example(ex):
            if ex.get("input", "") == "":
                return prompts["prompt_no_input"].format_map(ex)
            else:
                return prompts["prompt_input"].format_map(ex)

        sources = [format_example(example) for example in batch]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in batch]
        # TODO: this seems pretty wasteful since you end up tokenizing twice, but it's how the original code does it.
        examples = [s + t for s, t in zip(sources, targets)]
        sources_tokenized = tokenizer(sources, return_tensors="np", padding=False, truncation=True)
        examples_tokenized = tokenizer(examples, return_tensors="np", padding=False, truncation=True)

        source_lens = [len(s) for s in sources_tokenized["input_ids"]]

        return {
            "input_ids": examples_tokenized["input_ids"],
            "source_lens": source_lens,
        }

    dataset = dataset.map_batches(preprocess, batch_size=128, num_cpus=num_cpus_used_by_tokenizer(tokenizer))  # type: ignore
    dataset = dataset.build_or_load_cache(config.data_cache_dir, await_finished=True)  # type: ignore

    def _prepare_example(ex: dict) -> LmExample:
        """
        Prepare an example for training. This function converts the (cached) batch encoding into an LmExample.

        It goes through the following steps:

        1. Pad the batch to the maximum length.
        2. Mask out the input and prompt if requested.
        3. Create an LmExample with the input_ids as the input and the next token as the target.
        """
        # annoyingly, pad expects things to be batched so we have to prepend a batch axis
        ex = tokenizer.pad({k: np.expand_dims(v, 0) for k, v in ex.items()}, return_tensors="np", padding="max_length")
        ex = {k: v[0] for k, v in ex.items()}
        input_ids = hax.named(ex["input_ids"], "position")
        # mask out padding and anything before the start of the target
        Pos = input_ids.resolve_axis("position")
        if config.mask_inputs:
            loss_mask = hax.arange(Pos) >= ex["source_lens"] - 1  # should be minus 1?

            # don't predict the padding
            targets = hax.roll(input_ids, -1, Pos)
            loss_mask = loss_mask & (targets != tokenizer.pad_token_id)
            # to not predict EOS token since we don't have target!
            loss_mask = loss_mask & (1 - hax.nn.one_hot(-1, Pos, dtype=jax.numpy.bool_))
        else:
            loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jax.numpy.float32)
        lm_ex = LmExample.causal(input_ids, loss_mask=loss_mask, eos_id=tokenizer.eos_token_id)
        return lm_ex

    return dataset.map(_prepare_example)


def get_prompts(prompt_path) -> dict:
    prompts = DEFAULT_PROMPT_DICT
    if isinstance(prompt_path, dict):
        prompts = prompt_path
    elif prompt_path is not None:
        with fsspec.open(prompt_path) as f:
            prompts = json.load(f)
    return prompts


def train(config: TrainArgs):
    levanter.initialize(config)

    # Since Levanter has different implementations of models from HF, we need to convert the HF checkpoint.
    # This class is a wrapper around the HF checkpoint converter that also downloads the checkpoint if necessary.
    converter = HFCheckpointConverter.from_hf(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    model_config = converter.default_config

    if config.max_tune_length > model_config.Pos.size:
        logger.warning(
            f"max_tune_length ({config.max_tune_length}) is greater than the model's maximum length"
            f" ({model_config.Pos.size}). "
        )

    # Randomness in JAX is tightly controlled. We pass around a key that is used to generate random numbers.
    training_key, data_key = jrandom.split(jrandom.PRNGKey(config.trainer.seed), 2)

    # This is largely the same as in Alpaca. Only change is we use the fast tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.model_cache_dir,
        model_max_length=config.max_tune_length,
        padding_side="right",
    )
    num_new_tokens = add_special_tokens(tokenizer)
    logger.info(f"Added {num_new_tokens} new tokens")

    # modify converter to use our tokenizer, mostly so it saves the right vocab
    converter = converter.replaced(tokenizer=tokenizer)

    train_dataset = mk_dataset(config, tokenizer)
    train_dataset = PermutationDataset(train_dataset, data_key)

    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    with Trainer(config.trainer, optimizer, loss_fn=compute_next_token_loss) as trainer:  # type: ignore
        # how we shard parameters across devices
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # load the underlying hf model
        logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
        model: LmHeadModel = converter.load_pretrained(  # type: ignore
            model_config.model_type, axis_mapping=parameter_axis_mapping, dtype=trainer.mp.param_dtype
        )

        # this must be in jit b/c it uses arrays across accelerators (b/c of FSDP)
        model = hax.named_jit(lambda m: m.resize_vocab(len(tokenizer)))(model)

        # Levanter has two kinds of data loaders: sharded and replicated. Replicated is simpler and allows for
        # single pass training. Sharded only loads a subset of the data on each device, and is more efficient for large
        # datasets. We use replicated here since the dataset is small.
        loader = trainer.data_loader(train_dataset)
        loader = non_caching_cycle(loader)

        state = trainer.initial_state(training_key, model=model)

        # TODO: remove this. we don't need it now
        if int(state.step) != 0:
            logger.info(f"Resuming training from step {state.step}")
            for i in range(state.step):
                next(loader)  # type: ignore

        # We also save HF checkpoints periodically (and at the end of training).
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, trainer.run_id)

            trainer.add_hook(
                save_hf_checkpoint_callback(full_save_path, converter, upload_to_hf=config.hf_upload),
                every=config.hf_save_steps,
            )

        trainer.train(state, loader)


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


if __name__ == "__main__":
    levanter.config.main(train)()
