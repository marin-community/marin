import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import jax
import jax.random as jrandom

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCompatConfig, ModelWithHfSerializationMixin, save_hf_checkpoint_callback
from levanter.data.audio import AudioIODatasetConfig, AudioMixtureDatasetConfig, AudioTextDataset
from levanter.models.asr_model import ASRConfig, AudioTextExample
from levanter.models.whisper import WhisperConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count


logger = logging.getLogger(__name__)


@dataclass
class TrainASRConfig:
    data: Union[AudioIODatasetConfig, AudioMixtureDatasetConfig] = field(default_factory=AudioMixtureDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: ASRConfig = field(default_factory=WhisperConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    batch_size: int = 16

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    """if provided, this will override the model config in the config. if true, use the default hf checkpoint for this model class"""
    use_hf_model_config: bool = False  # if true, replace the model config with the hf config from the checkpoint
    data_seed: Optional[int] = None  # if provided, will override the data seed from the trainer

    # TODO: atm we don't support loading from a checkpoint that has a different tokenizer. this is a bit annoying
    # TODO: atm you have to at least specify a levanter model config with the same type as the hf checkpoint

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000


def main(config: TrainASRConfig):
    tokenizer = config.data.the_tokenizer

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
            converter = converter.replaced(
                reference_checkpoint=config.initialize_from_hf,
                tokenizer=tokenizer,
                feature_extractor=config.data.the_feature_extractor,
            )
        else:
            converter = converter.replaced(tokenizer=tokenizer, feature_extractor=config.data.the_feature_extractor)

        if config.use_hf_model_config:
            # TODO: log diff of old and new config
            # NB: gross mutability
            config.model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(config.model, HFCompatConfig):
        converter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer, feature_extractor=config.data.the_feature_extractor)
    else:
        converter = None

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    def compute_loss(
        m,
        example: AudioTextExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> jax.numpy.ndarray | hax.NamedArray:
        return m.compute_loss(example, key=key, reduction=reduction, reduction_axis=reduction_axis)

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with Trainer(config.trainer, optimizer, compute_loss) as trainer:  # type: ignore
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        compute_axis_mapping = trainer.compute_axis_mapping
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # some axes we need
        Batch = config.trainer.TrainBatch
        EvalBatch = config.trainer.EvalBatch
        Pos = config.model.Pos
        KeyPos = config.model.KeyPos

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        eval_datasets = config.data.validation_sets()
        train_dataset = AudioTextDataset(
            config.data.train_set(key=data_key),
            Pos,
            [config.model.Mels, config.model.MelPos],
            KeyPos,
            ignore_index=config.data.pad_token_id,
        )

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build_asr(Vocab, key=model_key))

        if int(state.step) == 0:
            # TODO: I don't love that we init the model twice, but it's not a big deal i think?
            if config.initialize_from_hf:
                # initialize from an hf pretrained model
                logger.info(
                    "No training checkpoint found. Initializing model from HF checkpoint"
                    f" '{converter.reference_checkpoint}'"
                )
                # this is a bit gross, but we want to free up the memory from the model we just built
                state = dataclasses.replace(state, model=None)
                assert isinstance(config.model.asr_model_type, ModelWithHfSerializationMixin)
                model = converter.load_pretrained(config.model.asr_model_type, axis_mapping=parameter_axis_mapping)
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        if len(eval_datasets) == 0:
            logger.warning("No evaluation datasets provided.")

        for name, eval_dataset in eval_datasets.items():
            hax_eval_dataset = AudioTextDataset(
                eval_dataset,
                Pos,
                [config.model.Mels, config.model.MelPos],
                KeyPos,
                ignore_index=config.data.pad_token_id,
            )
            trainer.add_eval_hook(hax_eval_dataset, name=name)

        trainer.add_hook(callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size), every=1)
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, trainer.run_id)

            trainer.add_hook(
                save_hf_checkpoint_callback(
                    full_save_path, converter, upload_to_hf=config.hf_upload or False, save_feature_extractor=True
                ),
                every=config.hf_save_steps,
            )

        # visualize log probs
        @named_jit(
            in_axis_resources=parameter_axis_mapping,
            axis_resources=compute_axis_mapping,
            out_axis_resources=compute_axis_mapping,
        )
        def compute_log_probs(model, example):
            model = trainer.mp.cast_to_compute(model)
            logprobs = model.compute_loss(example, key=None, reduction=None)
            # roll forward to get the loss for each predicted token
            logprobs = hax.roll(logprobs, 1, Pos)
            return logprobs.rearrange((EvalBatch, Pos)).array

        train_loader = trainer.data_loader(train_dataset, Batch).iter_from_step(state.step)

        ## OK, actually run training!
        trainer.train(state, train_loader)
        # checkpointer.on_step(last_step, force=True)


if __name__ == "__main__":
    levanter.config.main(main)()
