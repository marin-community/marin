import gc
import logging
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax
import jmp

import haliax
import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.data import DataLoader
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfigBase
from levanter.eval import TaggedEvaluator, eval_model
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel, compute_next_token_loss
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode
import dataclasses
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray


logger = logging.getLogger(__name__)

# Create ensemble model that averages logits
class EnsembleModel(LmHeadModel):
    models: list[LmHeadModel] = eqx.field()
    ensemble_method: str = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    _config: LmConfig = eqx.field(static=True)
    _vocab: Axis = eqx.field(static=True)
    _pos: Axis = eqx.field(static=True)
    _keypos: Axis = eqx.field(static=True)
    _embed: Axis = eqx.field(static=True)

    def __init__(self, models, ensemble_method="mean", temperature=1.0):
        self.models = models
        self.ensemble_method = ensemble_method
        self.temperature = temperature
        # Use the first model's config and axes
        self._config = models[0].config
        self._vocab = models[0].Vocab
        self._pos = models[0].Pos
        self._keypos = models[0].KeyPos
        self._embed = models[0].Embed

    @property
    def config(self):
        return self._config

    @property
    def Vocab(self):
        return self._vocab

    @property
    def Pos(self):
        return self._pos

    @property
    def KeyPos(self):
        return self._keypos

    @property
    def Embed(self):
        return self._embed

    @classmethod
    def init(cls, Vocab: Axis, config: LmConfig, *, key: PRNGKeyArray) -> "EnsembleModel":
        # This is a bit of a hack since we don't actually initialize the models here
        # The models are loaded from checkpoints instead
        raise NotImplementedError("EnsembleModel should be created with pre-trained models")

    def resize_vocab(self, new_size: int, key=None) -> "EnsembleModel":
        # Resize vocab for all models in the ensemble
        new_models = []
        for model in self.models:
            new_model = model.resize_vocab(new_size, key=key)
            new_models.append(new_model)
        return dataclasses.replace(self, models=new_models)

    def activations(self, input_ids, attn_mask=None, *, key=None):
        # Stack activations from all models with a "model" axis
        activations = []
        for model in self.models:
            act = model.activations(input_ids, attn_mask, key=key)
            if isinstance(act, tuple):
                act = act[0]  # Handle models that return (activations, aux_loss)
            activations.append(act)
        return hax.stack("model", activations)

    def get_lm_head(self):
        # Stack LM heads from all models with a "model" axis
        heads = [model.get_lm_head() for model in self.models]
        return hax.stack("model", heads)

    def __call__(self, input_ids, attn_mask=None, *, key=None):
        # More memory-efficient: accumulate mean incrementally instead of stacking all logits
        if self.ensemble_method == "mean":
            # Incremental mean to avoid holding all logits in memory
            result = None
            for i, model in enumerate(self.models):
                logits = model(input_ids, attn_mask, key=key)
                if self.temperature != 1.0:
                    logits = logits / self.temperature
                if result is None:
                    result = logits
                else:
                    # Incremental mean: result = (result * i + logits) / (i + 1)
                    result = result + (logits - result) / (i + 1)
            return result
        else:
            # For max/min, we need to compare all logits
            logits_list = []
            for model in self.models:
                logits = model(input_ids, attn_mask, key=key)
                if self.temperature != 1.0:
                    logits = logits / self.temperature
                logits_list.append(logits)
            
            stacked_logits = hax.stack("model", logits_list)
            
            if self.ensemble_method == "max":
                return hax.max(stacked_logits, axis="model")
            elif self.ensemble_method == "min":
                return hax.min(stacked_logits, axis="model")
            else:
                raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


@dataclass
class EvalEnsembleConfig:
    checkpoint_paths: list[str] = field(default_factory=list)
    hf_checkpoints: list[RepoRef] = field(default_factory=list)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: SingleDatasetLMConfigBase | LMMixtureDatasetConfig = field(default_factory=SingleDatasetLMConfigBase)
    model: LmConfig = field(default_factory=Gpt2Config)

    eval_on_train: bool = False

    log_entropy: bool = False
    log_top2_gap: bool = False
    log_param_stats: bool = False

    # Ensemble specific parameters
    ensemble_method: str = "mean"  # Options: "mean", "max", "min"
    temperature: float = 1.0  # Temperature for softmax before averaging logits


def main(config: EvalEnsembleConfig):
    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer

    Batch = config.trainer.EvalBatch
    Pos = config.model.Pos

    if config.eval_on_train:
        datasets_dict = config.data.train_sets(Pos, key=jax.random.PRNGKey(0))
        # need tagged eval sets for the evaluator
        datasets = [(ds, [name]) for name, ds in datasets_dict.items()]
    else:
        datasets = config.data.tagged_eval_sets(Pos)

    if not datasets:
        raise ValueError("no dataset found!")

    if config.trainer.max_eval_batches is not None:
        max_examples = config.trainer.max_eval_batches * config.trainer.eval_batch_size
        datasets = [(ds.take(max_examples), tags) for ds, tags in datasets]
    else:
        max_examples = None

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    if not config.checkpoint_paths and not config.hf_checkpoints:
        raise ValueError("Must specify either checkpoint_paths or hf_checkpoints")
    if config.checkpoint_paths and config.hf_checkpoints:
        raise ValueError("Must specify either checkpoint_paths or hf_checkpoints, not both")

    with config.trainer.use_device_mesh(), hax.axis_mapping(parameter_axis_mapping):
        evaluator = TaggedEvaluator(
            Batch, datasets, tokenizer, max_examples_per_dataset=max_examples, axis_mapping=compute_axis_mapping
        )

        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp


        # Load all models
        models = []
        if config.checkpoint_paths:
            raise NotImplementedError("Ensemble model loading from checkpoints is not implemented")
            # for checkpoint_path in config.checkpoint_paths:
            #     with use_cpu_device():
            #         model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
            #         model = load_checkpoint(model, checkpoint_path, subpath="model")
            #     model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)
            #     models.append(model)
        else:  # hf_checkpoints
            for i, hf_checkpoint in enumerate(config.hf_checkpoints):
                logger.info(f"Loading model {i+1}/{len(config.hf_checkpoints)}: {hf_checkpoint}")
                model_config = config.model
                if not hasattr(model_config, "hf_checkpoint_converter"):
                    raise ValueError("Model config does not have an HF checkpoint converter. Can't load HF checkpoint.")
                converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
                converter = converter.replaced(reference_checkpoint=hf_checkpoint, tokenizer=tokenizer)
                model = converter.load_pretrained(
                    model_config.model_type, ref=hf_checkpoint, dtype=mp.compute_dtype
                )
                model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)
                # Force transfer to TPU and free host memory before loading next model
                jax.block_until_ready(model)
                gc.collect()
                models.append(model)
                logger.info(f"Model {i+1} loaded and sharded successfully")
        

        # Create ensemble model
        logger.info(f"Creating ensemble model with {len(models)} models using method={config.ensemble_method}")
        ensemble_model = EnsembleModel(
            models, 
            ensemble_method=config.ensemble_method,
            temperature=config.temperature
        )
        logger.info("Ensemble model created successfully")

        # @hax.named_jit
        # def compute_loss(model: LmHeadModel, example: LmExample):
        #     with hax.axis_mapping(compute_axis_mapping):
        #         model = inference_mode(model, True)
        #         model = mp.cast_to_compute(model)
        #         return compute_next_token_loss(model, example, key=None)

        def compute_logits(model: LmHeadModel, example: LmExample):
            model = mp.cast_to_compute(model)
            with hax.axis_mapping(compute_axis_mapping):
                return model(example.tokens, key=None, attn_mask=example.attn_mask)

        logger.info("Starting evaluation (this may take a while for JIT compilation)...")
        log_dict = eval_model(evaluator, ensemble_model, prefix="eval")
        logger.info("Evaluation completed successfully")

        levanter.tracker.log(log_dict, step=0)

        print("Loss:", log_dict["eval/loss"])

        if config.log_entropy:
            logger.info("Computing entropy...")
            for name, dataset in config.data.validation_sets(Pos).items():
                if config.trainer.max_eval_batches is not None:
                    dataset = dataset.take(config.trainer.max_eval_batches * config.trainer.eval_batch_size)
                loader = DataLoader(dataset, batch_size=config.trainer.eval_batch_size)
                entropy_hist = levanter.analysis.compute_entropy_histogram(
                    ensemble_model,
                    Vocab,
                    compute_logits,
                    loader,
                )

                levanter.tracker.log(
                    {
                        f"analysis/{name}/entropy": entropy_hist,
                    },
                    step=0,
                )

        if config.log_top2_gap:
            logger.info("Computing top2_gap...")
            for name, dataset in config.data.validation_sets(Pos).items():
                if config.trainer.max_eval_batches is not None:
                    dataset = dataset.take(config.trainer.max_eval_batches * config.trainer.eval_batch_size)
                    loader = DataLoader(dataset, batch_size=config.trainer.eval_batch_size)
                    top2_gap_hist = levanter.analysis.compute_top2_gap_histogram(
                        ensemble_model,
                        Vocab,
                        compute_logits,
                        loader,
                    )

                    levanter.tracker.log(
                        {
                            f"analysis/{name}/top2_gap": top2_gap_hist,
                        },
                        step=0,
                    )

        if config.log_param_stats:
            logger.info("Computing param stats...")
            log_dict = haliax.named_jit(levanter.analysis.summary_statistics_for_tree)(
                "params", ensemble_model, split_scan_layers=True, include_histogram=True
            )

            levanter.tracker.log(log_dict, step=0)

    # ray tasks don't reliably wait for the subprocesses to finish, so we need to manually finish the tracker
    levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
