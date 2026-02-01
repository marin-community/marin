# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
VLM Evaluation using VLMEvalKit.

This script provides evaluation of Vision-Language Models using the VLMEvalKit
framework (https://github.com/open-compass/VLMEvalKit).

VLMEvalKit supports 80+ benchmarks including:
- MME, GQA, RealWorldQA, SEED, AI2D, OCRBench
- MMMU, MathVista, ScienceQA, and many more

Usage:
    python -m levanter.main.eval_vlm_evalkit \
        --checkpoint_path /path/to/checkpoint \
        --benchmarks '["MME", "GQA", "RealWorldQA"]'
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import equinox as eqx
import fsspec
import jax
import jmp
import pandas as pd
from tqdm import tqdm

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis, round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_processor, load_tokenizer
from levanter.data.image import BatchImageProcessor
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llava_onevision import LlavaOnevisionConfig, LlavaOnevisionModel
from levanter.models.qwen import Qwen3Config
from levanter.models.siglip import SiglipVisionConfig
from levanter.distributed import RayConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from levanter.utils.tree_utils import inference_mode

try:
    from vlmeval.smp import dump, load
    from vlmeval.dataset import build_dataset
    from vlmeval.tools import EVAL
    VLMEVAL_AVAILABLE = True
except ImportError as e:
    VLMEVAL_AVAILABLE = False
    VLMEVAL_IMPORT_ERROR = str(e)


logger = logging.getLogger(__name__)


# ============================================================================
# DEFAULT MODEL CONFIGURATION (matches demo_vlm_train.py)
# ============================================================================

FLASH_ATTENTION_BLOCK_SIZE = 1024

DEFAULT_VISION_CONFIG = SiglipVisionConfig(
    hidden_size=1152,
    intermediate_size=4304,
    num_hidden_layers=27,
    num_attention_heads=16,
    image_size=384,
    patch_size=16,
    gradient_checkpointing=False,
    flash_attention_block_size=FLASH_ATTENTION_BLOCK_SIZE,
)

DEFAULT_TEXT_CONFIG = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=6144,
    num_heads=16,
    num_kv_heads=8,
    num_layers=28,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
    gradient_checkpointing=False,
    flash_attention_block_size=FLASH_ATTENTION_BLOCK_SIZE,
)

DEFAULT_IMAGE_TOKEN_INDEX = 151655

DEFAULT_VLM_CONFIG = LlavaOnevisionConfig(
    vision_config=DEFAULT_VISION_CONFIG,
    text_config=DEFAULT_TEXT_CONFIG,
    vision_encoder_type="siglip",
    vision_feature_select_strategy="full",
    vision_aspect_ratio="single",
    disable_anyres=True,
    image_token_index=DEFAULT_IMAGE_TOKEN_INDEX,
    gradient_checkpointing=False,
)

DEFAULT_VISION_FEATURE_HEIGHT = DEFAULT_VISION_CONFIG.image_size // DEFAULT_VISION_CONFIG.patch_size

DEFAULT_PROCESSOR_PATH = "gs://marin-vlm/processors/llava-onevision-qwen2-0.5b-ov-hf"
DEFAULT_TOKENIZER_PATH = "gs://marin-vlm/tokenizers/Qwen3-1.7B"

DEFAULT_MESH_CONFIG = MeshConfig(
    axes={"data": -1, "replica": 1, "model": 1},
    compute_mapping={
        "vision_batch": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    },
    param_mapping={"embed": "data"},
)


def _default_trainer_config():
    # Disable Ray auto-start for local evaluation
    ray_config = RayConfig(auto_start_cluster=False)
    return TrainerConfig(mesh=DEFAULT_MESH_CONFIG, ray=ray_config)


@dataclass
class EvalVLMEvalKitConfig:
    """Configuration for VLM evaluation using VLMEvalKit."""

    # Checkpoint loading options
    checkpoint_path: Optional[str] = None
    """Path to a Levanter checkpoint directory."""
    hf_checkpoint: Optional[RepoRef] = None
    """HuggingFace checkpoint reference."""

    # Model config
    trainer: TrainerConfig = field(default_factory=_default_trainer_config)
    model: LlavaOnevisionConfig = field(default_factory=lambda: DEFAULT_VLM_CONFIG)

    # Benchmarks to run
    benchmarks: List[str] = field(default_factory=lambda: ["MME"])
    """List of VLMEvalKit benchmarks to run. Examples: MME, GQA, RealWorldQA, SEED, AI2D, OCRBench."""

    # Processor/tokenizer paths
    processor_path: Optional[str] = DEFAULT_PROCESSOR_PATH
    tokenizer_path: Optional[str] = DEFAULT_TOKENIZER_PATH

    # Evaluation settings
    max_eval_length: int = 4096
    max_gen_toks: int = 512
    temperature: float = 0.0
    max_examples: Optional[int] = None
    """Maximum number of examples per benchmark. If None, evaluates all examples."""
    vlm_batch_size: int = 1
    """Number of VLM requests to process in parallel. Higher values use more memory."""

    # Image processing
    image_size: int = 384
    patch_size: int = 16
    vision_feature_height: int = DEFAULT_VISION_FEATURE_HEIGHT
    disable_anyres: bool = True

    # Checkpoint configuration
    checkpoint_interval: int = 100
    """Save checkpoint every N examples. Set to 0 to disable checkpointing."""
    checkpoint_dir: Optional[str] = None
    """Directory for checkpoint files. Defaults to {output_dir}/checkpoints/."""
    resume_from_checkpoint: Optional[str] = None
    """Path to checkpoint file to resume from."""
    auto_resume: bool = True
    """If True, automatically detect and resume from the latest checkpoint."""

    # Output
    output_dir: Optional[str] = None
    """Directory to save evaluation results. If None, uses ./vlm_evalkit_results/."""
    save_samples: bool = True
    """If True, save individual sample outputs."""


def _load_vlm_model(
    config: EvalVLMEvalKitConfig,
    Vocab: Axis,
    mp: jmp.Policy,
    parameter_axis_mapping,
    tokenizer,
    key,
) -> LlavaOnevisionModel:
    """Load VLM model from checkpoint or HuggingFace."""
    if config.checkpoint_path is not None and config.hf_checkpoint is not None:
        raise ValueError("Cannot specify both checkpoint_path and hf_checkpoint")
    if config.checkpoint_path is None and config.hf_checkpoint is None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint")

    if config.checkpoint_path is not None:
        logger.info(f"Loading model from checkpoint: {config.checkpoint_path}")
        with use_cpu_device():
            model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
            model = load_checkpoint(model, config.checkpoint_path, subpath="model")
        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

    elif config.hf_checkpoint is not None:
        logger.info(f"Loading model from HuggingFace: {config.hf_checkpoint}")
        model_config = config.model

        from transformers import LlavaOnevisionConfig as HfLlavaOnevisionConfig
        converter = HFCheckpointConverter(
            LevConfigClass=model_config.__class__,
            reference_checkpoint=str(config.hf_checkpoint),
            HfConfigClass=HfLlavaOnevisionConfig,
            tokenizer=tokenizer,
            trust_remote_code=True,
        )
        model = converter.load_pretrained(
            model_config.model_type,
            ref=config.hf_checkpoint,
            config=model_config,
            dtype=mp.compute_dtype,
            resize_vocab_to_match_tokenizer=False,
        )
        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

    return model


def main(config: EvalVLMEvalKitConfig):
    """Main function for VLM evaluation using VLMEvalKit."""
    if not VLMEVAL_AVAILABLE:
        raise ImportError(
            f"VLMEvalKit is not available. Import error: {VLMEVAL_IMPORT_ERROR}"
        )

    levanter.initialize(config)

    # Determine processor/tokenizer paths
    processor_path = config.processor_path
    if processor_path is None:
        if config.hf_checkpoint is not None:
            processor_path = str(config.hf_checkpoint)
        else:
            processor_path = config.model.default_hf_checkpoint_path

    tokenizer_path = config.tokenizer_path or processor_path

    logger.info(f"Loading processor from: {processor_path}")
    logger.info(f"Loading tokenizer from: {tokenizer_path}")

    processor = load_processor(processor_path, trust_remote_code=True)
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Setup axes
    Batch = config.trainer.EvalBatch
    Pos = config.model.max_Pos.resize(config.max_eval_length)

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    vocab_size = len(tokenizer)
    mp: jmp.Policy = config.trainer.mp
    key = jax.random.PRNGKey(0)

    with config.trainer.use_device_mesh(), hax.axis_mapping(parameter_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size}")

        # Load model
        model = _load_vlm_model(
            config=config,
            Vocab=Vocab,
            mp=mp,
            parameter_axis_mapping=parameter_axis_mapping,
            tokenizer=tokenizer,
            key=key,
        )

        model = inference_mode(model, True)
        model = mp.cast_to_compute(model)

        # Create image processor
        if config.disable_anyres:
            grid_pinpoints = [[config.image_size, config.image_size]]
            max_num_patches = 0
        else:
            grid_pinpoints = None
            max_num_patches = 9

        image_processor = BatchImageProcessor(
            processor=processor,
            tokenizer=tokenizer,
            max_length=config.max_eval_length,
            padding=False,
            disable_anyres=config.disable_anyres,
            grid_pinpoints=grid_pinpoints,
            patch_size=config.patch_size,
            vision_feature_height=config.vision_feature_height,
            max_num_patches=max_num_patches,
        )

        # Create LevanterVLM adapter
        from levanter.vlm_evalkit_adapter import LevanterVLM

        vlm_model = LevanterVLM(
            levanter_model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            EvalBatch=Batch,
            EvalPos=Pos,
            axis_resources=compute_axis_mapping,
            mp=mp,
            max_gen_toks=config.max_gen_toks,
            temperature=config.temperature,
            vlm_batch_size=config.vlm_batch_size,
        )

        # Run VLMEvalKit evaluation
        logger.info(f"Starting VLMEvalKit evaluation on benchmarks: {config.benchmarks}")

        # Setup working directory
        work_dir = config.output_dir or "./vlm_evalkit_results"
        if not work_dir.startswith("gs://"):
            os.makedirs(work_dir, exist_ok=True)
        model_name = "levanter_vlm"

        all_results = {}
        all_predictions = {}  # Store detailed predictions for each benchmark
        for benchmark in config.benchmarks:
            logger.info(f"Evaluating on {benchmark}...")
            try:
                # Build dataset using VLMEvalKit
                dataset = build_dataset(benchmark)
                dataset_name = dataset.dataset_name
                logger.info(f"Dataset {dataset_name} has {len(dataset)} samples")

                # Limit examples if specified
                data = dataset.data
                if config.max_examples is not None and len(data) > config.max_examples:
                    data = data.head(config.max_examples)
                    logger.info(f"Limited to {config.max_examples} examples")

                # Run inference in batches (similar to vlm_eval_harness.py:600-705)
                results = {}
                batch_size = max(1, config.vlm_batch_size)
                num_batches = (len(data) + batch_size - 1) // batch_size

                for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {benchmark} (batch_size={batch_size})"):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, len(data))
                    batch_data = data.iloc[batch_start:batch_end]

                    # Prepare batch requests
                    vlm_requests = []
                    batch_indices = []

                    for i, (_, item) in enumerate(batch_data.iterrows()):
                        idx = item['index']

                        # Build prompt
                        if hasattr(vlm_model, 'use_custom_prompt') and vlm_model.use_custom_prompt(dataset_name):
                            struct = vlm_model.build_prompt(item, dataset=dataset_name)
                        else:
                            struct = dataset.build_prompt(item)

                        try:
                            # Parse messages and create VLM request
                            prompt, images = vlm_model._parse_messages(struct)
                            vlm_request = vlm_model._create_vlm_request(prompt, images, request_id=i)
                            vlm_requests.append(vlm_request)
                            batch_indices.append(idx)
                        except Exception as e:
                            logger.error(f"Error creating request for index {idx}: {e}")
                            results[idx] = ""

                    # Generate for batch
                    if vlm_requests:
                        try:
                            result = vlm_model._engine.generate(vlm_requests)

                            # Debug: log generation result info
                            num_tokens = len(result.tokens) if result.tokens else 0
                            logger.info(f"Batch {batch_idx}: {len(vlm_requests)} requests, {num_tokens} token sequences returned")

                            # Decode results
                            for i, idx in enumerate(batch_indices):
                                if result.tokens and i < len(result.tokens):
                                    generated_tokens = result.tokens[i]
                                    prompt_len = len(vlm_requests[i].prompt_tokens)
                                    new_tokens = generated_tokens[prompt_len:]
                                    text = vlm_model.tokenizer.decode(new_tokens, skip_special_tokens=True)
                                    results[idx] = text.strip()
                                    if not text.strip():
                                        logger.warning(f"Empty generation for index {idx}: prompt_len={prompt_len}, generated_len={len(generated_tokens)}, new_tokens_len={len(new_tokens)}")
                                else:
                                    logger.warning(f"No tokens for index {idx}: result.tokens={result.tokens is not None}, i={i}, num_tokens={num_tokens}")
                                    results[idx] = ""
                        except Exception as e:
                            logger.error(f"Error during batch generation: {e}")
                            import traceback
                            traceback.print_exc()
                            for idx in batch_indices:
                                if idx not in results:
                                    results[idx] = ""
                    else:
                        logger.warning(f"Batch {batch_idx}: No valid VLM requests created")

                    processed = batch_end
                    if processed % 50 == 0 or processed == len(data):
                        logger.info(f"Processed {processed}/{len(data)} samples")

                # Save predictions to Excel file
                result_file = os.path.join(work_dir, f'{model_name}_{dataset_name}.xlsx')
                pred_data = data.copy()
                pred_data['prediction'] = [str(results.get(x, '')) for x in pred_data['index']]
                if 'image' in pred_data.columns:
                    pred_data = pred_data.drop(columns=['image'])
                pred_data.to_excel(result_file, index=False)
                logger.info(f"Predictions saved to: {result_file}")

                # Collect predictions for JSON output
                all_predictions[benchmark] = [
                    {
                        "index": int(row['index']),
                        "question": str(row.get('question', '')),
                        "answer": str(row.get('answer', '')),
                        "prediction": str(row.get('prediction', '')),
                    }
                    for _, row in pred_data.iterrows()
                ]

                # Run evaluation
                eval_result = EVAL(dataset_name, result_file)
                all_results[benchmark] = eval_result
                logger.info(f"{benchmark} results: {eval_result}")

                # Log to tracker
                if isinstance(eval_result, dict):
                    for metric_name, value in eval_result.items():
                        if isinstance(value, (int, float)):
                            levanter.tracker.log({
                                f"vlm_evalkit/{benchmark}/{metric_name}": value
                            }, step=0)

            except Exception as e:
                logger.error(f"Failed to evaluate {benchmark}: {e}")
                import traceback
                traceback.print_exc()
                all_results[benchmark] = {"error": str(e)}

        # Print summary
        print("\n" + "=" * 60)
        print("VLMEvalKit Evaluation Results")
        print("=" * 60)

        for benchmark, metrics in all_results.items():
            print(f"\n{benchmark}:")
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric_name}: {value:.4f}" if isinstance(value, float) else f"  {metric_name}: {value}")
                    elif metric_name == "error":
                        print(f"  ERROR: {value}")

        print("=" * 60)

        # Save results (supports both local and GCS paths)
        output_dir = config.output_dir or "./vlm_evalkit_results"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_names = "_".join(config.benchmarks[:3])
        if len(config.benchmarks) > 3:
            benchmark_names += f"_and_{len(config.benchmarks) - 3}_more"

        results_data = {
            "timestamp": timestamp,
            "checkpoint": config.checkpoint_path or str(config.hf_checkpoint),
            "benchmarks": config.benchmarks,
            "results": all_results,
            "predictions": all_predictions,
            "metadata": {
                "model": {
                    "vision_encoder": config.model.vision_encoder_type,
                    "vision_hidden_size": config.model.vision_config.hidden_size,
                    "vision_image_size": config.model.vision_config.image_size,
                    "vision_patch_size": config.model.vision_config.patch_size,
                    "text_hidden_size": config.model.text_config.hidden_dim,
                    "text_num_layers": config.model.text_config.num_layers,
                    "text_max_seq_len": config.model.text_config.max_seq_len,
                },
                "eval_config": {
                    "max_eval_length": config.max_eval_length,
                    "max_gen_toks": config.max_gen_toks,
                    "max_examples": config.max_examples,
                    "vlm_batch_size": config.vlm_batch_size,
                    "temperature": config.temperature,
                },
                "image_processing": {
                    "image_size": config.image_size,
                    "patch_size": config.patch_size,
                    "vision_feature_height": config.vision_feature_height,
                    "disable_anyres": config.disable_anyres,
                },
                "paths": {
                    "processor_path": config.processor_path,
                    "tokenizer_path": config.tokenizer_path,
                },
                "hardware": {
                    "num_devices": jax.device_count(),
                    "device_type": str(jax.devices()[0].platform) if jax.devices() else "unknown",
                },
            },
        }

        # Use fsspec to support both local and GCS paths
        results_filename = f"results_{benchmark_names}_{timestamp}.json"
        fs, plain_path = fsspec.core.url_to_fs(output_dir)
        fs.makedirs(plain_path, exist_ok=True)
        results_file = os.path.join(plain_path, results_filename)

        with fs.open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        # Log with the original output_dir prefix for user reference
        display_path = f"{output_dir.rstrip('/')}/{results_filename}"
        logger.info(f"Results saved to: {display_path}")
        print(f"\nResults saved to: {display_path}")

        # Save samples to separate JSON file (like lm-eval-harness format)
        if all_predictions:
            samples_filename = f"samples_{benchmark_names}_{timestamp}.json"
            samples_file = os.path.join(plain_path, samples_filename)
            with fs.open(samples_file, "w") as f:
                json.dump(all_predictions, f, indent=2, default=str)
            samples_display_path = f"{output_dir.rstrip('/')}/{samples_filename}"
            logger.info(f"Sample outputs saved to: {samples_display_path}")
            print(f"Sample outputs saved to: {samples_display_path}")

        return all_results


if __name__ == "__main__":
    levanter.config.main(main)()
