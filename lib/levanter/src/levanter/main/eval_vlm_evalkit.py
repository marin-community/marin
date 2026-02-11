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

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
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
    # VLMEvalKit uses LMUDataRoot for saving intermediate files (e.g., score JSONs).
    # It doesn't support GCS paths, so we set it to a local path if not already local.
    lmu_data_root = os.environ.get("LMUDataRoot", "")
    if not lmu_data_root or lmu_data_root.startswith("gs://"):
        os.environ["LMUDataRoot"] = "/tmp/vlmeval_data"

    from vlmeval.smp import dump, load
    from vlmeval.dataset import build_dataset
    from vlmeval.tools import EVAL
    VLMEVAL_AVAILABLE = True
except ImportError as e:
    VLMEVAL_AVAILABLE = False
    VLMEVAL_IMPORT_ERROR = str(e)


logger = logging.getLogger(__name__)


def _extract_checkpoint_name(checkpoint_path: Optional[str], hf_checkpoint: Optional[str]) -> str:
    """Extract a clean name from checkpoint path for use in filenames.

    Converts paths like:
    - gs://marin-us-east1/checkpoints/vlm-model-name/hf/step-12345/ -> us_east1_vlm_model_name_step_12345
    - gs://marin-us-central1/hf/model-name/step-1000/ -> us_central1_model_name_step_1000
    - Qwen/Qwen2-VL-7B -> Qwen_Qwen2_VL_7B

    Special characters are replaced with underscores.
    """
    import re

    path = checkpoint_path or hf_checkpoint or "unknown"

    # Remove trailing slashes
    path = path.rstrip("/")

    # For GCS paths, try to extract meaningful parts
    if path.startswith("gs://"):
        # Remove gs:// prefix and bucket name
        parts = path.split("/")
        # parts[0] is "gs:", parts[1] is "", parts[2] is bucket name
        bucket_name = parts[2] if len(parts) > 2 else ""

        # Extract zone from bucket name (e.g., "marin-us-east1" -> "us-east1")
        zone_prefix = ""
        if bucket_name.startswith("marin-"):
            zone_prefix = bucket_name[len("marin-"):]  # e.g., "us-east1"

        # Find 'hf' directory if present and take everything after it
        if "hf" in parts:
            hf_idx = parts.index("hf")
            meaningful_parts = parts[hf_idx + 1:]
        else:
            # Otherwise take last 2-3 meaningful parts
            meaningful_parts = [p for p in parts[-3:] if p and p not in ("checkpoints", "hf")]

        path = "_".join([zone_prefix] + meaningful_parts) if zone_prefix else "_".join(meaningful_parts)
    else:
        # For HF model names like "Qwen/Qwen2-VL-7B", just use the path
        path = path.split("/")[-1] if "/" in path else path

    # Replace special characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_]", "_", path)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")

    # Truncate if too long (filesystem limits)
    if len(name) > 100:
        name = name[:100]

    return name or "unknown_checkpoint"


def _get_checkpoint_path(output_dir: str, checkpoint_name: str, benchmark: str) -> str:
    """Get the checkpoint file path for a given benchmark."""
    return f"{output_dir.rstrip('/')}/checkpoints/{checkpoint_name}_{benchmark}_checkpoint.json"


def _save_eval_checkpoint(
    checkpoint_path: str,
    benchmark: str,
    results: dict,
    processed_indices: list,
    total_samples: int,
):
    """Save evaluation checkpoint to support resume."""
    checkpoint_data = {
        "benchmark": benchmark,
        "results": {str(k): v for k, v in results.items()},  # Convert keys to strings for JSON
        "processed_indices": list(processed_indices),  # Keep original type (int or str)
        "total_samples": total_samples,
    }

    fs, plain_path = fsspec.core.url_to_fs(checkpoint_path)
    # Ensure parent directory exists
    parent_dir = "/".join(plain_path.rsplit("/", 1)[:-1])
    if parent_dir:
        fs.makedirs(parent_dir, exist_ok=True)

    with fs.open(plain_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    logger.info(f"Checkpoint saved: {checkpoint_path} ({len(processed_indices)}/{total_samples} samples)")


def _load_eval_checkpoint(checkpoint_path: str) -> Optional[dict]:
    """Load evaluation checkpoint if it exists."""
    try:
        fs, plain_path = fsspec.core.url_to_fs(checkpoint_path)
        if not fs.exists(plain_path):
            return None

        with fs.open(plain_path, "r") as f:
            checkpoint_data = json.load(f)

        # Convert string keys back to their original type (int if possible, otherwise keep as string)
        def _try_parse_key(k):
            try:
                return int(k)
            except ValueError:
                return k

        checkpoint_data["results"] = {_try_parse_key(k): v for k, v in checkpoint_data["results"].items()}
        checkpoint_data["processed_indices"] = set(checkpoint_data["processed_indices"])

        logger.info(f"Loaded checkpoint: {checkpoint_path} ({len(checkpoint_data['processed_indices'])}/{checkpoint_data['total_samples']} samples)")
        return checkpoint_data
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def _delete_eval_checkpoint(checkpoint_path: str):
    """Delete checkpoint file after successful completion."""
    try:
        fs, plain_path = fsspec.core.url_to_fs(checkpoint_path)
        if fs.exists(plain_path):
            fs.rm(plain_path)
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")


def _compute_per_sample_results(dataset, dataset_name, pred_data):
    """Compute per-sample parsed answers and correctness using VLMEvalKit's scoring logic.

    Handles three dataset types:
    - VQA (ChartQA, GQA, TextVQA, DocVQA, etc.): Uses process_line + hit_calculate
    - MCQ (MMMU, MMStar, AI2D, etc.): Uses prefetch_answer for rule-based extraction
    - Y/N (MME, POPE, etc.): Uses YOrN_Extraction

    Returns a list of dicts with 'parsed_answer', 'correct', and 'score' per sample,
    or None if parsing is not supported or fails.
    """
    try:
        dataset_type = getattr(dataset, 'TYPE', '')

        if dataset_type == 'VQA':
            return _compute_vqa_per_sample(dataset_name, pred_data)
        elif dataset_type in ('MCQ', 'MCQ_MMMU_Pro'):
            return _compute_mcq_per_sample(pred_data)
        elif dataset_type == 'Y/N':
            return _compute_yn_per_sample(pred_data)
        else:
            logger.info(f"Per-sample scoring not implemented for dataset type '{dataset_type}'")
            return None
    except Exception as e:
        logger.warning(f"Failed to compute per-sample results for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _compute_vqa_per_sample(dataset_name, pred_data):
    """Compute per-sample results for VQA-type benchmarks."""
    from vlmeval.dataset.utils.vqa_eval import process_line, hit_calculate
    from vlmeval.smp import listinstr

    lines = [pred_data.iloc[i] for i in range(len(pred_data))]

    if listinstr(['ChartQA'], dataset_name):
        method = 'relaxed_accuracy'
    elif listinstr(['OCRVQA', 'GQA'], dataset_name):
        method = 'accuracy'
    elif listinstr(['DocVQA', 'InfoVQA'], dataset_name):
        method = 'anls'
    elif listinstr(['TextVQA'], dataset_name):
        method = 'vqa_score'
    else:
        method = 'vqa_score'

    results = [process_line(line, method=method) for line in lines]
    hits = hit_calculate(results, dataset_name)

    per_sample = []
    for res, hit_score in zip(results, hits):
        per_sample.append({
            'parsed_answer': str(res.get('pred', '')),
            'correct': bool(hit_score > 0),
            'score': round(float(hit_score), 4),
        })
    return per_sample


def _compute_mcq_per_sample(pred_data):
    """Compute per-sample results for MCQ-type benchmarks using rule-based extraction."""
    from vlmeval.dataset.utils.multiple_choice import prefetch_answer

    per_sample = []
    for _, row in pred_data.iterrows():
        prediction = str(row.get('prediction', ''))
        answer = str(row.get('answer', '')).strip().upper()

        # prefetch_answer uses can_infer() for rule-based letter extraction (no GPT needed)
        extracted = prefetch_answer(row)
        if extracted:
            parsed = str(extracted).strip().upper()
        else:
            parsed = prediction.strip()

        correct = (parsed == answer)
        per_sample.append({
            'parsed_answer': parsed,
            'correct': correct,
            'score': 1.0 if correct else 0.0,
        })
    return per_sample


def _compute_yn_per_sample(pred_data):
    """Compute per-sample results for Y/N-type benchmarks."""
    from vlmeval.dataset.utils.yorn import YOrN_Extraction

    per_sample = []
    for _, row in pred_data.iterrows():
        prediction = str(row.get('prediction', ''))
        answer = str(row.get('answer', '')).strip()

        extracted = YOrN_Extraction(prediction)
        correct = (extracted.lower() == answer.lower()) if extracted != 'Unknown' else False
        per_sample.append({
            'parsed_answer': extracted,
            'correct': correct,
            'score': 1.0 if correct else 0.0,
        })
    return per_sample


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

# 4B model configuration (Qwen3-4B)
DEFAULT_4B_TEXT_CONFIG = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=2560,
    intermediate_dim=9728,
    num_heads=32,
    num_kv_heads=8,
    num_layers=36,
    head_dim=128,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
    gradient_checkpointing=False,
    flash_attention_block_size=FLASH_ATTENTION_BLOCK_SIZE,
)

DEFAULT_4B_VLM_CONFIG = LlavaOnevisionConfig(
    vision_config=DEFAULT_VISION_CONFIG,
    text_config=DEFAULT_4B_TEXT_CONFIG,
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
    axes={"data": 1, "replica": 1, "model": -1},  # model parallelism: shard attention heads across chips
    shared_mapping={"kv_head": "model"},  # shard kv_head across model axis to reduce per-chip VMEM in paged attention
    compute_mapping={
        "vision_batch": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    },
    param_mapping={"embed": "data"},  # data=1, so effectively unsharded
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


def _detect_and_apply_4b_config(config: EvalVLMEvalKitConfig) -> EvalVLMEvalKitConfig:
    """Auto-detect 4B model from checkpoint path and switch to 4B config if needed."""
    checkpoint_path = config.checkpoint_path or (str(config.hf_checkpoint) if config.hf_checkpoint else None)
    if checkpoint_path is None:
        return config

    # Check if the checkpoint path contains "4b" (case-insensitive)
    if "4b" not in checkpoint_path.lower():
        return config

    # Only override if the user hasn't manually specified a non-default model config
    if config.model == DEFAULT_VLM_CONFIG:
        logger.info(f"Auto-detected 4B model from checkpoint path: {checkpoint_path}")
        logger.info("Switching to 4B model configuration (hidden_dim=2560, num_heads=32, num_layers=36)")
        config = dataclasses.replace(config, model=DEFAULT_4B_VLM_CONFIG)
    return config


def main(config: EvalVLMEvalKitConfig):
    """Main function for VLM evaluation using VLMEvalKit."""
    if not VLMEVAL_AVAILABLE:
        raise ImportError(
            f"VLMEvalKit is not available. Import error: {VLMEVAL_IMPORT_ERROR}"
        )

    config = _detect_and_apply_4b_config(config)

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
    # Use bfloat16 for inference to reduce vmem usage and match KV cache dtype
    mp: jmp.Policy = jmp.get_policy("compute=bfloat16,params=bfloat16,output=bfloat16")
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
            add_generation_prompt=True,
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
        # VLMEvalKit requires local paths for intermediate files (uses os.path.exists internally)
        # Use local temp dir for VLMEvalKit, then save final results to output_dir (which may be GCS)
        output_dir = config.output_dir or "./vlm_evalkit_results"
        local_work_dir = "./vlm_evalkit_work"
        os.makedirs(local_work_dir, exist_ok=True)
        model_name = "levanter_vlm"

        # Get checkpoint name for this evaluation run
        eval_checkpoint_name = _extract_checkpoint_name(
            config.checkpoint_path,
            str(config.hf_checkpoint) if config.hf_checkpoint else None
        )

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
                processed_indices = set()
                batch_size = max(1, config.vlm_batch_size)
                num_batches = (len(data) + batch_size - 1) // batch_size
                results_since_last_checkpoint = 0

                # Try to load checkpoint for resume
                checkpoint_path = _get_checkpoint_path(output_dir, eval_checkpoint_name, benchmark)
                if config.auto_resume:
                    checkpoint_data = _load_eval_checkpoint(checkpoint_path)
                    if checkpoint_data is not None:
                        results = checkpoint_data["results"]
                        processed_indices = checkpoint_data["processed_indices"]
                        logger.info(f"Resuming {benchmark} from checkpoint: {len(processed_indices)}/{len(data)} samples already processed")

                for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {benchmark} (batch_size={batch_size})"):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, len(data))
                    batch_data = data.iloc[batch_start:batch_end]

                    # Prepare batch requests
                    vlm_requests = []
                    batch_indices = []

                    for i, (_, item) in enumerate(batch_data.iterrows()):
                        idx = item['index']

                        # Skip already processed indices (for resume)
                        if idx in processed_indices:
                            continue

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
                            processed_indices.add(idx)

                    # Generate for batch
                    if vlm_requests:
                        try:
                            result = vlm_model._engine.generate(vlm_requests)

                            # Debug: log generation result info
                            num_tokens = len(result.tokens) if result.tokens else 0
                            logger.info(f"Batch {batch_idx}: {len(vlm_requests)} requests, {num_tokens} token sequences returned")

                            # Decode results
                            # Note: result.tokens contains only the newly generated tokens, not prompt + generation
                            for i, idx in enumerate(batch_indices):
                                if result.tokens and i < len(result.tokens):
                                    generated_tokens = result.tokens[i]
                                    text = vlm_model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                                    results[idx] = text.strip()
                                    processed_indices.add(idx)
                                    if not text.strip():
                                        logger.warning(f"Empty generation for index {idx}: generated_len={len(generated_tokens)}")
                                else:
                                    logger.warning(f"No tokens for index {idx}: result.tokens={result.tokens is not None}, i={i}, num_tokens={num_tokens}")
                                    results[idx] = ""
                                    processed_indices.add(idx)
                        except Exception as e:
                            logger.error(f"Error during batch generation: {e}")
                            import traceback
                            traceback.print_exc()
                            for idx in batch_indices:
                                if idx not in results:
                                    results[idx] = ""
                                processed_indices.add(idx)
                    else:
                        logger.warning(f"Batch {batch_idx}: No valid VLM requests created")

                    processed = len(processed_indices)
                    results_since_last_checkpoint += len(batch_indices)
                    if processed % 50 == 0 or processed == len(data):
                        logger.info(f"Processed {processed}/{len(data)} samples")

                    # Save checkpoint periodically (use >= to avoid skipping when batch_size doesn't align with interval)
                    if config.checkpoint_interval > 0 and results_since_last_checkpoint >= config.checkpoint_interval:
                        _save_eval_checkpoint(
                            checkpoint_path=checkpoint_path,
                            benchmark=benchmark,
                            results=results,
                            processed_indices=list(processed_indices),
                            total_samples=len(data),
                        )
                        results_since_last_checkpoint = 0

                # Save predictions to Excel file
                result_file = os.path.join(local_work_dir, f'{model_name}_{dataset_name}.xlsx')
                pred_data = data.copy()
                pred_data['prediction'] = [str(results.get(x, '')) for x in pred_data['index']]
                if 'image' in pred_data.columns:
                    pred_data = pred_data.drop(columns=['image'])
                pred_data.to_excel(result_file, index=False)
                logger.info(f"Predictions saved to: {result_file}")

                # Run evaluation
                eval_result = EVAL(dataset_name, result_file)

                # Compute per-sample parsed answers and correctness
                per_sample_results = _compute_per_sample_results(dataset, dataset_name, pred_data)

                # Collect predictions for JSON output, enriched with parsed answers and correctness
                predictions_list = []
                for i, (_, row) in enumerate(pred_data.iterrows()):
                    entry = {
                        "index": row['index'],  # Keep original type (int or str)
                        "question": str(row.get('question', '')),
                        "answer": str(row.get('answer', '')),
                        "prediction": str(row.get('prediction', '')),
                    }
                    if per_sample_results is not None and i < len(per_sample_results):
                        entry["parsed_answer"] = per_sample_results[i]["parsed_answer"]
                        entry["correct"] = per_sample_results[i]["correct"]
                        entry["score"] = per_sample_results[i]["score"]
                    predictions_list.append(entry)
                all_predictions[benchmark] = predictions_list
                # Convert DataFrame to dict for proper JSON serialization
                # VLMEvalKit's EVAL() often returns pandas DataFrames which get
                # truncated with "..." when serialized via str()/repr()
                if isinstance(eval_result, pd.DataFrame):
                    records = eval_result.to_dict(orient='records')
                    eval_result = records[0] if len(records) == 1 else records
                elif isinstance(eval_result, pd.Series):
                    eval_result = eval_result.to_dict()
                all_results[benchmark] = eval_result
                logger.info(f"{benchmark} results: {eval_result}")

                # Delete checkpoint after successful completion
                _delete_eval_checkpoint(checkpoint_path)

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
                # Save checkpoint on error for resume
                try:
                    if config.checkpoint_interval > 0 and results and processed_indices:
                        _save_eval_checkpoint(
                            checkpoint_path=checkpoint_path,
                            benchmark=benchmark,
                            results=results,
                            processed_indices=list(processed_indices),
                            total_samples=len(data),
                        )
                except NameError:
                    pass  # Variables not defined yet

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

        # Save results (supports both local and GCS paths via fsspec)
        # output_dir was set earlier and supports GCS paths

        # Use checkpoint name instead of timestamp for better tracking and resume support
        checkpoint_name = _extract_checkpoint_name(config.checkpoint_path, str(config.hf_checkpoint) if config.hf_checkpoint else None)
        benchmark_names = "_".join(config.benchmarks[:3])
        if len(config.benchmarks) > 3:
            benchmark_names += f"_and_{len(config.benchmarks) - 3}_more"

        results_data = {
            "checkpoint_name": checkpoint_name,
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
        results_filename = f"results_{benchmark_names}_{checkpoint_name}.json"
        fs, plain_path = fsspec.core.url_to_fs(output_dir)
        fs.makedirs(plain_path, exist_ok=True)
        results_file = os.path.join(plain_path, results_filename)

        with fs.open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=_json_default)

        # Log with the original output_dir prefix for user reference
        display_path = f"{output_dir.rstrip('/')}/{results_filename}"
        logger.info(f"Results saved to: {display_path}")
        print(f"\nResults saved to: {display_path}")

        # Save samples to separate JSON file (like lm-eval-harness format)
        if all_predictions:
            samples_filename = f"samples_{benchmark_names}_{checkpoint_name}.json"
            samples_file = os.path.join(plain_path, samples_filename)
            with fs.open(samples_file, "w") as f:
                json.dump(all_predictions, f, indent=2, default=_json_default)
            samples_display_path = f"{output_dir.rstrip('/')}/{samples_filename}"
            logger.info(f"Sample outputs saved to: {samples_display_path}")
            print(f"Sample outputs saved to: {samples_display_path}")

        return all_results


def _json_default(obj):
    """JSON serialization fallback that handles DataFrames, numpy types, etc."""
    if isinstance(obj, pd.DataFrame):
        records = obj.to_dict(orient='records')
        return records[0] if len(records) == 1 else records
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    levanter.config.main(main)()
