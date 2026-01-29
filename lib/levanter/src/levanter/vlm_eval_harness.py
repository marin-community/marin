# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
VLM (Vision-Language Model) Evaluation Harness Adapter for Levanter.

This module provides integration between Levanter VLM models and the lm-eval-harness
framework for VLM benchmark evaluation. It supports multimodal tasks that involve
both images and text.

Key Features:
- Supports generate_until tasks for VLM benchmarks
- Handles image processing using BatchImageProcessor
- Uses LlavaInferenceEngine for VLM generation

Supported Benchmarks:
- MMMU, ChartQA (already in lm-eval-harness)
- MME, GQA, RealWorldQA, SEED, MMStar, AI2D, OCRBench (custom tasks)
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
from haliax import Axis, NamedArray
from haliax.partitioning import ResourceMapping
from PIL import Image
from tqdm_loggable.auto import tqdm

import levanter.tracker
from levanter.data.image import BatchImageProcessor
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.llava_onevision import (
    LlavaInferenceEngine,
    LlavaOnevisionModel,
    VLMRequest,
)
from levanter.utils.hf_utils import HfTokenizer

try:
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import TemplateLM
    from lm_eval.models.utils import handle_stop_sequences
except ImportError:
    TemplateLM = object
    Instance = object
    handle_stop_sequences = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VLMTaskConfig:
    """Configuration for a VLM evaluation task."""

    task: str
    """The name of the task to run."""
    task_alias: str | None = None
    """An alias for the task. We log this name to wandb."""
    num_fewshot: int | None = None

    def to_dict(self):
        """Convert the TaskConfig to a dictionary, excluding None values."""
        base_dict = dataclasses.asdict(self)
        return {k: v for k, v in base_dict.items() if v is not None}


@dataclass(frozen=True)
class VLMEvalHarnessConfig:
    """Configuration for running VLM evaluation with lm-eval-harness."""

    task_spec: list[VLMTaskConfig | str]
    """List of tasks to evaluate."""
    max_examples: int | None = None
    """Maximum number of examples per task."""
    max_length: int | None = None
    """Maximum sequence length."""
    max_images: int = 10
    """Maximum number of images per sample."""
    image_size: int = 384
    """Image size for processing."""
    patch_size: int = 16
    """Patch size for vision encoder. Default: 16 for SigLIP patch16."""
    vision_feature_height: int = 24
    """Vision feature height (num_image_tokens = height^2). Default: 24 for image_size=384, patch_size=16."""
    bootstrap_iters: int = 0
    apply_chat_template: bool = True
    confirm_run_unsafe_code: bool = True
    custom_task_path: str | None = None
    """Path to custom VLM task definitions (YAML files). If None, uses default configs/vlm_tasks/."""
    generation_kwargs: dict = field(
        default_factory=lambda: {"max_gen_toks": 256, "temperature": 0.0, "n": 1, "seed": None}
    )
    vlm_batch_size: int = 1
    """Number of VLM requests to process in parallel. Higher values use more memory but improve throughput.
    Set to -1 to auto-detect based on device count. Default: 1 (sequential processing)."""

    def to_task_spec(self) -> list[str | dict]:
        """Convert task specifications to a list of dictionaries or strings."""
        result = []
        for task in self.task_spec:
            if isinstance(task, str):
                result.append(task)
            else:
                result.append(task.to_dict())
        return result


class LevanterVLMHarnessLM(TemplateLM):
    """
    Levanter VLM implementation of the lm-eval-harness TemplateLM interface.

    This class provides the interface between Levanter VLM models and the lm-eval-harness
    evaluation framework, handling multimodal inputs (images + text).
    """

    MULTIMODAL = True

    def __init__(
        self,
        model: LlavaOnevisionModel,
        tokenizer: HfTokenizer,
        image_processor: BatchImageProcessor,
        EvalBatch: Axis,
        EvalPos: Axis,
        axis_resources: ResourceMapping,
        mp: jmp.Policy | None = None,
        generation_kwargs: dict | None = None,
        max_images: int = 10,
        vlm_batch_size: int = 1,
    ):
        """
        Initialize the VLM harness adapter.

        Args:
            model: The LlavaOnevision model
            tokenizer: HuggingFace tokenizer
            image_processor: BatchImageProcessor for processing images
            EvalBatch: Batch axis for evaluation
            EvalPos: Position axis for evaluation
            axis_resources: Resource mapping for sharding
            mp: Mixed precision policy
            generation_kwargs: Default generation parameters
            max_images: Maximum number of images per sample
            vlm_batch_size: Number of VLM requests to process in parallel (default: 1)
        """
        super().__init__()
        self.model = model
        self._tokenizer = tokenizer
        self.image_processor = image_processor
        self.EvalBatch = EvalBatch
        self.EvalPos = EvalPos
        self.axis_resources = axis_resources
        self.mp = mp
        self._generation_kwargs = generation_kwargs or {"max_gen_toks": 256, "temperature": 0.0, "n": 1, "seed": None}
        self.max_images = max_images
        self.vlm_batch_size = vlm_batch_size

        # Engine will be created lazily
        self._engine: LlavaInferenceEngine | None = None
        self._engine_config: InferenceEngineConfig | None = None

        # Sample logging
        self.sample_outputs: dict[str, list[dict]] = {}
        self._current_task = "vlm_task"

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def tokenizer_name(self) -> str:
        """Return a string identifier for the tokenizer."""
        if hasattr(self.tokenizer, "name_or_path"):
            return self.tokenizer.name_or_path
        return "unknown_tokenizer"

    @property
    def eot_token_id(self) -> int:
        """Return the end-of-text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        """Backward compatibility property for max_gen_toks."""
        return self._generation_kwargs.get("max_gen_toks", 256)

    @property
    def generation_kwargs(self):
        """Get the generation kwargs."""
        return self._generation_kwargs

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        """Required abstract method from TemplateLM. Not yet implemented for VLM."""
        raise NotImplementedError("_loglikelihood_tokens is not yet supported for VLM evaluation")

    def chat_template(self, chat_template: str | None = None) -> str | None:
        """Return the chat template for this model."""
        if chat_template is not None:
            return chat_template
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            return self.tokenizer.chat_template
        return None

    def apply_chat_template(self, chat_history: list[dict], **kwargs) -> str:
        """Apply chat template to format a conversation history."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=kwargs.get("add_generation_prompt", True),
            **{k: v for k, v in kwargs.items() if k != "add_generation_prompt"},
        )

    def set_current_task(self, task_name: str):
        """Set the current task for sample logging."""
        self._current_task = task_name
        if task_name not in self.sample_outputs:
            self.sample_outputs[task_name] = []

    def get_sample_outputs(self) -> dict[str, list[dict]]:
        """Get all stored sample outputs."""
        return self.sample_outputs

    def clear_sample_outputs(self):
        """Clear all stored sample outputs."""
        self.sample_outputs.clear()

    def _get_engine(self) -> LlavaInferenceEngine:
        """Lazily create and return the inference engine."""
        if self._engine is None:
            max_length = self.EvalPos.size
            # Use vlm_batch_size for max_seqs to enable batched VLM inference
            max_seqs = max(1, self.vlm_batch_size)
            self._engine_config = InferenceEngineConfig(
                max_stop_seqs=4,
                max_stop_tokens=16,
                max_seq_len=max_length,
                max_seqs=max_seqs,
                page_size=8,
                compute_dtype=jnp.bfloat16,
                hbm_utilization=0.5,
            )
            self._engine = LlavaInferenceEngine.from_model_with_config(
                model=self.model,
                tokenizer=self.tokenizer,
                config=self._engine_config,
                Vocab=self.model.Vocab,
            )
        return self._engine

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int | None = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        """Tokenize a string or list of strings."""
        encoding = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids

        if left_truncate_len:
            if not isinstance(string, str):
                encoding = [enc[-left_truncate_len:] for enc in encoding]
            else:
                encoding = encoding[-left_truncate_len:]

        return encoding

    def _process_image(self, image: Any) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, str):
            # URL or file path
            if image.startswith(("http://", "https://")):
                import requests
                from io import BytesIO
                response = requests.get(image)
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                return Image.open(image).convert("RGB")
        elif isinstance(image, dict) and "bytes" in image:
            # HuggingFace bytes format
            from io import BytesIO
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
        elif isinstance(image, bytes):
            from io import BytesIO
            return Image.open(BytesIO(image)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")

    def _create_vlm_request(
        self,
        prompt: str,
        images: List[Image.Image],
        gen_kwargs: dict,
    ) -> VLMRequest:
        """Create a VLMRequest from prompt and images."""
        # For lm-eval-harness VLM tasks, the prompt already contains <image> placeholders
        # from doc_to_text(). We should NOT add {"type": "image"} content items,
        # as that would create duplicate placeholders.
        #
        # Instead, create a simple text message and let the HF processor
        # replace <image> placeholders with actual image tokens.
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        # Use the image processor to process the example
        # The processor will:
        # 1. Apply chat template to get formatted text
        # 2. Count <image> placeholders and match with provided images
        # 3. Replace <image> with actual image tokens
        example = {"messages": messages, "images": images}
        processed = self.image_processor([example])[0]

        # Extract processed data
        input_ids = processed["input_ids"]
        pixel_values = processed.get("pixel_values")
        grid_mask = processed.get("grid_mask")
        unpad_indices = processed.get("unpad_indices")
        num_unpadded_features = processed.get("num_unpadded_features")

        # Convert input_ids to list for prompt_tokens
        input_ids_list = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)

        # Create sequence decoding parameters
        max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
        temperature = gen_kwargs.get("temperature", 0.0)
        seed = gen_kwargs.get("seed")

        base_key = jrandom.PRNGKey(42 if seed is None else seed)
        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(len(input_ids_list) + max_gen_toks, dtype=jnp.int32),
            stop_tokens=None,  # Will be set later if needed
            temperature=jnp.array(temperature, dtype=jnp.float32),
            key=base_key,
        )

        # Convert to NamedArray format
        # Model expects 5D: (batch, num_patches, channels, height, width)
        Batch = hax.Axis("batch", 1)
        TotalPatches = hax.Axis("TotalPatches", pixel_values.shape[0]) if pixel_values is not None else None

        pixel_values_na = None
        grid_mask_na = None
        unpad_indices_na = None
        input_ids_na = None

        if pixel_values is not None:
            # pixel_values shape: (TOTAL_PATCHES, C, H, W) -> (batch, TOTAL_PATCHES, channels, height, width)
            pixel_values_na = hax.named(
                jnp.array(pixel_values)[None, ...],  # Add batch dimension
                (Batch, TotalPatches, hax.Axis("channels", pixel_values.shape[1]),
                 hax.Axis("height", pixel_values.shape[2]), hax.Axis("width", pixel_values.shape[3]))
            )

        if grid_mask is not None:
            # grid_mask shape: (TOTAL_PATCHES,) -> (batch, TOTAL_PATCHES)
            grid_mask_na = hax.named(jnp.array(grid_mask)[None, ...], (Batch, TotalPatches,))

        if unpad_indices is not None:
            UnpadFeatures = hax.Axis("UnpadFeatures", len(unpad_indices))
            unpad_indices_na = hax.named(jnp.array(unpad_indices), (UnpadFeatures,))

        # Convert input_ids to NamedArray with shape (batch, position)
        # LlavaInferenceEngine.generate() calls set_request_data(input_ids=...) which expects NamedArray
        Position = hax.Axis("position", len(input_ids_list))
        input_ids_array = jnp.array(input_ids_list, dtype=jnp.int32).reshape(1, -1)
        input_ids_na = hax.named(input_ids_array, (Batch, Position))

        return VLMRequest(
            prompt_tokens=input_ids_list,
            request_id=0,
            decode_params=seq_params,
            n_generations=1,
            pixel_values=pixel_values_na,
            grid_mask=grid_mask_na,
            input_ids=input_ids_na,
            unpad_indices=unpad_indices_na,
            num_unpadded_features=num_unpadded_features,
        )

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        Generate text for VLM requests with images.

        Processes requests in batches for better efficiency. Each batch is sent
        to the engine together, which processes them (currently sequentially,
        but this enables future batched processing optimizations).

        Args:
            requests: List of Instance objects from lm-eval-harness
                     Each request has args: (context, gen_kwargs, {"visual": images})
            disable_tqdm: Whether to disable progress bar

        Returns:
            List of generated strings
        """
        if self.tokenizer.pad_token_id is None:
            logger.warning("No pad token set. Setting to eos token.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        engine = self._get_engine()
        results: List[str] = []

        # Batch size for processing (can be tuned based on memory)
        batch_size = 8

        # Process requests in batches
        num_batches = (len(requests) + batch_size - 1) // batch_size
        pbar = tqdm(range(num_batches), desc="VLM generate_until (batched)", disable=disable_tqdm)

        for batch_idx in pbar:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(requests))
            batch_requests = requests[batch_start:batch_end]

            # Prepare VLM requests for this batch
            vlm_requests = []
            batch_contexts = []
            batch_kwargs = []
            batch_pil_images = []

            for req in batch_requests:
                # Extract context, gen_kwargs, and multimodal args
                args = req.args
                if len(args) >= 3 and isinstance(args[2], dict):
                    context = args[0]
                    gen_kwargs = args[1] if len(args) > 1 else {}
                    multimodal_args = args[2]
                    images = multimodal_args.get("visual", [])
                elif len(args) >= 2:
                    context = args[0]
                    gen_kwargs = args[1] if isinstance(args[1], dict) else {}
                    images = []
                else:
                    context = args[0]
                    gen_kwargs = {}
                    images = []

                # Process generation kwargs
                processed_kwargs = self._modify_gen_kwargs(gen_kwargs.copy())

                # Override with our default kwargs
                for key, value in self._generation_kwargs.items():
                    if key not in processed_kwargs:
                        processed_kwargs[key] = value

                # Convert images to PIL format
                pil_images = [self._process_image(img) for img in images] if images else []

                batch_contexts.append(context)
                batch_kwargs.append(processed_kwargs)
                batch_pil_images.append(pil_images)

                try:
                    if pil_images:
                        # Create VLM request with images
                        vlm_request = self._create_vlm_request(context, pil_images, processed_kwargs)
                        vlm_requests.append(vlm_request)
                    else:
                        # No images - append None placeholder
                        vlm_requests.append(None)
                except Exception as e:
                    logger.error(f"Error creating VLM request: {e}")
                    vlm_requests.append(None)

            # Generate for all valid requests in the batch
            valid_requests = [r for r in vlm_requests if r is not None]
            valid_indices = [i for i, r in enumerate(vlm_requests) if r is not None]

            batch_results = [""] * len(vlm_requests)

            if valid_requests:
                try:
                    # Generate using the engine for all valid requests at once
                    result = engine.generate(valid_requests)

                    # Decode the generated tokens for each request
                    for i, idx in enumerate(valid_indices):
                        if result.tokens and i < len(result.tokens):
                            generated_tokens = result.tokens[i]
                            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                            # Post-process with stop sequences
                            until = batch_kwargs[idx].get("until", [])
                            if until:
                                for stop_seq in until:
                                    if stop_seq in text:
                                        text = text.split(stop_seq)[0]
                                        break

                            batch_results[idx] = text
                except Exception as e:
                    logger.error(f"Error during batch VLM generation: {e}")

            # Add batch results to overall results
            for i, text in enumerate(batch_results):
                results.append(text)

                # Log sample if enabled
                bucket = self.sample_outputs.get(self._current_task, [])
                if len(bucket) < 100:  # Limit samples
                    req = batch_requests[i]
                    sample_data = {
                        "prompt": batch_contexts[i],
                        "generation": text,
                        "num_images": len(batch_pil_images[i]) if batch_pil_images[i] else 0,
                    }
                    # Try to get expected answer from the request's doc
                    if hasattr(req, 'doc') and req.doc is not None:
                        doc = req.doc
                        # Common answer field names in VLM benchmarks
                        for key in ['answer', 'target', 'gold', 'label', 'response']:
                            if key in doc:
                                sample_data["expected"] = str(doc[key])
                                break
                        # Also capture question if available
                        for key in ['question', 'query', 'text']:
                            if key in doc and key not in sample_data:
                                sample_data["question"] = str(doc[key])[:200]
                                break
                    # Capture task name
                    if hasattr(req, 'task_name'):
                        sample_data["task"] = req.task_name
                    bucket.append(sample_data)
                    self.sample_outputs[self._current_task] = bucket

                    # Incremental log to wandb (every 10 samples)
                    if len(bucket) % 10 == 0:
                        try:
                            levanter.tracker.log({
                                f"vlm_eval/{self._current_task}/progress": len(bucket),
                                f"vlm_eval/{self._current_task}/latest_prompt": sample_data.get("prompt", "")[:100],
                                f"vlm_eval/{self._current_task}/latest_generation": text[:200],
                                f"vlm_eval/{self._current_task}/latest_expected": sample_data.get("expected", "N/A"),
                            }, step=len(bucket))
                            logger.info(f"[{self._current_task}] Progress: {len(bucket)} samples evaluated")
                        except Exception as e:
                            logger.debug(f"Failed to log incremental progress: {e}")

        return results

    def loglikelihood(self, requests: List[Instance], disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of generating a continuation from a context.

        Note: This is a simplified implementation. For multiple-choice tasks,
        we compute the forward pass and return log probabilities.
        """
        # For now, we raise NotImplementedError since most VLM benchmarks
        # use generate_until rather than loglikelihood
        raise NotImplementedError(
            "loglikelihood for VLM is not yet implemented. "
            "Most VLM benchmarks use generate_until instead."
        )

    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        raise NotImplementedError("loglikelihood_rolling is not supported for VLM")

    @staticmethod
    def _modify_gen_kwargs(kwargs: dict) -> dict:
        """Modify generation kwargs to standardize parameters."""
        # Handle temperature
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            kwargs["temperature"] = max(0.0, float(kwargs["temperature"]))
        else:
            kwargs.setdefault("temperature", 0.0)

        # Handle do_sample parameter
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False and kwargs["temperature"] > 0.0:
            raise ValueError(
                f"Conflicting parameters: do_sample=False but temperature={kwargs['temperature']} > 0.0."
            )

        # Handle max_gen_toks parameter
        if "max_gen_toks" in kwargs and kwargs["max_gen_toks"] is not None:
            kwargs["max_gen_toks"] = int(kwargs["max_gen_toks"])
        else:
            kwargs.setdefault("max_gen_toks", 256)

        # Handle n generations parameter
        if "n" in kwargs and kwargs["n"] is not None:
            kwargs["n"] = int(kwargs["n"])
        else:
            kwargs.setdefault("n", 1)

        return kwargs


def _get_default_vlm_tasks_path() -> str:
    """Get the default path to VLM task configurations."""
    import os
    # Try to find configs/vlm_tasks relative to this file or the project root
    # This file is at lib/levanter/src/levanter/vlm_eval_harness.py
    # configs/vlm_tasks is at the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 4 levels: levanter -> src -> levanter -> lib -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    vlm_tasks_path = os.path.join(project_root, "configs", "vlm_tasks")
    if os.path.exists(vlm_tasks_path):
        return vlm_tasks_path
    # Fallback: try relative to current working directory
    cwd_path = os.path.join(os.getcwd(), "configs", "vlm_tasks")
    if os.path.exists(cwd_path):
        return cwd_path
    return vlm_tasks_path  # Return default even if not found


def run_vlm_eval_harness(
    model: LlavaOnevisionModel,
    tokenizer: HfTokenizer,
    image_processor: BatchImageProcessor,
    config: VLMEvalHarnessConfig,
    EvalBatch: Axis,
    EvalPos: Axis,
    axis_resources: ResourceMapping,
    mp: jmp.Policy | None = None,
) -> dict:
    """
    Run VLM evaluation using lm-eval-harness.

    Args:
        model: The LlavaOnevision model
        tokenizer: HuggingFace tokenizer
        image_processor: BatchImageProcessor for image processing
        config: VLM evaluation configuration
        EvalBatch: Batch axis for evaluation
        EvalPos: Position axis for evaluation
        axis_resources: Resource mapping for sharding
        mp: Mixed precision policy

    Returns:
        Dictionary containing evaluation results
    """
    import os
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    # Register custom VLM tasks
    custom_task_path = config.custom_task_path or _get_default_vlm_tasks_path()
    if os.path.exists(custom_task_path):
        logger.info(f"Registering custom VLM tasks from: {custom_task_path}")
        # Use TaskManager to include custom tasks
        task_manager = TaskManager(include_path=custom_task_path)
    else:
        logger.warning(f"Custom task path not found: {custom_task_path}")
        task_manager = None

    # Create the VLM harness adapter
    vlm_lm = LevanterVLMHarnessLM(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        EvalBatch=EvalBatch,
        EvalPos=EvalPos,
        axis_resources=axis_resources,
        mp=mp,
        generation_kwargs=config.generation_kwargs,
        max_images=config.max_images,
    )

    # Run evaluation
    task_spec = config.to_task_spec()
    logger.info(f"Running VLM evaluation on tasks: {task_spec}")

    # Set current task for sample logging
    task_names = []
    for task in task_spec:
        if isinstance(task, str):
            task_names.append(task)
        elif isinstance(task, dict):
            task_names.append(task.get("task", "unknown"))
    for task_name in task_names:
        vlm_lm.set_current_task(task_name)

    # Use task_manager if custom tasks are registered
    eval_kwargs = {
        "model": vlm_lm,
        "tasks": task_spec,
        "limit": config.max_examples,
        "bootstrap_iters": config.bootstrap_iters,
        "apply_chat_template": config.apply_chat_template,
        "confirm_run_unsafe_code": config.confirm_run_unsafe_code,
    }
    if task_manager is not None:
        eval_kwargs["task_manager"] = task_manager

    results = evaluator.simple_evaluate(**eval_kwargs)

    # Log results
    if results and "results" in results:
        for task_name, metrics in results["results"].items():
            logger.info(f"{task_name}: {metrics}")
            # Log to tracker
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    levanter.tracker.log({f"vlm_eval/{task_name}/{metric_name}": value}, step=0)

    # Get sample outputs and add to results
    sample_outputs = vlm_lm.get_sample_outputs()
    if sample_outputs:
        results["sample_outputs"] = sample_outputs
        # Log samples to wandb as a table
        try:
            import wandb
            for task_name, samples in sample_outputs.items():
                if samples:
                    # Create wandb table with sample outputs
                    table = wandb.Table(columns=["prompt", "generation", "expected", "num_images"])
                    for sample in samples[:50]:  # Limit to 50 samples per task
                        table.add_data(
                            sample.get("prompt", "")[:500],  # Truncate long prompts
                            sample.get("generation", ""),
                            sample.get("expected", "N/A"),
                            sample.get("num_images", 0),
                        )
                    levanter.tracker.log({f"vlm_eval/{task_name}/samples": table}, step=0)
                    logger.info(f"Logged {len(samples)} samples for {task_name} to wandb")
        except Exception as e:
            logger.warning(f"Failed to log samples to wandb: {e}")

    return results


def run_vlm_benchmark_direct(
    model: LlavaOnevisionModel,
    tokenizer: HfTokenizer,
    image_processor: BatchImageProcessor,
    benchmark_name: str,
    EvalBatch: Axis,
    EvalPos: Axis,
    axis_resources: ResourceMapping,
    mp: jmp.Policy | None = None,
    max_examples: int | None = None,
    generation_kwargs: dict | None = None,
) -> dict:
    """
    Run a single VLM benchmark directly without lm-eval-harness task definitions.

    This is useful for benchmarks not included in lm-eval-harness (MME, GQA,
    RealWorldQA, SEED, MMStar, AI2D, OCRBench).

    Args:
        model: The LlavaOnevision model
        tokenizer: HuggingFace tokenizer
        image_processor: BatchImageProcessor
        benchmark_name: Name of the benchmark (e.g., "mme", "gqa")
        EvalBatch: Batch axis
        EvalPos: Position axis
        axis_resources: Resource mapping
        mp: Mixed precision policy
        max_examples: Maximum examples to evaluate
        generation_kwargs: Generation parameters

    Returns:
        Dictionary with evaluation results
    """
    from datasets import load_dataset
    from tqdm import tqdm

    # Benchmark configurations
    BENCHMARK_CONFIGS = {
        "mme": {
            "dataset": "lmms-lab/MME",
            "split": "test",
            "image_key": "image",
            "question_key": "question",
            "answer_key": "answer",
            "prompt_template": "<image>\n{question}\nPlease answer yes or no.",
        },
        "gqa": {
            "dataset": "lmms-lab/GQA",
            "split": "testdev_balanced",
            "image_key": "image",
            "question_key": "question",
            "answer_key": "answer",
            "prompt_template": "<image>\n{question}\nAnswer the question using a single word or phrase.",
        },
        "realworldqa": {
            "dataset": "xai-org/RealworldQA",
            "split": "test",
            "image_key": "image",
            "question_key": "question",
            "answer_key": "answer",
            "prompt_template": "<image>\n{question}\nAnswer with the option letter (A, B, C, or D).",
            "has_choices": True,
        },
        "seed": {
            "dataset": "AILab-CVC/SEED-Bench",
            "split": "test",
            "image_key": "image",
            "question_key": "question",
            "answer_key": "answer",
            "prompt_template": "<image>\n{question}\nAnswer with the option letter from the given choices.",
            "has_choices": True,
        },
        "mmstar": {
            "dataset": "Lin-Chen/MMStar",
            "split": "val",
            "image_key": "image",
            "question_key": "question",
            "answer_key": "answer",
            "prompt_template": "<image>\n{question}\nAnswer with the option letter (A, B, C, or D).",
            "has_choices": True,
        },
        "ai2d": {
            "dataset": "lmms-lab/ai2d",
            "split": "test",
            "image_key": "image",
            "question_key": "question",
            "answer_key": "answer",
            "prompt_template": "<image>\n{question}\nAnswer with the option letter.",
            "has_choices": True,
        },
        "ocrbench": {
            "dataset": "echo840/OCRBench",
            "split": "test",
            "image_key": "image",
            "question_key": "question",
            "answer_key": "answer",
            "prompt_template": "<image>\n{question}",
        },
    }

    benchmark_name = benchmark_name.lower()
    if benchmark_name not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(BENCHMARK_CONFIGS.keys())}")

    config = BENCHMARK_CONFIGS[benchmark_name]
    gen_kwargs = generation_kwargs or {"max_gen_toks": 64, "temperature": 0.0}

    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']}")
    try:
        dataset = load_dataset(config["dataset"], split=config["split"], trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset {config['dataset']}: {e}")
        return {"error": str(e)}

    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    # Create VLM adapter
    vlm_lm = LevanterVLMHarnessLM(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        EvalBatch=EvalBatch,
        EvalPos=EvalPos,
        axis_resources=axis_resources,
        mp=mp,
        generation_kwargs=gen_kwargs,
    )

    # Run evaluation
    correct = 0
    total = 0
    results_list = []

    for example in tqdm(dataset, desc=f"Evaluating {benchmark_name}"):
        # Get image
        image = example.get(config["image_key"])
        if image is None:
            continue

        # Format prompt
        question = example.get(config["question_key"], "")
        prompt = config["prompt_template"].format(question=question)

        # Add choices if applicable
        if config.get("has_choices"):
            choices = example.get("choices", example.get("choice_list", []))
            if choices:
                choice_text = "\n".join([f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
                prompt = prompt.replace("{question}", f"{question}\n\n{choice_text}")

        # Get reference answer
        reference = example.get(config["answer_key"], "")

        # Generate
        try:
            # Create a mock Instance for generate_until
            class MockInstance:
                def __init__(self, args):
                    self.args = args

            instance = MockInstance((prompt, gen_kwargs, {"visual": [image]}))
            outputs = vlm_lm.generate_until([instance], disable_tqdm=True)
            prediction = outputs[0] if outputs else ""
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            prediction = ""

        # Evaluate
        is_correct = _evaluate_answer(prediction, reference, benchmark_name)
        if is_correct:
            correct += 1
        total += 1

        results_list.append({
            "prediction": prediction,
            "reference": str(reference),
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"{benchmark_name} accuracy: {accuracy:.4f} ({correct}/{total})")

    # Log samples to wandb
    try:
        import wandb
        table = wandb.Table(columns=["prediction", "reference", "correct"])
        for result in results_list[:50]:  # Limit to 50 samples
            table.add_data(
                result.get("prediction", "")[:500],
                result.get("reference", ""),
                result.get("correct", False),
            )
        levanter.tracker.log({f"vlm_eval/{benchmark_name}/samples": table}, step=0)
        logger.info(f"Logged {min(len(results_list), 50)} samples for {benchmark_name} to wandb")
    except Exception as e:
        logger.warning(f"Failed to log samples to wandb: {e}")

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results_list[:100],  # Sample results
    }


def _evaluate_answer(prediction: str, reference: str, benchmark_name: str) -> bool:
    """Evaluate if prediction matches reference for different benchmarks."""
    import re
    import string

    pred = prediction.strip().lower()
    ref = str(reference).strip().lower()

    # For yes/no benchmarks (MME)
    if benchmark_name == "mme":
        pred_yn = "yes" if "yes" in pred else ("no" if "no" in pred else pred.split()[0] if pred else "")
        return pred_yn == ref

    # For multiple choice benchmarks
    if benchmark_name in ["realworldqa", "seed", "mmstar", "ai2d"]:
        # Extract letter from prediction
        pred_letter = ""
        if pred and pred[0] in "abcdefgh":
            pred_letter = pred[0]
        else:
            for char in pred:
                if char in "abcdefgh":
                    pred_letter = char
                    break

        # Handle reference (could be letter or index)
        if ref.isdigit():
            ref_letter = chr(ord("a") + int(ref))
        elif len(ref) == 1 and ref in "abcdefgh":
            ref_letter = ref
        else:
            ref_letter = ref[0] if ref and ref[0] in "abcdefgh" else ref

        return pred_letter == ref_letter

    # For open-ended QA (GQA, OCRBench)
    # Normalize and compare
    def normalize(s):
        s = s.lower()
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = " ".join(s.split())
        return s

    pred_norm = normalize(pred)
    ref_norm = normalize(ref)

    return pred_norm == ref_norm or ref_norm in pred_norm


__all__ = [
    "VLMTaskConfig",
    "VLMEvalHarnessConfig",
    "LevanterVLMHarnessLM",
    "run_vlm_eval_harness",
    "run_vlm_benchmark_direct",
]
