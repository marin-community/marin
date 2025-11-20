#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PlantCAD evaluation script: Performance on a single, zero-shot DNA sequence conservation task
"""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import fsspec
import haliax as hax
import haliax.haxtyping as ht
import jax.numpy as jnp
import numpy as np
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.utilities.json_encoder import CustomJsonEncoder

logger = logging.getLogger("ray")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class DnaEvalConfig:
    """Configuration for DNA model evaluation on conservation prediction."""

    checkpoint_path: str
    output_path: str
    dataset_path: str = "plantcad/evolutionary-constraint"
    dataset_config: str = "10k"
    dataset_split: str = "validation"
    batch_size: int = 16  # Largest batch size for 600M model + 40G A100
    max_samples: int | None = None  # Set to limit samples for testing
    device: str = "cuda"
    dtype: str = "bfloat16"
    random_seed: int = 42


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class ConservationResult:
    """Result of scoring an evaluation dataset for conservation."""

    scores: list[float]
    labels: list[int]


# Tensor type aliases
TokenArray = ht.Int[ht.NamedArray, "batch position"]
LogitArray = ht.Float[ht.NamedArray, "batch position vocab"]
PositionArray = ht.Int[ht.NamedArray, "batch"]
ScoreArray = ht.Float[ht.NamedArray, "batch"]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_nucleotide_token_ids(tokenizer: AutoTokenizer) -> dict[str, int]:
    """Get the token IDs for nucleotides A, C, T, G."""
    nucleotide_ids = {}
    # Use lowercase as this is what the plantcad tokenizer normalizes to first
    for nucleotide in ["a", "c", "t", "g"]:
        token_id = tokenizer.convert_tokens_to_ids(nucleotide)
        if token_id is None:
            raise ValueError(f"Nucleotide '{nucleotide}' not found in tokenizer")
        nucleotide_ids[nucleotide] = int(token_id)

    # Assert that all token IDs are unique
    token_id_values = list(nucleotide_ids.values())
    assert len(token_id_values) == len(set(token_id_values))

    return nucleotide_ids


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve checkpoint path, downloading from HuggingFace if needed."""
    protocol = fsspec.utils.get_protocol(checkpoint_path)
    if protocol != "hf":
        return checkpoint_path

    # Remove protocol prefix to get path
    path = checkpoint_path.removeprefix("hf://")
    # Parse org/repo/path/to/folder format
    path_parts = path.split("/")
    if len(path_parts) >= 2:
        repo_id = "/".join(path_parts[:2])  # org/repo
        folder_path = "/".join(path_parts[2:]) if len(path_parts) > 2 else ""
    else:
        repo_id = path
        folder_path = ""

    # Download to HF cache
    api = HfApi()
    local_path = api.snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{folder_path}/*" if folder_path else "*",
    )

    final_path = os.path.join(local_path, folder_path) if folder_path else local_path
    logger.info(f"Downloaded HF checkpoint to: {final_path}")
    return final_path


def load_eval_dataset(
    dataset_path: str,
    dataset_config: str | None,
    dataset_split: str,
    max_samples: int | None,
    random_seed: int,
) -> Dataset:
    """Load and validate evaluation dataset."""
    logger.info(f"Loading dataset from {dataset_path} (config={dataset_config}, split={dataset_split})")

    dataset = load_dataset(dataset_path, dataset_config, split=dataset_split)

    if max_samples is not None and len(dataset) > max_samples:
        logger.info(f"Downsampling dataset to {max_samples} samples")
        dataset = dataset.shuffle(seed=random_seed)
        dataset = dataset.select(range(max_samples))

    # Validate labels
    labels = np.array(dataset["label"])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Label must be 0 or 1; got unique values: {np.unique(labels)}")

    logger.info(f"Loaded and validated dataset with {len(dataset)} samples")
    return dataset


# -----------------------------------------------------------------------------
# Evaluation functions
# -----------------------------------------------------------------------------
def create_alternate_sequences(
    tokens: TokenArray,
    nucleotide_positions: PositionArray,
    nucleotide_token_ids: list[int],
) -> tuple[ht.Int[ht.NamedArray, "batch position variant"], ht.Int[ht.NamedArray, " batch"]]:
    """Create 4 alternative sequences with different nucleotides at target positions.

    For each input sequence, creates 4 variants where the nucleotide at the target position
    is replaced with each of the 4 possible nucleotides (A, C, T, G). Also determines which
    variant corresponds to the original sequence.

    Args:
        tokens: Token sequences with axes (Batch, Position)
        nucleotide_positions: Target positions for substitution with axes (Batch,)
        nucleotide_token_ids: List of 4 nucleotide token IDs (typically [A, C, T, G])

    Returns:
        tuple containing:
        - alt_sequences: Array with axes (Batch, Position, Variant) where Variant has size 4,
          containing the original sequences with nucleotides substituted at target positions
        - ref_indexes: Array with axes (Batch,) indicating which variant index corresponds
          to the original nucleotide for each sequence

    Raises:
        ValueError: If any target position contains a nucleotide not in nucleotide_token_ids
    """
    Batch, Position = tokens.axes

    # Create a new axis for the 4 nucleotide variants
    Variant = hax.Axis("variant", 4)

    # Create array of nucleotide token IDs to substitute
    nucleotide_ids = hax.named(jnp.array(nucleotide_token_ids), Variant)

    # Broadcast tokens to include the variant dimension: (Batch, Position) -> (Batch, Position, Variant)
    tokens_expanded = hax.broadcast_to(tokens, (Batch, Position, Variant))

    # Create position mask for target positions
    position_indices = hax.broadcast_axis(hax.arange(Position), Batch)  # (Position,) -> (Batch, Position)
    target_positions = hax.broadcast_axis(nucleotide_positions, Position)  # (Batch,) -> (Batch, Position)
    position_mask = position_indices == target_positions
    assert position_mask.axes == (Batch, Position)

    # Apply substitutions using where: replace tokens at target positions with nucleotide variants
    alt = hax.where(position_mask, nucleotide_ids, tokens_expanded)
    assert alt.axes == (Batch, Position, Variant)

    # Determine what index across the Variant axis corresponds to the "ref" sequence;
    # This should be which variant sequence has a token id equal to the actual token id in tokens_expanded
    ref_mask = alt[Position, nucleotide_positions] == tokens_expanded[Position, nucleotide_positions]
    assert ref_mask.axes == (Batch, Variant)
    ref_cts = hax.sum(ref_mask, axis=Variant)
    assert 0 <= ref_cts.max().item() <= 1
    if (invalid := ref_cts == 0).any().item():
        pos = nucleotide_positions[Batch, invalid]
        tok = tokens_expanded[Batch, invalid][Position, pos]
        raise ValueError(
            "Found invalid sequences in batch with OOV nucleotides at target positions;\n"
            f"Target positions: {pos.array} \n"
            f"Valid nucleotide token IDs: {nucleotide_token_ids} \n"
            f"Invalid tokens: {tok.array} "
        )
    ref = hax.argmax(ref_mask, axis=Variant)
    assert ref.axes == (Batch,)

    return alt, ref


def compute_sequence_logprob(
    logits: LogitArray,
    tokens: TokenArray,
) -> ScoreArray:
    """Compute log probabilities for token sequences.

    Sums log probabilities of predicting each next token in the sequence.

    Args:
        logits: Model logits with axes (Batch, Position, Vocab)
        tokens: Token sequences with axes (Batch, Position)

    Returns:
        Log probabilities with axes (Batch,), representing total sequence
        likelihood under a causal language model
    """
    Batch, Position, Vocab = logits.axes

    # Compute log probabilities of *all* tokens
    log_probs = hax.nn.log_softmax(logits, axis=Vocab)
    assert log_probs.axes == (Batch, Position, Vocab)

    # Align log probabilities to their corresponding true tokens, i.e. shift
    # next token logits from model one token to the right (first token is ignored)
    aligned_log_probs = log_probs[Position, :-1]
    aligned_tokens = tokens[Position, 1:]
    AlignedPosition = aligned_log_probs.resolve_axis(Position.name)
    assert aligned_log_probs.axes == (Batch, AlignedPosition, Vocab)
    assert aligned_tokens.axes == (Batch, AlignedPosition)

    # Select the log probabilities of only the true tokens for each sequence
    # and sum them to get a log probability by sequence
    token_log_probs = hax.take(aligned_log_probs, index=aligned_tokens, axis=Vocab)
    assert token_log_probs.axes == (Batch, AlignedPosition)
    sequence_log_probs = hax.sum(token_log_probs.astype(jnp.float32), axis=AlignedPosition)
    assert sequence_log_probs.axes == (Batch,)

    return sequence_log_probs


def compute_causal_conservation(
    tokens: TokenArray,
    logit_function: Callable[[TokenArray], LogitArray],
    nucleotide_positions: PositionArray,
    nucleotide_token_ids: list[int],
) -> ScoreArray:
    """Compute conservation scores using causal language modeling.

    Creates 4 nucleotide variants at target positions and compares their
    relative likelihoods to measure evolutionary conservation.

    Args:
        tokens: Token sequences with axes (Batch, Position)
        logit_function: Function that takes tokens and returns model logits
        nucleotide_positions: Target positions for substitution with axes (Batch,)
        nucleotide_token_ids: List of nucleotide token IDs (typically A, C, T, G)

    Returns:
        Conservation scores with axes (Batch,), representing the log probability
        of the original sequence relative to nucleotide variants
    """
    Batch, Position = tokens.axes

    # Create alternate/variant sequences with all possible nucleotides at target positions
    alt_sequences, ref_indexes = create_alternate_sequences(tokens, nucleotide_positions, nucleotide_token_ids)
    Variant = alt_sequences.resolve_axis("variant")
    assert alt_sequences.axes == (Batch, Position, Variant)
    assert ref_indexes.axes == (Batch,)

    # Stack variant and batch dimensions for model input
    # TODO: can this be done with Axis objects instead?
    batch_alt_sequences = hax.rearrange(
        alt_sequences, "{batch position variant} -> (batch_variant: batch variant) position"
    )
    VariantBatch = batch_alt_sequences.resolve_axis("batch_variant")

    # Run inference for all reference/alternate sequences
    logits = logit_function(batch_alt_sequences)
    # Always promote to full precision for zero-shot evaluation
    logits = logits.astype(jnp.float32)
    Vocab = logits.resolve_axis("vocab")
    assert logits.axes == (VariantBatch, Position, Vocab)

    # Compute marginal log probabilities for all variant sequences
    alternate_log_probs = compute_sequence_logprob(logits, batch_alt_sequences)

    # Unstack the variant dimension in order to renormalize the true (i.e. ref), input
    # sequences by the log probability of each variant sequence
    sequence_log_probs = hax.rearrange(
        alternate_log_probs, "(batch_variant: batch variant) -> batch variant", batch=Batch, variant=Variant
    )
    marginal_log_probs = hax.nn.log_softmax(sequence_log_probs, axis=Variant)
    assert marginal_log_probs.axes == (Batch, Variant)

    # Select the log probabilities of the true (i.e. ref) sequences
    ref_log_probs = marginal_log_probs[Variant, ref_indexes]
    assert ref_log_probs.axes == (Batch,)

    return ref_log_probs


def score_eval_dataset(
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    logit_function: Callable[[TokenArray], LogitArray],
    batch_size: int = 32,
    log_progress: bool = True,
) -> ConservationResult:
    """Score evaluation dataset based on zero-shot conservation prediction."""

    # Get nucleotide token mappings from tokenizer
    nucleotide_token_ids = list(get_nucleotide_token_ids(tokenizer).values())  # [id_A, id_C, id_T, id_G]

    all_scores = []
    all_labels = []
    total_processed = 0

    # Set batches for processing
    batches = eval_dataset.with_format(None).batch(batch_size=batch_size)
    total_batches = len(batches)
    progress_interval = max(1, total_batches // 20)  # Every 5%
    if log_progress:
        logger.info(f"Processing {len(eval_dataset)} samples in {total_batches} batches (batch_size={batch_size})")

    for batch_index, batch_data in enumerate(batches):
        # Tokenize sequences
        sequences = batch_data["seq"]
        labels = batch_data["label"]
        pos = batch_data["pos"]
        assert isinstance(sequences, list)
        assert isinstance(labels, list)
        assert isinstance(pos, list)

        # Tokenize and convert to JAX arrays
        tokenized = tokenizer(sequences, padding=False, add_special_tokens=False, truncation=False, return_tensors="np")
        tokens = hax.named(tokenized["input_ids"], ("batch", "position"))
        nucleotide_positions = hax.named(pos, ("batch",))

        # Compute conservation score for each example in batch
        scores = compute_causal_conservation(
            tokens=tokens,
            logit_function=logit_function,
            nucleotide_positions=nucleotide_positions,
            nucleotide_token_ids=nucleotide_token_ids,
        )
        assert len(scores.array) == len(labels)

        # Aggregate scores and labels
        all_scores.extend(scores.tolist())
        all_labels.extend(labels)
        total_processed += len(sequences)

        # Log progress every 5% of batches
        if log_progress and (batch_index % progress_interval == 0 or batch_index == total_batches - 1):
            progress_pct = ((batch_index + 1) / total_batches) * 100
            logger.info(
                f"Progress: {batch_index + 1}/{total_batches} batches ({progress_pct:.1f}%) - "
                f"{len(all_scores)} scores generated"
            )

    return ConservationResult(scores=all_scores, labels=all_labels)


def evaluate_conservation_scores(scores: ConservationResult) -> dict[str, float]:
    """Calculate ROC AUC and other metrics from scores."""
    from sklearn.metrics import roc_auc_score

    if len(scores.scores) == 0:
        raise ValueError("No valid conservation scores found")

    n_unmasked_total = len(scores.scores)
    valid_mask = ~np.isnan(scores.scores)
    filtered_scores = np.array(scores.scores)[valid_mask]
    filtered_labels = np.array(scores.labels)[valid_mask]

    if len(filtered_scores) == 0:
        raise ValueError("No valid (non-NaN) scores found after filtering")

    if (n_filtered := n_unmasked_total - len(filtered_scores)) > 0:
        logger.info(f"Filtered out {n_filtered} samples with NaN scores")

    # Compute metrics
    roc_auc = roc_auc_score(filtered_labels, filtered_scores)
    n_positive = filtered_labels.sum()
    n_total = len(filtered_labels)

    results = {
        "roc_auc": roc_auc,
        "n_total": n_total,
        "n_positive": int(n_positive),
        "n_negative": n_total - int(n_positive),
        "balance": n_positive / n_total,
    }

    return results


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
def run_plantcad_evaluation(config: DnaEvalConfig) -> dict:
    """Run PlantCAD model evaluation on conservation prediction."""
    logger.info("🧬 PlantCAD Model Evaluation")
    logger.info("=" * 64)
    logger.info(f"Checkpoint:   {config.checkpoint_path}")
    logger.info(f"Dataset:      {config.dataset_path}")
    logger.info(f"Config:       {config.dataset_config}")
    logger.info(f"Batch size:   {config.batch_size}")
    logger.info(f"Max samples:  {config.max_samples}")
    logger.info(f"Device:       {config.device}")
    logger.info(f"Dtype:        {config.dtype}")
    logger.info("=" * 64)

    # Resolve and load checkpoint
    logger.info("Resolving checkpoint path...")
    resolved_checkpoint = resolve_checkpoint_path(config.checkpoint_path)
    logger.info(f"Resolved to: {resolved_checkpoint}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_checkpoint)
    logger.info(f"Tokenizer vocab size: {len(tokenizer.vocab)}")

    # Load model
    logger.info("Loading model...")
    torch_dtype = getattr(torch, config.dtype) if config.dtype else None
    model = AutoModelForCausalLM.from_pretrained(
        resolved_checkpoint,
        trust_remote_code=True,
        **(dict(torch_dtype=torch_dtype) if torch_dtype is not None else {}),
    )
    model = model.to(device=config.device, dtype=torch_dtype)
    model.eval()
    logger.info("Model loaded successfully")

    # Create logit function
    def logit_function(tokens: TokenArray) -> LogitArray:
        Batch, Position = tokens.axes
        token_array = np.array(tokens.array, dtype=np.int64)
        input_ids = torch.from_numpy(token_array).to(config.device)

        with torch.inference_mode():
            outputs = model(input_ids)
            logits = outputs.logits.float().cpu().numpy()
            return hax.named(logits, (Batch, Position, "vocab"))

    # Load dataset
    dataset = load_eval_dataset(
        config.dataset_path,
        config.dataset_config,
        config.dataset_split,
        config.max_samples,
        config.random_seed,
    )

    # Generate scores
    logger.info("Generating conservation scores...")
    result = score_eval_dataset(
        tokenizer=tokenizer,
        logit_function=logit_function,
        eval_dataset=dataset,
        batch_size=config.batch_size,
    )
    logger.info(f"Generated {len(result.scores)} conservation scores")

    # Evaluate and save results
    metrics = evaluate_conservation_scores(result)

    results_file = os.path.join(config.output_path, "results.json")

    output_data = {
        "checkpoint_path": config.checkpoint_path,
        "resolved_checkpoint": resolved_checkpoint,
        "dataset_path": config.dataset_path,
        "dataset_config": config.dataset_config,
        "metrics": metrics,
    }

    with fsspec.open(results_file, "w") as f:
        json.dump(output_data, f, indent=2, cls=CustomJsonEncoder)

    logger.info(f"Results saved to: {results_file}")

    return output_data


# -----------------------------------------------------------------------------
# Experiment setup
# -----------------------------------------------------------------------------
checkpoint_path = "hf://plantcad/marin_exp1729__pcv1_600m_c512__checkpoints/local_store/checkpoints/plantcad-train-600m-r16-a1bc43/hf/step-26782"
evaluation_step = ExecutorStep(
    name="plantcad-eval",
    fn=run_plantcad_evaluation,
    config=DnaEvalConfig(
        checkpoint_path=versioned(checkpoint_path),
        output_path=this_output_path(),
    ),
)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(steps=[evaluation_step])
