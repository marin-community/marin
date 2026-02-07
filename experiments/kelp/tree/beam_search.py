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

"""Beam search for tree diffusion inference.

Following Algorithm 2 of Tree Diffusion (Kapur et al., 2024): maintain a beam
of candidate programs, expand each by generating model-predicted edits, score
by cumulative log-probability, and prune to the top-k at each depth step.

The reverse (denoising) process iteratively refines programs:
1. For each beam candidate, run the AR model to generate an edit
2. Apply the edit to produce a new candidate program
3. Score candidates by cumulative log-probability
4. Keep the top beam_size candidates
5. Repeat until max_depth or convergence
"""

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.tree.constrained_decoding import (
    apply_bracket_constraints,
    validate_edit,
)
from experiments.kelp.tree.edit_model import EditModelParams, forward
from experiments.kelp.tree.mutation import Mutation
from experiments.kelp.tree.tokenizer import TreeDiffusionTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BeamCandidate:
    """A candidate program in the beam."""

    source: str
    """Current program source."""

    score: float
    """Cumulative log-probability of edits applied so far."""

    depth: int
    """Number of edits applied."""

    edits: tuple[Mutation, ...]
    """History of edits applied."""


def _ar_generate_tokens(
    params: EditModelParams,
    context_token_ids: list[int],
    cfg: TreeDiffusionConfig,
    tokenizer: TreeDiffusionTokenizer,
    key: PRNGKeyArray,
    temperature: float = 1.0,
    max_new_tokens: int = 64,
) -> tuple[list[int], float]:
    """Autoregressively generate edit tokens (POS + replacement + EOS).

    Starting from [context..., SOS], generates tokens one at a time until
    EOS is produced or max_new_tokens is reached.

    Args:
        params: Model parameters.
        context_token_ids: Tokenized context program.
        cfg: Model configuration.
        tokenizer: Tokenizer.
        key: PRNG key.
        temperature: Sampling temperature.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Tuple of (generated_token_ids, cumulative_log_prob). The generated
        tokens start after SOS and include everything up to (and including)
        EOS. If generation is truncated, EOS is not included.
    """
    # Start with context + SOS.
    current_ids = context_token_ids + [tokenizer.sos_token_id]
    generated: list[int] = []
    total_log_prob = 0.0

    for step in range(max_new_tokens):
        key, sample_key = jax.random.split(key)

        # Run full forward pass (no KV cache for simplicity).
        input_ids = jnp.array([current_ids], dtype=jnp.int32)
        logits = forward(params, input_ids, cfg)

        # Take logits at the last position.
        next_logits = logits[0, -1, :]

        # Apply bracket constraints on replacement tokens (not on POS token).
        if step > 0:
            partial = tokenizer.decode_source(generated[1:] if len(generated) > 1 else [])
            next_logits = apply_bracket_constraints(next_logits, partial, tokenizer)

        # Temperature scaling.
        if temperature > 0:
            scaled_logits = next_logits / temperature
        else:
            scaled_logits = next_logits

        # Sample.
        log_probs = jax.nn.log_softmax(scaled_logits)

        if temperature > 0:
            next_token = jax.random.categorical(sample_key, scaled_logits)
        else:
            next_token = jnp.argmax(scaled_logits)

        next_token_id = int(next_token)
        token_log_prob = float(log_probs[next_token_id])
        total_log_prob += token_log_prob

        generated.append(next_token_id)
        current_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

    return generated, total_log_prob


def generate_edit(
    params: EditModelParams,
    source: str,
    cfg: TreeDiffusionConfig,
    tokenizer: TreeDiffusionTokenizer,
    key: PRNGKeyArray,
    temperature: float = 1.0,
    max_replacement_len: int = 64,
) -> tuple[Mutation | None, float]:
    """Generate a single edit from the model via autoregressive decoding.

    Encodes the source program as context, then autoregressively generates:
    1. A position token (WHERE to edit)
    2. Replacement tokens (WHAT to insert)
    3. EOS token

    The edit is validated against the source to ensure it produces valid Python.

    Args:
        params: Model parameters.
        source: Current program source.
        cfg: Model configuration.
        tokenizer: Tokenizer.
        key: PRNG key.
        temperature: Sampling temperature (0 = greedy).
        max_replacement_len: Maximum replacement length in tokens.

    Returns:
        Tuple of (mutation, log_probability). Returns (None, -inf) if the
        generated edit is invalid or decoding fails.
    """
    context_tokens = tokenizer.encode_source(source)

    # Cap context to avoid exceeding max_seq_len.
    max_context = cfg.max_seq_len - max_replacement_len - 3  # SOS + POS + EOS
    if len(context_tokens) > max_context:
        context_tokens = context_tokens[:max_context]

    generated, log_prob = _ar_generate_tokens(
        params=params,
        context_token_ids=context_tokens,
        cfg=cfg,
        tokenizer=tokenizer,
        key=key,
        temperature=temperature,
        max_new_tokens=max_replacement_len + 2,  # POS + replacement + EOS
    )

    if not generated:
        return None, float("-inf")

    # First generated token should be a position token.
    pos_token_id = generated[0]
    if not tokenizer.is_position_token(pos_token_id):
        return None, float("-inf")

    edit_position = tokenizer.position_from_token(pos_token_id)

    # Remaining tokens (excluding EOS) are the replacement.
    replacement_tokens = []
    for tid in generated[1:]:
        if tid == tokenizer.eos_token_id:
            break
        replacement_tokens.append(tid)

    replacement_source = tokenizer.decode_source(replacement_tokens)

    # Find the end of the original span at the edit position.
    # Use the AST to find the node boundary.
    original_span_end = _find_span_end(source, edit_position)
    if original_span_end is None:
        return None, float("-inf")

    original = source[edit_position:original_span_end]

    mutation = Mutation(
        start=edit_position,
        end=original_span_end,
        replacement=replacement_source,
        node_type="unknown",
        original=original,
    )

    if validate_edit(source, mutation):
        return mutation, log_prob

    return None, float("-inf")


def _find_span_end(source: str, start_offset: int) -> int | None:
    """Find the end of the AST node at the given character offset.

    Walks the AST to find the innermost node whose start position matches
    the offset, and returns its end position.

    Returns None if no matching node is found.
    """
    import ast as _ast

    from experiments.kelp.tree.mutation import _node_source_span
    from experiments.kelp.tree.subtree_bank import EXTRACTABLE_TYPES

    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return None

    best_end = None
    best_size = float("inf")

    for node in _ast.walk(tree):
        type_name = type(node).__name__
        if type_name not in EXTRACTABLE_TYPES:
            continue

        span = _node_source_span(source, node)
        if span is None:
            continue

        node_start, node_end = span
        if node_start == start_offset:
            # Prefer the smallest (most specific) node.
            size = node_end - node_start
            if size < best_size:
                best_end = node_end
                best_size = size

    return best_end


def beam_search(
    params: EditModelParams,
    initial_programs: list[str],
    cfg: TreeDiffusionConfig,
    tokenizer: TreeDiffusionTokenizer,
    key: PRNGKeyArray,
    beam_size: int = 16,
    expansions_per_beam: int = 3,
    max_depth: int = 30,
    temperature: float = 1.0,
) -> list[BeamCandidate]:
    """Run beam search to refine programs through iterative edits.

    Following the paper's reverse process: maintain a beam of candidate
    programs, expand each by sampling model-predicted edits, score by
    cumulative log-probability, and keep the top-k.

    Args:
        params: Trained model parameters.
        initial_programs: Starting programs (corrupted/noisy).
        cfg: Model configuration.
        tokenizer: Tokenizer.
        key: PRNG key.
        beam_size: Number of candidates to keep at each step.
        expansions_per_beam: Number of edits to try per candidate.
        max_depth: Maximum number of edit steps.
        temperature: Sampling temperature.

    Returns:
        List of BeamCandidate sorted by score (best first), length <= beam_size.
    """
    # Initialize beam from starting programs.
    beam: list[BeamCandidate] = [
        BeamCandidate(source=prog, score=0.0, depth=0, edits=()) for prog in initial_programs[:beam_size]
    ]

    for depth in range(max_depth):
        key, step_key = jax.random.split(key)
        expansion_keys = jax.random.split(step_key, len(beam) * expansions_per_beam)

        candidates: list[BeamCandidate] = []
        # Keep current beam members as candidates (they may be better than expansions).
        candidates.extend(beam)

        key_idx = 0
        for parent in beam:
            for _ in range(expansions_per_beam):
                mutation, log_prob = generate_edit(
                    params=params,
                    source=parent.source,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    key=expansion_keys[key_idx],
                    temperature=temperature,
                )
                key_idx += 1

                if mutation is None:
                    continue

                new_source = mutation.apply(parent.source)
                candidates.append(
                    BeamCandidate(
                        source=new_source,
                        score=parent.score + log_prob,
                        depth=parent.depth + 1,
                        edits=parent.edits + (mutation,),
                    )
                )

        # Deduplicate by source (keep highest score).
        seen: dict[str, BeamCandidate] = {}
        for c in candidates:
            if c.source not in seen or c.score > seen[c.source].score:
                seen[c.source] = c

        # Sort by score (highest first) and prune to beam_size.
        beam = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:beam_size]

        if not beam:
            break

        logger.debug(f"Beam search depth={depth}: {len(beam)} candidates, " f"best_score={beam[0].score:.4f}")

        # Early stopping: if no new edits were applied, all candidates are stale.
        all_unchanged = all(c.depth <= depth for c in beam)
        if all_unchanged and depth > 0:
            logger.debug("Beam search converged (no new edits applied)")
            break

    return beam


def best_of_n(
    params: EditModelParams,
    source: str,
    cfg: TreeDiffusionConfig,
    tokenizer: TreeDiffusionTokenizer,
    key: PRNGKeyArray,
    n: int = 16,
    max_depth: int = 30,
    temperature: float = 1.0,
) -> list[BeamCandidate]:
    """Best-of-N sampling: run N independent rollouts, return all results.

    A simpler alternative to beam search that generates N independent
    edit sequences and returns them sorted by score. Each rollout greedily
    applies one edit at a time until max_depth or failure.

    Args:
        params: Trained model parameters.
        source: Starting program (corrupted/noisy).
        cfg: Model configuration.
        tokenizer: Tokenizer.
        key: PRNG key.
        n: Number of independent rollouts.
        max_depth: Maximum edits per rollout.
        temperature: Sampling temperature.

    Returns:
        List of BeamCandidate sorted by score (best first).
    """
    rollout_keys = jax.random.split(key, n)
    results: list[BeamCandidate] = []

    for i in range(n):
        candidate = BeamCandidate(source=source, score=0.0, depth=0, edits=())
        rollout_key = rollout_keys[i]

        for step in range(max_depth):
            rollout_key, step_key = jax.random.split(rollout_key)

            mutation, log_prob = generate_edit(
                params=params,
                source=candidate.source,
                cfg=cfg,
                tokenizer=tokenizer,
                key=step_key,
                temperature=temperature,
            )

            if mutation is None:
                break

            new_source = mutation.apply(candidate.source)
            candidate = BeamCandidate(
                source=new_source,
                score=candidate.score + log_prob,
                depth=candidate.depth + 1,
                edits=candidate.edits + (mutation,),
            )

        results.append(candidate)

    # Deduplicate and sort.
    seen: dict[str, BeamCandidate] = {}
    for c in results:
        if c.source not in seen or c.score > seen[c.source].score:
            seen[c.source] = c

    return sorted(seen.values(), key=lambda c: c.score, reverse=True)
