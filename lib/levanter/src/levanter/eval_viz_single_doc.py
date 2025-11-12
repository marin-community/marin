"""
Utilities to visualize per-token log probabilities for the single training document
selected by SpliceSingleDocumentLMConfig. Writes an HTML similar to viz_logprobs.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, Union

import jax
import jax.numpy as jnp

import haliax as hax
from haliax.partitioning import ResourceMapping, set_mesh

import levanter
from levanter.analysis.visualization import visualize_log_probs
from levanter.data.splice_dataset import SpliceSingleDocumentLMConfig
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfigBase
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.models.loss import next_token_loss
from levanter.utils.hf_utils import HfTokenizer


@dataclass
class VizSingleDocConfig:
    """Configuration for visualizing log probabilities on the single training document.

    Attributes:
        output_prefix: If provided, write HTMLs under this fsspec path prefix (e.g., GCS).
                       If None, write to local "analysis" directory.
        filename_prefix: Prefix for the HTML filename (before step number and .html).
        max_tokens: If provided, truncate the document to this many tokens for speed.
        include_argmax: Whether to compute and include argmax tokens in tooltips.
        verbose: Print basic diagnostics on each invocation.
    """

    output_prefix: Optional[str] = None
    filename_prefix: str = "viz_single_doc"
    max_tokens: Optional[int] = None
    include_argmax: bool = True
    verbose: bool = False


def _decode_tokens_pretty(tok: HfTokenizer, ids: jnp.ndarray) -> list[str]:
    # Prefer convert_ids_to_tokens for faithful token boundaries if available
    arr = list(map(int, jax.device_get(ids)))
    if hasattr(tok, "convert_ids_to_tokens"):
        tokens = tok.convert_ids_to_tokens(arr)
        # Clean up BPE markers for readable output
        cleaned = []
        for t in tokens:
            t_str = str(t)
            # Replace Ġ (U+0120, BPE space marker) with actual space
            t_str = t_str.replace('Ġ', ' ')
            # Replace Ċ (U+010A, BPE newline marker) with actual newline
            t_str = t_str.replace('Ċ', '\n')
            cleaned.append(t_str)
        return cleaned
    else:
        # Fallback: return token IDs as strings to preserve alignment
        return [str(i) for i in arr]


def viz_single_doc_callback(
    cfg: VizSingleDocConfig,
    tokenizer: HfTokenizer,
    axis_resources: ResourceMapping,
    mp,
    data_config: Union[LMMixtureDatasetConfig, SingleDatasetLMConfigBase, SpliceSingleDocumentLMConfig],
    *,
    device_mesh=None,
):
    """Returns a training hook that writes an HTML of per-token log-probabilities for
    the training document chosen by SpliceSingleDocumentLMConfig.

    If the current data config is not SpliceSingleDocumentLMConfig, this is a no-op.
    """

    doc_tokens_state: Optional[jnp.ndarray] = None
    doc_info_state: Optional[dict] = None

    def _maybe_get_doc_tokens(model_Pos) -> Optional[jnp.ndarray]:
        nonlocal doc_tokens_state
        nonlocal doc_info_state
        if doc_tokens_state is not None:
            return doc_tokens_state
        if not isinstance(data_config, SpliceSingleDocumentLMConfig):
            return None
        try:
            caches = data_config.build_caches("train", monitors=False)
            if hasattr(data_config, "_select_doc_tokens_and_index"):
                arr, idx = data_config._select_doc_tokens_and_index(caches, model_Pos)  # type: ignore[attr-defined]
            else:
                arr = data_config._select_doc_tokens(caches, model_Pos)
                idx = getattr(data_config, "doc_index", None)
            doc = jnp.asarray(arr, dtype=jnp.int32)
            if cfg.max_tokens is not None:
                doc = doc[: int(cfg.max_tokens)]
            doc_tokens_state = doc
            doc_info_state = {
                "dataset_name": getattr(data_config, "dataset_name", None),
                "doc_index": int(idx) if idx is not None else -1,
                "length": int(doc.shape[0]),
            }
            return doc
        except Exception:
            return None

    def _compute_logprobs_for_doc(model: LmHeadModel, doc_tokens_1d: jnp.ndarray):
        # Use the model's configured Pos (e.g., 4096) to avoid flash-attention block constraints.
        # Left-align the document tokens and pad the tail; mask out padded positions from loss.
        Pos = model.Pos
        S = int(Pos.size)
        L = int(doc_tokens_1d.shape[0])

        # Truncate if the document is longer than Pos
        doc_use = doc_tokens_1d[:S]
        L_eff = min(L, S)

        # Build tokens of length S with left-aligned doc tokens
        tokens_full = jnp.zeros((S,), dtype=jnp.int32)
        tokens_full = tokens_full.at[:L_eff].set(doc_use)
        tokens_named = hax.named(tokens_full, Pos)

        # Loss mask: only positions within the doc contribute (next-token, so up to L_eff-1)
        lm_mask = jnp.zeros((S,), dtype=jnp.int32)
        if L_eff >= 2:
            lm_mask = lm_mask.at[: L_eff - 1].set(1)
        loss_named = hax.named(lm_mask, Pos)

        # Build causal example with explicit mask
        ex = LmExample.causal(tokens_named, loss_mask=loss_named)

        def _compute():
            m = model
            if mp is not None:
                m = mp.cast_to_compute(m)
            activations = m.activations(ex.tokens, ex.attn_mask)
            logits = hax.dot(activations, m.get_lm_head(), axis=m.Embed)
            nll = next_token_loss(Pos=Pos, Vocab=m.Vocab, logits=logits, true_ids=ex.tokens, loss_mask=ex.loss_mask, reduction=None)
            logprobs = -nll
            # Shift by 1 to align each token's displayed log-prob with the token predicted at that position
            logprobs = hax.roll(logprobs, 1, Pos)
            if cfg.include_argmax:
                logits = hax.roll(logits, 1, Pos)
                argmax_ids = hax.argmax(logits, axis=m.Vocab)
            else:
                argmax_ids = None

            # Return trimmed arrays to the original document length for visualization
            # (return plain JAX arrays to avoid needing NamedArray slicing here)
            lp = (logprobs.array if hasattr(logprobs, "array") else logprobs)
            lp = lp[:L_eff]
            if argmax_ids is not None:
                am = (argmax_ids.array if hasattr(argmax_ids, "array") else argmax_ids)
                am = am[:L_eff]
            else:
                am = None
            return lp, am

        if device_mesh is not None:
            with set_mesh(device_mesh), hax.axis_mapping(axis_resources):
                return _compute()
        else:
            with hax.axis_mapping(axis_resources):
                return _compute()

    def _write_html(step_num: int, tokens_1d: jnp.ndarray, logprobs_named, argmax_named):
        # Prepare decoded tokens and numpy arrays for visualization
        L = int(tokens_1d.shape[0])
        token_ids = jax.device_get(tokens_1d)
        logprobs = jax.device_get(logprobs_named.array if hasattr(logprobs_named, "array") else logprobs_named)
        # Ensure numeric dtype for HTML formatting
        logprobs = np.asarray(logprobs, dtype=float)
        if logprobs.ndim == 1:
            logprobs = logprobs.reshape(1, L)
        if cfg.include_argmax and argmax_named is not None:
            argmax = jax.device_get(argmax_named.array if hasattr(argmax_named, "array") else argmax_named)
            if argmax.ndim == 1:
                argmax = argmax.reshape(1, L)
            # Decode argmax token ids to readable tokens for tooltips
            argmax_toks = [_decode_tokens_pretty(tokenizer, jnp.asarray(argmax[0]))]
        else:
            argmax_toks = None

        tokens_str = [_decode_tokens_pretty(tokenizer, jnp.asarray(token_ids))]

        # Determine output path
        # Use a stable prefix and append the step number so we get a new file per callback
        name = f"{cfg.filename_prefix}_{step_num}.html"
        if cfg.output_prefix:
            if cfg.output_prefix.endswith("/"):
                path = cfg.output_prefix + name
            else:
                path = cfg.output_prefix + "/" + name
        else:
            # local analysis directory
            path = "analysis/" + name

        # Write the HTML visualization
        visualize_log_probs(tokens_str, logprobs, path, argmaxes=argmax_toks)

        # Log as artifact if a tracker is active
        try:
            _trk = levanter.tracker
            _trk.current_tracker().log_artifact(path, name=name, type="viz_single_doc")
        except Exception:
            pass

    def cb(step, force: bool = False):
        # Skip at step 0 unless forced (avoid uninitialized runs)
        if step.step == 0 and not force:
            return

        model = step.eval_model
        doc = _maybe_get_doc_tokens(model.Pos)
        if doc is None:
            return

        logprobs_named, argmax_named = _compute_logprobs_for_doc(model, doc)
        _write_html(int(step.step), doc, logprobs_named, argmax_named)

    return cb
