import html
import os
import re
from functools import partial
from typing import Any, List, Optional

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from levanter.callbacks import StepInfo
from levanter.data import DataLoader
from levanter.utils.hf_utils import HfTokenizer


def visualize_log_probs(
    tokens: List[List[str]], log_probs: np.ndarray, output_path: str, argmaxes: Optional[List[List[Any]]] = None
):
    """
    Visualizes token log probabilities by embedding data attributes in spans and computing
    their color using JavaScript. Optionally, if argmaxes is provided, each span's title will include
    the argmax value.

    Args:
        tokens: List of list of tokens for each document.
        log_probs: Array of log probabilities, shape (num_docs, num_tokens).
        output_path: Path to write the HTML file to. Fsspec paths are supported.
        argmaxes: Optional list of list of argmax values corresponding to each token.
    """
    html = (
        "<html><head><meta charset='utf-8'><style>"
        ".lp span { font-family: monospace; padding: 0.2em; border-radius: 2px; }"
        "</style></head><body><div class='lp'>"
    )
    for doc_idx, (doc, logpdoc) in enumerate(zip(tokens, log_probs)):
        for i, (token, lp) in enumerate(zip(doc, logpdoc)):
            if argmaxes is not None:
                argmax_val = _escape(argmaxes[doc_idx][i]).replace("'", "&apos;")
                html += f"<span data-lp='{lp:.3f}' data-argmax='{argmax_val}'>{_escape(token)}</span>"
            else:
                html += f"<span data-lp='{lp:.3f}'>{_escape(token)}</span>"
        html += "<br><br>"
    html += (
        "</div>"
        "<script>"
        "const SCALE = 5;\n"
        "document.querySelectorAll('.lp span').forEach(el => {\n"
        "  let lp = parseFloat(el.getAttribute('data-lp'));\n"
        "  clamp_lp = Math.max(lp, -10);\n"
        "  const norm = (clamp_lp + 10) / 10;\n"
        "  const hue = 60 * norm;\n"
        "  const alpha = (1 - Math.exp(lp / SCALE)).toFixed(2);\n"
        "  el.style.background = `hsla(${hue}, 100%, 50%, ${alpha})`;\n"
        "  const argmax = el.getAttribute('data-argmax');\n"
        "  if (argmax !== null) {\n"
        "    el.title = `lp: ${lp.toFixed(3)} | argmax: ${argmax}`;\n"
        "  } else {\n"
        "    el.title = lp.toFixed(3);\n"
        "  }"
        "});"
        "</script>"
        "</body></html>"
    )

    with fsspec.open(output_path, "w") as f:
        f.write(html)


def _escape(s: str) -> str:
    out = html.escape(s, quote=False)
    # we want newlines and other special characters to be visible
    # we also want strings of spaces to be visible
    out = out.replace("\n", "⏎").replace("\t", "⇥")
    out = re.sub(r"  +", lambda m: " " + "␣" * (len(m.group(0)) - 1), out)
    return out


def compute_and_visualize_log_probs(path: str, model, tokenizer, log_prob_fn, test_data, max_docs=128):
    """
    Compute and visualize log probabilities for a given model and dataset.
    Args:
        path:  Path to write the HTML file to. Fsspec paths are supported.
        model:  Model to use for computing log probabilities.
        tokenizer:  Tokenizer to use for decoding tokens.
        log_prob_fn:  Function to compute log probabilities. Can return log probabilities and argmaxes or just log probabilities.
        test_data: Test data to use for computing log probabilities.
        max_docs: Maximum number of documents to visualize.

    Returns:

    """
    log_probs = []
    targets = []
    argmaxes: list = []
    for batch in test_data:
        out = log_prob_fn(model, batch)
        if len(out) == 2:
            b_logprobs, b_argmaxes = out
            log_probs.append(b_logprobs)
            argmaxes.append(b_argmaxes)
        else:
            b_logprobs = out
            log_probs.append(b_logprobs)

        targets.append(batch)

        # TODO: haliax-ify?
        if len(targets) * b_logprobs.shape[0] >= max_docs:
            break
    log_probs = _concatenate(log_probs)
    targets = _concatenate([t.tokens.array for t in targets])
    if argmaxes:
        argmaxes_array = _concatenate(argmaxes)
    else:
        argmaxes_array = None
    # gather the log probs and targets
    # TODO: is this still necessary?
    (targets, log_probs, argmaxes_array) = multihost_utils.process_allgather(
        (targets, log_probs, argmaxes_array), tiled=True
    )
    log_probs = log_probs[:max_docs]
    targets = targets[:max_docs]
    targets = np.array(targets)
    tokens = [_decode_tokens_pretty(tokenizer, t) for t in targets]
    if argmaxes_array is not None:
        argmaxes_array = argmaxes_array[:max_docs]
        argmaxes_array = [_decode_tokens_pretty(tokenizer, a) for a in argmaxes_array]
    log_probs = np.array(log_probs)
    visualize_log_probs(tokens, log_probs, path, argmaxes=argmaxes_array)


def visualize_log_prob_diff(
    tokens: List[List[str]],
    log_probs_a: np.ndarray,
    log_probs_b: np.ndarray,
    output_path: str,
    diff_threshold: float = 5.0,
):
    """
    Visualizes the difference in token log probabilities between two models by embedding
    data attributes in spans and computing their color and border styles using JavaScript.

    In the generated HTML:
      - The **background color** indicates the difference between model1 and model2 log probabilities.
          - **Blue** for positive differences.
          - **Orange** for negative differences.
          - Near-zero differences are nearly transparent.
      - The **overline** (top border) represents model1's absolute log probability using the absolute color scale.
      - The **underline** (bottom border) represents model2's absolute log probability using the same scale.
      - Extra vertical spacing is added to prevent the border lines from bleeding into adjacent rows.
      - Hover over a token to see a tooltip with the difference and both models’ log probabilities.

    Args:
        tokens: List of list of tokens for each document.
        log_probs_a: Array of log probabilities from model A, shape (num_docs, num_tokens).
        log_probs_b: Array of log probabilities from model B, shape (num_docs, num_tokens).
        output_path: Path to write the HTML file to. Fsspec paths are supported.
        diff_threshold: Maximum absolute difference to consider; differences beyond this are clamped.
    """
    html_str = (
        "<html><head><meta charset='utf-8'><style>"
        ".lp span { font-family: monospace; padding: 0.2em; border-radius: 2px; display: inline-block; margin: 0.2em"
        " 0; }"
        "</style></head><body>"
        # Quick description at the top of the page
        "<div style='font-family: sans-serif; margin-bottom: 1em;'>"
        "<h2>Log Probability Difference Visualization</h2>"
        "<p>"
        "Each token's <strong>background color</strong> indicates the difference between "
        "model1 and model2 log probabilities (<em>blue</em> for positive, <em>orange</em> for negative differences). "
        "An <strong>overline</strong> (top border) shows model1's absolute log probability and an "
        "<strong>underline</strong> (bottom border) shows model2's absolute log probability, both using the same"
        " absolute color scale. "
        "Extra spacing between tokens prevents the border lines from bleeding into adjacent rows. "
        "Hover over a token to see exact values."
        "</p>"
        "</div>"
        "<div class='lp'>"
    )
    for doc, lp_a_doc, lp_b_doc in zip(tokens, log_probs_a, log_probs_b):
        for token, lp_a, lp_b in zip(doc, lp_a_doc, lp_b_doc):
            diff = lp_a - lp_b
            html_str += (
                f"<span data-diff='{diff:.3f}' data-lp1='{lp_a:.3f}' data-lp2='{lp_b:.3f}'>{_escape(token)}</span>"
            )
        html_str += "<br><br>"
    html_str += (
        "</div>"
        "<script>\n"
        "const SCALE = 3;\n"
        f"const threshold = {diff_threshold};\n"
        "\n"
        "// Helper function to compute border thickness from a log probability.\n"
        "function thicknessFromLP(lp) {\n"
        "  lp = Math.max(lp, -10);\n"
        "  const norm = (lp + 10) / 10;\n"
        "  return (1 + (1 - norm) * 4).toFixed(1) + 'px';\n"
        "}\n"
        "\n"
        "// Helper function that computes a color from an absolute log probability using the same scale\n"
        "function absoluteColor(lp) {\n"
        "  let clamped = Math.max(lp, -10);\n"
        "  const norm = (clamped + 10) / 10;\n"
        "  const hue = (60 * norm).toFixed(0); // 0° for very low, 60° for higher probabilities\n"
        "  const alpha = (1 - Math.exp(clamped / SCALE)).toFixed(2);\n"
        "  return `hsla(${hue}, 100%, 50%, ${alpha})`;\n"
        "}\n"
        "\n"
        "document.querySelectorAll('.lp span').forEach(el => {\n"
        "  let diff = parseFloat(el.getAttribute('data-diff'));\n"
        "  diff = Math.max(-threshold, Math.min(threshold, diff));\n"
        "  const alpha = (1 - Math.exp(-Math.abs(diff) / SCALE)).toFixed(2);\n"
        "  let bgColor = 'transparent';\n"
        "  if(diff > 0) {\n"
        "    bgColor = `hsla(219, 70%, 50%, ${alpha})`;\n"
        "  } else if(diff < 0) {\n"
        "    bgColor = `hsla(39, 100%, 50%, ${alpha})`;\n"
        "  }\n"
        "  el.style.background = bgColor;\n"
        "\n"
        "  let lp1 = parseFloat(el.getAttribute('data-lp1'));\n"
        "  let lp2 = parseFloat(el.getAttribute('data-lp2'));\n"
        "\n"
        "  // Compute border thickness based on each model's log probability\n"
        "  let thickness1 = thicknessFromLP(lp1);\n"
        "  let thickness2 = thicknessFromLP(lp2);\n"
        "\n"
        "  // Use the absolute color scale for each model's log probability\n"
        "  let color1 = absoluteColor(lp1);\n"
        "  let color2 = absoluteColor(lp2);\n"
        "\n"
        "  // Set an overline (top border) for model1 and an underline (bottom border) for model2\n"
        "  el.style.borderTop = thickness1 + ' solid ' + color1;\n"
        "  el.style.borderBottom = thickness2 + ' solid ' + color2;\n"
        "\n"
        "  el.title = `diff: ${diff.toFixed(3)} | model1: ${lp1.toFixed(3)}, model2: ${lp2.toFixed(3)}`;\n"
        "});\n"
        "</script>\n"
        "</body></html>"
    )

    with fsspec.open(output_path, "w") as f:
        f.write(html_str)


def compute_and_diff_log_probs(path: str, model, comparison_model, tokenizer, log_prob_fn, test_data, max_docs=128):
    """
    Compute and visualize the difference in log probabilities between two models for a given dataset.

    Args:
        path: Path to write the HTML file to. Fsspec paths are supported.
        model: Model to use for computing log probabilities.
        comparison_model: Model to compare against.
        tokenizer: Tokenizer to use for decoding tokens.
        log_prob_fn: Function to compute log probabilities. Can return log probabilities and argmaxes or just log probabilities.
        test_data: Test data to use for computing log probabilities.
        max_docs: Maximum number of documents to visualize.
    """

    log_probs_a = []
    log_probs_b = []
    targets = []
    for batch in test_data:
        targets.append(batch)

        out = log_prob_fn(model, batch)
        if len(out) == 2:
            b_logprobs, b_argmaxes = out
            log_probs_a.append(b_logprobs)
        else:
            b_logprobs = out
            log_probs_a.append(b_logprobs)

        compare_out = log_prob_fn(comparison_model, batch)
        if len(compare_out) == 2:
            b_logprobs, b_argmaxes = compare_out
            log_probs_b.append(b_logprobs)
        else:
            b_logprobs = compare_out
            log_probs_b.append(b_logprobs)

        if len(targets) * b_logprobs.shape[0] >= max_docs:
            break

    log_probs_a = _concatenate(log_probs_a)
    log_probs_b = _concatenate(log_probs_b)
    log_probs_a = np.array(log_probs_a[:max_docs])
    log_probs_b = np.array(log_probs_b[:max_docs])

    targets = _concatenate([t.tokens.array for t in targets])
    tokens = [_decode_tokens_pretty(tokenizer, t) for t in targets]
    visualize_log_prob_diff(tokens, log_probs_a, log_probs_b, path)


@partial(jax.jit, out_shardings=None)
def _concatenate(x):
    if isinstance(x[0], np.ndarray):
        return np.concatenate(x)
    return jnp.concatenate(x)


def _decode_tokens_pretty(tok, ids):
    # we want to make sure we don't have any weird characters in the output
    # so we'll decode the tokens and then escape them
    if hasattr(tok, "convert_ids_to_tokens"):
        return [str(t) for t in tok.convert_ids_to_tokens(ids)]
    else:
        return [str(t) for t in tok.decode(ids)]


def cb_compute_and_visualize_log_probs(
    test_data: DataLoader, tokenizer: HfTokenizer, log_prob_fn, html_dir: str, max_docs=128
):
    """
        Computes log probabilities for a dataset and visualizes them using visdom.

        Args:
            test_data (DataLoader): The test data to use.
            tokenizer (HfTokenizer): The tokenizer to use.
            log_prob_fn (function): A function that takes a model and a batch; then returns the log probabilities for each token.
            html_dir (str): The directory where the HTML output will be written.
            max_docs (int): The maximum number of documents to process.

        Returns:
    function: A function that takes a step info and computes and visualizes the log probabilities.
    """

    def compute_and_viz_log_probs(step: StepInfo):
        model = step.eval_model
        os.makedirs(html_dir, exist_ok=True)
        path = os.path.join(html_dir, f"step_{step.step}.html")

        compute_and_visualize_log_probs(path, model, tokenizer, log_prob_fn, test_data, max_docs=max_docs)
        # TODO: convert to generic logging
        import wandb

        wandb.log({"log_probs": wandb.Html(path)}, step=step.step)

    return compute_and_viz_log_probs
