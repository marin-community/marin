# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute Patel et al. (arXiv 2605.18607) proxy metrics on OT Agent trace evaluations.

Same eval_model x trace_model grid as ``aggregate_ot_analysis_masked.py``, but
instead of loss summaries it computes proxy metrics from the saved top-k
next-token distributions, aggregated over assistant (expert) tokens only.

A proxy metric is a weighted average over the expert trajectory of a token-level
core metric (paper Eq. 1):

    Phi(M; x, y) = sum_t s * m_t * w_t / sum_t w_t

We compute the paper's two single-metric winners plus the cross-entropy baseline:

  - ``freq_top5``    : core = top-5 accuracy 1[y_t in top-5], weight = freq(y_t)
  - ``invfreq_top1`` : core = top-1 accuracy 1[y_t in top-1], weight = 1 - freq(y_t)
  - ``ce_uniform``   : core = cross-entropy -log p(y_t), weight = 1 (sign -1)

``freq(token)`` is the unigram frequency over the assistant tokens of every
trace corpus (the expert-trajectory corpus, paper Appendix A.1), built in a
first pass before the per-pair aggregation.

Alignment: ``losses[t]`` and ``top_k_token_ids[t]`` both describe the
distribution over ``token_ids[t+1]``, so the expert token at prediction step t
is ``y = token_ids[t+1]``. The assistant mask is reconstructed by re-applying
the chat template (as in ``aggregate_ot_analysis_masked.py``) and shifted
``mask[1:] + [0]`` so ``mask[t]`` marks whether the predicted token is an
assistant token.

Usage:
    uv run lib/marin/src/marin/run/ray_run.py -- python experiments/agent_scaling/aggregate_ot_proxies.py
"""

import json
import os
from collections import Counter
from dataclasses import dataclass

import fsspec
from transformers import AutoTokenizer

from zephyr import Dataset, ZephyrContext

from experiments.agent_scaling.download_ot_traces import BASE_MODEL, build_steps as build_trace_steps
from experiments.agent_scaling.ot_trace_logprobs import build_steps as build_eval_steps
from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from marin.execution.executor import ExecutorStep, executor_main, output_path_of
from marin.execution.types import this_output_path
from marin.utils import fsspec_exists

# Width of the top-k accuracy core metric for the frequency-weighted proxy.
TOP5_K = 5


@dataclass(frozen=True)
class AggregateConfig:
    eval_step_paths: dict[str, str]  # dir name -> resolved logprobs step path
    trace_step_paths: dict[str, str]  # dir name -> resolved traces step path
    output_path: str


def aggregate_results(config: AggregateConfig):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.chat_template = QWEN_3_CHAT_TEMPLATE

    def assistant_encoding(conversations: list[dict]) -> dict:
        """Re-tokenize one conversation; returns ``input_ids`` and ``assistant_masks``."""
        return tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )

    def get_reward(trace: dict) -> float:
        if "result" in trace and trace["result"] == "1.0":
            return 1.0
        elif "result" in trace and trace["result"] == "0.0":
            return -1.0
        elif "result" in trace and trace["result"] == "AgentTimeoutError":
            return -1.0
        else:
            return 0.0

    # === Pass 1: unigram frequency over assistant tokens of every trace corpus. ===

    def trace_assistant_counts(trace_model: str) -> dict[int, int]:
        tr_path = os.path.join(config.trace_step_paths[trace_model], "traces.jsonl.gz")
        counts: Counter[int] = Counter()
        with fsspec.open(tr_path, "rt", compression="gzip") as f:
            for line in f:
                enc = assistant_encoding(json.loads(line)["conversations"])
                counts.update(
                    tid for tid, m in zip(enc["input_ids"], enc["assistant_masks"], strict=True) if m
                )
        return dict(counts)

    freq_ctx = ZephyrContext(name="ot-proxies-freq")
    freq_pipeline = Dataset.from_list([{"trace_model": tm} for tm in config.trace_step_paths]).map(
        lambda p: trace_assistant_counts(p["trace_model"])
    )
    total_counts: Counter[int] = Counter()
    for partial in freq_ctx.execute(freq_pipeline).results:
        total_counts.update(partial)
    total_tokens = sum(total_counts.values())
    freq = {tid: count / total_tokens for tid, count in total_counts.items()}

    # === Pass 2: per (eval_model, trace_model) proxy aggregates. ===

    def compute_pair_stats(eval_model: str, trace_model: str) -> dict:
        lp_path = os.path.join(config.eval_step_paths[eval_model], trace_model, "outputs.jsonl.gz")
        tr_path = os.path.join(config.trace_step_paths[trace_model], "traces.jsonl.gz")

        success_path = lp_path + ".SUCCESS"
        if not fsspec_exists(success_path):
            raise FileNotFoundError(f"No SUCCESS marker at {success_path}")

        metric_names = (
            "freq_top5",
            "invfreq_top1",
            "ce_uniform",
            "reward",
            "has_reward",
            "n_masked_tokens",
            "n_total_tokens",
        )
        sums = dict.fromkeys(metric_names, 0.0)
        counts = dict.fromkeys(metric_names, 0)
        with fsspec.open(lp_path, "rt", compression="gzip") as lp_f, fsspec.open(
            tr_path, "rt", compression="gzip"
        ) as tr_f:
            for logprob_line, trace_line in zip(lp_f, tr_f, strict=True):
                logprob = json.loads(logprob_line)
                trace = json.loads(trace_line)
                token_ids = logprob["token_ids"]
                losses = logprob["losses"]
                top_k_ids = logprob["top_k_token_ids"]

                # Assistant mask, truncated to the saved (max_eval_length) span and
                # shifted so mask[t] marks whether the token predicted at step t
                # (token_ids[t+1]) is an assistant token.
                mask = assistant_encoding(trace["conversations"])["assistant_masks"]
                mask = mask[: len(losses)]
                mask = mask[1:] + [0]

                # Proxy aggregation (Eq. 1) over assistant prediction steps.
                num_freq_top5 = denom_freq = 0.0
                num_invfreq_top1 = denom_invfreq = 0.0
                sum_ce = 0.0
                n_masked = 0
                for t in range(len(losses) - 1):
                    if not mask[t]:
                        continue
                    expert = token_ids[t + 1]
                    w = freq[expert]
                    num_freq_top5 += w * (1.0 if expert in top_k_ids[t][:TOP5_K] else 0.0)
                    denom_freq += w
                    num_invfreq_top1 += (1.0 - w) * (1.0 if expert == top_k_ids[t][0] else 0.0)
                    denom_invfreq += 1.0 - w
                    sum_ce += losses[t]
                    n_masked += 1

                row = {
                    # nan (not 0.0) when no assistant token falls in the span, so
                    # the per-pair mean below skips the trace rather than biasing it.
                    "freq_top5": num_freq_top5 / denom_freq if n_masked else float("nan"),
                    "invfreq_top1": num_invfreq_top1 / denom_invfreq if n_masked else float("nan"),
                    "ce_uniform": -sum_ce / n_masked if n_masked else float("nan"),
                    "reward": get_reward(trace),
                    "has_reward": float("result" in trace),
                    "n_masked_tokens": n_masked,
                    "n_total_tokens": len(token_ids),
                }
                for name, value in row.items():
                    if value == value:
                        sums[name] += value
                        counts[name] += 1

        return {name: sums[name] / counts[name] if counts[name] else float("nan") for name in metric_names}

    eval_models = config.eval_step_paths.keys()
    trace_models = config.trace_step_paths.keys()
    pairs = [{"eval_model": em, "trace_model": tm} for em in eval_models for tm in trace_models]

    ctx = ZephyrContext(name="aggregate-ot-proxies")
    pipeline = Dataset.from_list(pairs).map(lambda p: compute_pair_stats(p["eval_model"], p["trace_model"]))
    results = ctx.execute(pipeline).results

    output_file = os.path.join(config.output_path, "results.jsonl.gz")
    with fsspec.open(output_file, "wt", compression="gzip") as f:
        for pair, result in zip(pairs, results, strict=True):
            row = {**pair, **result}
            f.write(json.dumps(row) + "\n")


def build_aggregate_step() -> ExecutorStep:
    eval_steps_dict = build_eval_steps()
    trace_steps_dict = build_trace_steps()

    eval_step_paths = {
        eval_dir: output_path_of(step_info["logprobs"]) for eval_dir, step_info in eval_steps_dict.items()
    }
    trace_step_paths = {
        trace_dir: output_path_of(step_info["traces"]) for trace_dir, step_info in trace_steps_dict.items()
    }

    return ExecutorStep(
        name="agent_scaling/aggregate_ot_proxies",
        fn=aggregate_results,
        config=AggregateConfig(
            eval_step_paths=eval_step_paths,
            trace_step_paths=trace_step_paths,
            output_path=this_output_path(),
        ),
    )


if __name__ == "__main__":
    executor_main(steps=[build_aggregate_step()], description="Proxy metrics on OT Agent traces.")
