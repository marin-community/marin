# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proxy metrics on OT Agent traces, with tool-call tokens masked out of the weighting.

Identical to ``aggregate_ot_proxies.py`` except for the set of positions the
proxy is aggregated over: here tokens inside ``<tool_call>...</tool_call>`` spans
are excluded (their weight ``w_t`` is set to 0), so the proxy is computed over
the agent's natural-language reasoning only, not the structured tool-call tokens.

Tool-call spans are detected from the special tokens in the saved ``token_ids``
(the same scan as ``aggregate_ot_analysis_masked_tools.py``); the aggregation
mask is ``assistant AND NOT tool_call``.

``freq()`` is left unchanged from ``aggregate_ot_proxies.py`` (built over all
assistant tokens), so this script is a clean ablation of it: only the
aggregation mask differs, isolating the effect of dropping tool-call tokens.

See ``aggregate_ot_proxies.py`` for the proxy definitions and alignment notes.

Usage:
    uv run lib/marin/src/marin/run/ray_run.py -- python experiments/agent_scaling/aggregate_ot_proxies_no_tools.py
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

# Qwen3 special tokens delimiting a tool call.
TOOL_CALL_START = 151657  # <tool_call>
TOOL_CALL_END = 151658  # </tool_call>


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

    def get_tool_call_mask(token_ids: list[int]) -> list[int]:
        """1 for tokens inside <tool_call>...</tool_call> spans (delimiters included)."""
        mask = [0] * len(token_ids)
        inside = False
        for i, tid in enumerate(token_ids):
            if tid == TOOL_CALL_START:
                inside = True
            if inside:
                mask[i] = 1
            if tid == TOOL_CALL_END:
                inside = False
        return mask

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
    # Unchanged from aggregate_ot_proxies.py: freq is over all assistant tokens
    # (tool-call tokens included), so this script differs only in the aggregation mask.

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

    freq_ctx = ZephyrContext(name="ot-proxies-no-tools-freq")
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

                # Aggregation mask: assistant tokens with <tool_call>...</tool_call>
                # spans removed, then shifted so mask[t] marks whether the token
                # predicted at step t (token_ids[t+1]) is an in-scope token.
                a_mask = assistant_encoding(trace["conversations"])["assistant_masks"][: len(losses)]
                tc_mask = get_tool_call_mask(token_ids)
                mask = [1 if a and not tc else 0 for a, tc in zip(a_mask, tc_mask, strict=True)]
                mask = mask[1:] + [0]

                # Proxy aggregation (Eq. 1) over assistant, non-tool-call prediction steps.
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
                    # nan (not 0.0) when no in-scope token falls in the span, so
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

    ctx = ZephyrContext(name="aggregate-ot-proxies-no-tools")
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
        name="agent_scaling/aggregate_ot_proxies_no_tools",
        fn=aggregate_results,
        config=AggregateConfig(
            eval_step_paths=eval_step_paths,
            trace_step_paths=trace_step_paths,
            output_path=this_output_path(),
        ),
    )


if __name__ == "__main__":
    executor_main(steps=[build_aggregate_step()], description="Proxy metrics on OT Agent traces (tool calls masked out).")
