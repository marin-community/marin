# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-pair loss summaries for OT Agent trace evaluations, masked to tool call tokens only.

Same as aggregate_ot_analysis_masked.py but only sums losses over tokens inside
<tool_call>...</tool_call> spans, identified by scanning for the special tokens
in the logprobs output.

Usage:
    uv run lib/marin/src/marin/run/ray_run.py -- python experiments/agent_scaling/aggregate_ot_analysis_masked_tools.py
"""

import json
import os
from dataclasses import dataclass

import fsspec
import pandas as pd

from zephyr import Dataset, ZephyrContext

from experiments.agent_scaling.download_ot_traces import build_steps as build_trace_steps
from experiments.agent_scaling.ot_trace_logprobs import build_steps as build_eval_steps
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.utils import fsspec_exists

@dataclass(frozen=True)
class AggregateConfig:
    eval_step_paths: dict[str, str]  # dir name -> resolved logprobs step path
    trace_step_paths: dict[str, str]  # dir name -> resolved traces step path
    output_path: str


def aggregate_results(config: AggregateConfig):
    TOOL_CALL_START = 151657  # <tool_call>
    TOOL_CALL_END = 151658  # </tool_call>

    def get_tool_call_mask(token_ids: list[int]) -> list[int]:
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

    def compute_pair_stats(eval_model: str, trace_model: str) -> dict:
        lp_path = os.path.join(config.eval_step_paths[eval_model], trace_model, "outputs.jsonl.gz")
        tr_path = os.path.join(config.trace_step_paths[trace_model], "traces.jsonl.gz")

        success_path = lp_path + ".SUCCESS"
        if not fsspec_exists(success_path):
            raise FileNotFoundError(f"No SUCCESS marker at {success_path}")

        logprob_records = []
        with fsspec.open(lp_path, "rt", compression="gzip") as f:
            for line in f:
                logprob_records.append(json.loads(line))

        trace_records = []
        with fsspec.open(tr_path, "rt", compression="gzip") as f:
            for line in f:
                trace_records.append(json.loads(line))

        rows = []
        for logprob, trace in zip(logprob_records, trace_records, strict=True):
            mask = get_tool_call_mask(logprob["token_ids"])
            masked_losses = [l for l, m in zip(logprob["losses"], mask) if m]
            n_masked = len(masked_losses)
            total_masked_loss = sum(masked_losses)
            reward = get_reward(trace)

            row = {
                "reward": reward,
                "has_reward": float("result" in trace),
                "avg_loss": total_masked_loss / n_masked if n_masked > 0 else 0.0,
                "loss": total_masked_loss,
                "weighted_loss": total_masked_loss * reward,
                "weighted_avg_loss": (total_masked_loss * reward) / n_masked if n_masked > 0 else 0.0,
                "n_masked_tokens": n_masked,
                "n_total_tokens": len(logprob["token_ids"]),
            }
            rows.append(row)

        return pd.DataFrame(rows).mean().to_dict()

    eval_models = config.eval_step_paths.keys()
    trace_models = config.trace_step_paths.keys()
    pairs = [{"eval_model": em, "trace_model": tm} for em in eval_models for tm in trace_models]

    ctx = ZephyrContext(name="aggregate-ot-results-masked-tools")
    pipeline = Dataset.from_list(pairs).map(lambda p: compute_pair_stats(p["eval_model"], p["trace_model"]))
    results = list(ctx.execute(pipeline))

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
        name="agent_scaling/aggregate_ot_analysis_masked_tools",
        fn=aggregate_results,
        config=AggregateConfig(
            eval_step_paths=eval_step_paths,
            trace_step_paths=trace_step_paths,
            output_path=this_output_path(),
        ),
    )


if __name__ == "__main__":
    executor_main(steps=[build_aggregate_step()], description="Aggregate OT Agent traces results (tool calls only).")
