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

import argparse
import os
import re

from datetime import datetime
import plotly.graph_objects as go


def parse_ratio_from_tags_or_name(name: str, tags: list[str]) -> tuple[float, float] | None:
    """Parse (pretraining_ratio, finemath_ratio) from run name or tags.

    Supported patterns (case-sensitive):
      - Tags or name including: dclm-<int>, finemath-<int>
        e.g., dclm-70, finemath-30
      - Name formats used in provided experiments like:
        llama-1b-finemath-control-cpt-0.7-0.3 or ...-cpt-0.70-0.30

    Returns None if not found.
    """

    # Try tags like dclm-70 and finemath-30
    dclm_pct = None
    finemath_pct = None
    for tag in [*tags, name]:
        m_d = re.search(r"(?:^|[^\w])dclm-(\d{1,3})(?:[^\d]|$)", tag)
        m_f = re.search(r"(?:^|[^\w])finemath-(\d{1,3})(?:[^\d]|$)", tag)
        if m_d:
            try:
                dclm_pct = int(m_d.group(1))
            except ValueError:
                pass
        if m_f:
            try:
                finemath_pct = int(m_f.group(1))
            except ValueError:
                pass
    if dclm_pct is not None and finemath_pct is not None:
        total = dclm_pct + finemath_pct
        if total > 0:
            return dclm_pct / 100.0, finemath_pct / 100.0

    # Try name like ...-cpt-0.7-0.3
    m = re.search(r"cpt-([01](?:\.\d+)?)-([01](?:\.\d+)?)", name)
    if m:
        try:
            pre = float(m.group(1))
            fine = float(m.group(2))
            return pre, fine
        except ValueError:
            return None

    return None


def main():
    parser = argparse.ArgumentParser(description="Plot MMLU mathematics metrics vs pretraining ratio from W&B runs.")
    parser.add_argument("--entity", required=True, help="W&B entity/user or team")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument(
        "--api_key", default=os.getenv("WANDB_API_KEY"), help="W&B API key; defaults to env WANDB_API_KEY"
    )
    parser.add_argument(
        "--filter_tag", action="append", default=[], help="Only include runs having this tag (repeatable)"
    )
    parser.add_argument("--name_contains", default=None, help="Only include runs whose name contains this substring")
    parser.add_argument(
        "--save",
        default=None,
        help=(
            "Optional path to save plot (html recommended). If endswith .html saves interactive HTML; "
            "otherwise attempts static image with kaleido."
        ),
    )
    parser.add_argument(
        "--include_control", action="store_true", help="Include control runs even if tags imply MCQ stage"
    )
    parser.add_argument("--log_entity", default=None, help="W&B entity to log the plot run to (defaults to --entity)")
    parser.add_argument("--log_project", default=None, help="W&B project to log the plot run to (defaults to --project)")
    parser.add_argument("--run_name", default=None, help="Name for the W&B plot run")
    parser.add_argument(
        "--metric_name",
        default="choice_logprob_norm",
        choices=["choice_logprob_norm", "acc_norm", "nll", "bpb"],
        help="Metric name to plot",
    )
    parser.add_argument("--metric_pattern", default=None, help="Metric pattern to plot")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("WANDB_API_KEY not provided. Pass --api_key or export WANDB_API_KEY.")

    import wandb

    # Ensure login for both API and logging
    try:
        wandb.login(key=args.api_key, relogin=True)
    except Exception:
        pass

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    # Define metric groups (name, regex patterns list, pretty title, y-axis label)
    if args.metric_pattern is None:
        metric_groups = [
            (
                "mmlu_math_choice_logprob_norm",
                [re.compile(r"^lm_eval/mmlu_.*_mathematics_5shot/choice_logprob_norm$")],
                "MMLU mathematics 5-shot choice_logprob_norm vs pretraining ratio",
                "choice_logprob_norm",
            ),
            (
                "mmlu_math_acc_norm",
                [re.compile(r"^lm_eval/mmlu_.*_mathematics_5shot/acc_norm$")],
                "MMLU mathematics 5-shot acc_norm vs pretraining ratio",
                "acc_norm",
            ),
            (
                "gsm8k_cot acc",
                [re.compile(r"^lm_eval/gsm8k_cot/exact_match,strict-match$")],
                "GSM8K loss 8-shot acc vs pretraining ratio",
                "acc",
            ),
            (
                "math_500_loss_bpb",
                [re.compile(r"^lm_eval/math_500_loss/bpb$")],
                "Math 500 loss bpb vs pretraining ratio",
                "bpb",
            ),
        ]
    else:
        metric_groups = [
            (
                "custom_metric",
                [re.compile(args.metric_pattern)],
                f"Custom metric {args.metric_pattern} vs pretraining ratio",
                args.metric_name,
            )
        ]

    # Log to a new W&B run once
    log_entity = args.log_entity or args.entity
    log_project = args.log_project or args.project
    default_run_name = args.run_name or f"midtrain-metrics-vs-ratio-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    wb_run = wandb.init(
        project=log_project,
        entity=log_entity,
        name=default_run_name,
        tags=["plot", "midtrain", "ratio"],
        config={
            "source_entity": args.entity,
            "source_project": args.project,
            "filters": {
                "name_contains": args.name_contains,
                "filter_tag": args.filter_tag,
            },
        },
        reinit=False,
        allow_val_change=True,
    )

    # Helper to build, log figure and table for one metric group
    def build_and_log(metric_key_prefix: str, patterns: list[re.Pattern[str]], title: str, y_label: str) -> None:
        collected: list[tuple[float, dict[str, float], str]] = []

        for run in runs:
            name: str = run.name or ""
            tags: list[str] = list(run.tags or [])

            if args.name_contains and args.name_contains not in name:
                continue
            if args.filter_tag and not all(t in tags for t in args.filter_tag):
                continue

            ratio = parse_ratio_from_tags_or_name(name, tags)
            if not ratio:
                continue
            pre_ratio, fine_ratio = ratio

            if not args.include_control:
                pass

            history_val: dict[str, float] = {}
            for key, val in (run.summary or {}).items():
                if not isinstance(key, str):
                    continue
                for pat in patterns:
                    if pat.match(key):
                        try:
                            if isinstance(val, int | float):
                                history_val[key] = float(val)
                            elif isinstance(val, dict) and "value" in val:
                                history_val[key] = float(val["value"])  # sometimes scalars stored as {value: x}
                        except Exception:
                            pass

            if not history_val:
                try:
                    _ = run.history(keys=["_step"], pandas=False)
                    all_hist = run.history(samples=20000, pandas=False)
                    last_values: dict[str, float] = {}
                    for row in all_hist:
                        for k, v in row.items():
                            if isinstance(k, str) and any(pat.match(k) for pat in patterns):
                                if isinstance(v, int | float):
                                    last_values[k] = float(v)
                                elif isinstance(v, dict) and "value" in v:
                                    try:
                                        last_values[k] = float(v["value"])
                                    except Exception:
                                        pass
                    history_val = last_values
                except Exception:
                    pass

            if not history_val:
                continue

            collected.append((pre_ratio, history_val, name))

        if not collected:
            print(f"No collected data for {metric_key_prefix}")
            return

        collected.sort(key=lambda x: x[0])

        all_metric_names: list[str] = []
        for _, metrics_map, _ in collected:
            for k in metrics_map.keys():
                if k not in all_metric_names:
                    all_metric_names.append(k)
        all_metric_names.sort()

        ratios: list[float] = [r for r, _, _ in collected]
        series: dict[str, list[float | None]] = {m: [] for m in all_metric_names}
        for _, metrics_map, _ in collected:
            for m in all_metric_names:
                series[m].append(metrics_map.get(m))

        fig = go.Figure()
        for m, ys in series.items():
            if all(v is None for v in ys):
                continue
            fig.add_trace(go.Scatter(x=ratios, y=ys, mode="lines+markers", name=m))

        fig.update_layout(
            title=title,
            xaxis_title="Pretraining ratio (dclm)",
            yaxis_title=y_label,
            template="plotly_white",
            legend=dict(font=dict(size=10)),
        )

        wandb.log({f"{metric_key_prefix}_vs_ratio": fig})

        table = wandb.Table(columns=["pretraining_ratio", "metric", "value", "source_run_name"])
        for ratio_val, metrics_map, src_name in collected:
            for metric_name in all_metric_names:
                v = metrics_map.get(metric_name)
                if v is None:
                    continue
                table.add_data(ratio_val, metric_name, float(v), src_name)
        wandb.log({f"{metric_key_prefix}_vs_ratio_table": table})

    # Build and log for each metric group
    for key_prefix, patterns, title, y_label in metric_groups:
        build_and_log(key_prefix, patterns, title, y_label)

    print(f"Logged plots to W&B: {wb_run.url}")


if __name__ == "__main__":
    main()
