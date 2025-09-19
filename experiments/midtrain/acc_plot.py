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

    if args.metric_pattern is None:
        metric_patterns = [
            re.compile(r"^lm_eval/mmlu_.*_mathematics_5shot/" + args.metric_name + "$"),
            re.compile(r"^lm_eval/gsm8k_loss_8shot/" + args.metric_name + "$"),
        ]
    else:
        metric_patterns = [
            re.compile(args.metric_pattern),
        ]

    # Collect: ratio -> { split_metric_name -> value }
    collected: list[tuple[float, dict[str, float], str]] = []  # (pre_ratio, metrics_map, run_name)

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

        # Optionally exclude runs that have mcq in tags unless explicitly included
        if not args.include_control:
            # We don't filter here strictly; user asked to just plot by ratio.
            pass

        # Extract metrics matching the patterns
        history_val: dict[str, float] = {}
        # Prefer run.summary first; if absent, try scan history for last value
        for key, val in (run.summary or {}).items():
            if not isinstance(key, str):
                continue
            for pat in metric_patterns:
                if pat.match(key):
                    try:
                        if isinstance(val, int | float):
                            history_val[key] = float(val)
                        elif isinstance(val, dict) and "value" in val:
                            history_val[key] = float(val["value"])  # sometimes scalars stored as {value: x}
                    except Exception:
                        pass

        # If not found in summary, optionally look in history for the last value
        if not history_val:
            try:
                _ = run.history(keys=["_step"], pandas=False)
                # Fetch full keys separately to minimize payload
                all_hist = run.history(samples=20000, pandas=False)
                last_values: dict[str, float] = {}
                for row in all_hist:
                    for k, v in row.items():
                        if isinstance(k, str) and any(pat.match(k) for pat in metric_patterns):
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
        raise SystemExit("No runs found with parsed ratios and matching MMLU math metrics.")

    # Sort by pretraining ratio ascending
    collected.sort(key=lambda x: x[0])

    # Build consistent metric ordering across runs
    all_metric_names: list[str] = []
    for _, metrics_map, _ in collected:
        for k in metrics_map.keys():
            if k not in all_metric_names:
                all_metric_names.append(k)
    all_metric_names.sort()

    # Prepare data for plotting: for each metric, a series over ratios
    ratios: list[float] = [r for r, _, _ in collected]
    series: dict[str, list[float | None]] = {m: [] for m in all_metric_names}
    for _, metrics_map, _ in collected:
        for m in all_metric_names:
            series[m].append(metrics_map.get(m))

    fig = go.Figure()
    for m, ys in series.items():
        # Plot lines with markers; skip metrics entirely missing
        if all(v is None for v in ys):
            continue
        fig.add_trace(go.Scatter(x=ratios, y=ys, mode="lines+markers", name=m))

    fig.update_layout(
        title="MMLU mathematics 5-shot " + args.metric_name + " vs pretraining ratio",
        xaxis_title="Pretraining ratio (dclm)",
        yaxis_title=args.metric_name,
        template="plotly_white",
        legend=dict(font=dict(size=10)),
    )

    # Log to a new W&B run
    log_entity = args.log_entity or args.entity
    log_project = args.log_project or args.project
    default_run_name = args.run_name or f"mmlu-math-vs-ratio-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    wb_run = wandb.init(
        project=log_project,
        entity=log_entity,
        name=default_run_name,
        tags=["plot", "mmlu", "mathematics", args.metric_name],
        config={
            "source_entity": args.entity,
            "source_project": args.project,
            "filters": {
                "name_contains": args.name_contains,
                "filter_tag": args.filter_tag,
            },
            "num_series": len(all_metric_names),
            "num_points": len(ratios),
        },
        reinit=False,
        allow_val_change=True,
    )

    # Log the interactive figure
    wandb.log({"mmlu_math_vs_ratio": fig})

    # Log a table of the underlying data
    table = wandb.Table(columns=["pretraining_ratio", "metric", "value", "source_run_name"])
    for ratio_val, metrics_map, src_name in collected:
        for metric_name in all_metric_names:
            v = metrics_map.get(metric_name)
            if v is None:
                continue
            table.add_data(ratio_val, metric_name, float(v), src_name)
    wandb.log({"mmlu_math_vs_ratio_table": table})

    print(f"Logged plot to W&B: {wb_run.url}")

    # Optionally save locally
    if args.save:
        out_path = args.save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            if out_path.lower().endswith(".html"):
                fig.write_html(out_path, include_plotlyjs="cdn")
            else:
                # Requires kaleido for static image export
                fig.write_image(out_path, scale=2)
            print(f"Saved plot to {out_path}")
        except Exception as e:
            # Fallback to HTML next to requested path
            fallback = os.path.splitext(out_path)[0] + ".html"
            try:
                fig.write_html(fallback, include_plotlyjs="cdn")
                print(f"Static image save failed ({e}). Wrote HTML to {fallback}")
            except Exception:
                print(f"Failed to save plot locally: {e}")


if __name__ == "__main__":
    main()
