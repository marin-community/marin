# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "wandb",
#   "wandb-workspaces",
# ]
# ///
"""Build a W&B report for historical 60M/300M swarm rank crossovers."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import wandb_workspaces.reports.v2 as wr

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "reference_outputs" / "historical_swarm_crossover_wandb_report_20260527"
)
ENTITY = "marin-community"
PROJECT = "marin"
METRIC = "eval/uncheatable_eval/github_python/loss"
RUN_PROGRESS = "run_progress"
GLOBAL_STEP = "global_step"
WANDB_TIMEOUT = 45


@dataclass(frozen=True)
class SwarmSpec:
    """Local metadata source for one historical swarm."""

    key: str
    display_name: str
    csv_path: Path
    source_experiment: str
    report_query: str
    report_filter_note: str


SWARMS = (
    SwarmSpec(
        key="60m",
        display_name="60M/1.2B qsplit240",
        csv_path=SCRIPT_DIR / "two_phase_many.csv",
        source_experiment="pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240",
        report_query="ngd3dm2_qsplit240",
        report_filter_note="Tags include pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240 for core rows.",
    ),
    SwarmSpec(
        key="300m",
        display_name="300M/6B qsplit240",
        csv_path=SCRIPT_DIR / "metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv",
        source_experiment="pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b",
        report_query="ngd3dm2_qsplit2",
        report_filter_note=(
            "W&B display names are shortened to ngd3dm2_qsplit2~<hash>; local source_experiment "
            "remains pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b."
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-step-coverage", type=float, default=0.9)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--refresh", action="store_true", help="Re-query W&B histories even if cached.")
    parser.add_argument("--skip-report", action="store_true", help="Only write local artifacts and W&B analysis run.")
    parser.add_argument("--draft", action="store_true", help="Save the W&B report as a draft.")
    return parser.parse_args()


def load_swarm_rows(spec: SwarmSpec) -> pd.DataFrame:
    """Load completed signal rows for one swarm."""
    frame = pd.read_csv(spec.csv_path)
    if "source_experiment" in frame.columns:
        frame = frame[frame["source_experiment"].astype(str).eq(spec.source_experiment)]
    if "row_kind" in frame.columns:
        frame = frame[frame["row_kind"].astype(str).eq("signal")]
    frame = frame[frame["status"].astype(str).eq("completed")].copy()
    frame = frame[frame["wandb_run_id"].notna()].copy()
    frame["wandb_run_id"] = frame["wandb_run_id"].astype(str)
    return frame.sort_values(["run_id", "run_name"], na_position="last").reset_index(drop=True)


def fetch_history(entity: str, project: str, metric: str, spec_key: str, run_row: dict[str, object]) -> pd.DataFrame:
    """Fetch one run history from W&B."""
    run_id = str(run_row["wandb_run_id"])
    api = wandb.Api(timeout=WANDB_TIMEOUT)
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history(keys=[metric, GLOBAL_STEP, RUN_PROGRESS], samples=10000, pandas=True)
    if history.empty or metric not in history.columns:
        return pd.DataFrame()
    history = history[history[metric].notna()].copy()
    if history.empty:
        return pd.DataFrame()
    history = history[[GLOBAL_STEP, RUN_PROGRESS, metric]].copy()
    history["scale_key"] = spec_key
    history["run_name"] = str(run_row["run_name"])
    history["run_id"] = run_row.get("run_id")
    history["wandb_run_id"] = run_id
    history["wandb_display_name"] = run.name
    return history


def fetch_histories(
    *,
    entity: str,
    project: str,
    metric: str,
    rows_by_scale: dict[str, pd.DataFrame],
    output_path: Path,
    max_workers: int,
    refresh: bool,
) -> pd.DataFrame:
    """Fetch or load cached W&B histories."""
    if output_path.exists() and not refresh:
        return pd.read_csv(output_path)
    frames: list[pd.DataFrame] = []
    for spec in SWARMS:
        rows = rows_by_scale[spec.key]
        run_rows = rows.to_dict(orient="records")
        print(f"Fetching {spec.display_name}: {len(run_rows)} W&B histories", flush=True)
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_history, entity, project, metric, spec.key, row): str(row["wandb_run_id"])
                for row in run_rows
            }
            for index, future in enumerate(cf.as_completed(futures), start=1):
                run_id = futures[future]
                try:
                    frame = future.result()
                except Exception as exc:
                    print(f"warning: failed to fetch {spec.key} {run_id}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                if not frame.empty:
                    frames.append(frame)
                if index % 25 == 0 or index == len(run_rows):
                    print(f"  {spec.key}: {index}/{len(run_rows)}", flush=True)
    if not frames:
        raise RuntimeError("No W&B histories were fetched.")
    histories = pd.concat(frames, ignore_index=True)
    histories.to_csv(output_path, index=False)
    return histories


def rank_pair_flip_fraction(row: pd.Series, final_row: pd.Series) -> float:
    """Fraction of pairwise order relations that differ from final order."""
    present = row.notna() & final_row.notna()
    names = list(row[present].index)
    flips = 0
    total = 0
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            current_delta = row[left] - row[right]
            final_delta = final_row[left] - final_row[right]
            if current_delta == 0 or final_delta == 0:
                continue
            total += 1
            flips += int(np.sign(current_delta) != np.sign(final_delta))
    return float(flips / total) if total else float("nan")


def topk_overlap(row: pd.Series, final_row: pd.Series, k: int) -> float:
    """Top-k overlap with final ranking; lower loss is better."""
    present = row.notna() & final_row.notna()
    if int(present.sum()) < k:
        return float("nan")
    current_top = set(row[present].sort_values().head(k).index)
    final_top = set(final_row[present].sort_values().head(k).index)
    return float(len(current_top & final_top) / k)


def analyze_histories(histories: pd.DataFrame, metric: str, min_step_coverage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rank stability diagnostics by scale and progress."""
    step_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for spec in SWARMS:
        scale_hist = histories[histories["scale_key"].eq(spec.key)].copy()
        pivot = scale_hist.pivot_table(index=GLOBAL_STEP, columns="run_name", values=metric, aggfunc="last")
        if pivot.empty:
            continue
        min_runs = int(np.ceil(min_step_coverage * pivot.shape[1]))
        common = pivot.drop(index=0, errors="ignore").dropna(axis=0, thresh=min_runs)
        common = common.loc[sorted(common.index)]
        if common.empty:
            continue
        final_step = int(common.index.max())
        final_row = common.loc[final_step]
        progress_by_step = scale_hist.groupby(GLOBAL_STEP)[RUN_PROGRESS].median()
        for step, row in common.iterrows():
            present = row.notna() & final_row.notna()
            if int(present.sum()) < 3:
                continue
            current_rank = row[present].rank(ascending=True)
            final_rank = final_row[present].rank(ascending=True)
            spearman = float(current_rank.corr(final_rank, method="pearson"))
            step_rows.append(
                {
                    "scale_key": spec.key,
                    "scale": spec.display_name,
                    "global_step": int(step),
                    "run_progress": float(progress_by_step.loc[step]),
                    "n_runs": int(present.sum()),
                    "rank_spearman_vs_final": spearman,
                    "pair_flip_fraction_vs_final": rank_pair_flip_fraction(row, final_row),
                    "top10_overlap_vs_final": topk_overlap(row, final_row, 10),
                    "top25_overlap_vs_final": topk_overlap(row, final_row, 25),
                    "is_final_step": int(step) == final_step,
                }
            )
        final_values = final_row.dropna().sort_values()
        for rank, (run_name, loss) in enumerate(final_values.items(), start=1):
            final_rows.append(
                {
                    "scale_key": spec.key,
                    "scale": spec.display_name,
                    "final_rank": rank,
                    "run_name": run_name,
                    "final_loss": float(loss),
                    "final_step": final_step,
                }
            )
    return pd.DataFrame(step_rows), pd.DataFrame(final_rows)


def plot_rank_stability(step_stats: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Write static rank-stability plots."""
    paths: dict[str, Path] = {}
    color_by_scale = {"60m": "#377eb8", "300m": "#e41a1c"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for spec in SWARMS:
        scale_stats = step_stats[step_stats["scale_key"].eq(spec.key)]
        color = color_by_scale[spec.key]
        axes[0].plot(
            scale_stats["run_progress"],
            scale_stats["rank_spearman_vs_final"],
            marker="o",
            label=spec.display_name,
            color=color,
        )
        axes[1].plot(
            scale_stats["run_progress"],
            scale_stats["pair_flip_fraction_vs_final"],
            marker="o",
            label=spec.display_name,
            color=color,
        )
    axes[0].set_title("Rank Spearman vs final checkpoint")
    axes[0].set_xlabel("run progress")
    axes[0].set_ylabel("Spearman rho")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Pairwise order flips vs final checkpoint")
    axes[1].set_xlabel("run progress")
    axes[1].set_ylabel("flipped pair fraction")
    axes[1].set_ylim(-0.02, 0.5)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    path = output_dir / "rank_stability_vs_final.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["rank_stability"] = path

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for spec in SWARMS:
        scale_stats = step_stats[step_stats["scale_key"].eq(spec.key)]
        ax.plot(
            scale_stats["run_progress"],
            scale_stats["top10_overlap_vs_final"],
            marker="o",
            label=f"{spec.display_name} top-10",
            color=color_by_scale[spec.key],
        )
        ax.plot(
            scale_stats["run_progress"],
            scale_stats["top25_overlap_vs_final"],
            marker="s",
            linestyle="--",
            label=f"{spec.display_name} top-25",
            color=color_by_scale[spec.key],
        )
    ax.set_title("Top-k overlap with final ranking")
    ax.set_xlabel("run progress")
    ax.set_ylabel("overlap")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    path = output_dir / "topk_overlap_vs_final.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["topk_overlap"] = path
    return paths


def plot_loss_curves(histories: pd.DataFrame, final_rankings: pd.DataFrame, metric: str, output_dir: Path) -> dict[str, Path]:
    """Write loss curve plots for each historical swarm."""
    paths: dict[str, Path] = {}
    for spec in SWARMS:
        scale_hist = histories[histories["scale_key"].eq(spec.key)]
        top_final = set(
            final_rankings[final_rankings["scale_key"].eq(spec.key)].sort_values("final_rank").head(8)["run_name"]
        )
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        for run_name, group in scale_hist.groupby("run_name"):
            group = group.sort_values(RUN_PROGRESS)
            if run_name in top_final:
                ax.plot(group[RUN_PROGRESS], group[metric], linewidth=1.8, alpha=0.95, label=run_name)
            else:
                ax.plot(group[RUN_PROGRESS], group[metric], linewidth=0.5, color="#999999", alpha=0.14)
        ax.set_title(f"{spec.display_name}: {metric}")
        ax.set_xlabel("run progress")
        ax.set_ylabel("loss")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=7, title="final top 8")
        path = output_dir / f"{spec.key}_loss_curves_final_top8.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths[f"{spec.key}_loss_curves"] = path
    return paths


def write_summary(
    output_dir: Path,
    histories: pd.DataFrame,
    step_stats: pd.DataFrame,
    final_rankings: pd.DataFrame,
    rows_by_scale: dict[str, pd.DataFrame],
    report_url: str | None = None,
) -> Path:
    """Write a Markdown summary."""
    lines = [
        "# Historical Swarm Crossover Diagnostics",
        "",
        f"- Metric: `{METRIC}`.",
        f"- W&B project: `{ENTITY}/{PROJECT}`.",
        f"- Report URL: {report_url or 'pending'}",
        "",
        "## Swarm Filters",
        "",
    ]
    for spec in SWARMS:
        lines.extend(
            [
                f"- `{spec.display_name}`: `{len(rows_by_scale[spec.key])}` completed rows from `{spec.csv_path}`.",
                f"  - Local source experiment: `{spec.source_experiment}`.",
                f"  - W&B UI query hint: `{spec.report_query}`.",
                f"  - Note: {spec.report_filter_note}",
            ]
        )
    lines.extend(["", "## Rank Stability", ""])
    for spec in SWARMS:
        scale_stats = step_stats[step_stats["scale_key"].eq(spec.key)]
        first = scale_stats.iloc[0]
        late = scale_stats[scale_stats["run_progress"].ge(0.8)].iloc[0]
        final = scale_stats.iloc[-1]
        lines.extend(
            [
                f"- `{spec.display_name}` histories fetched: `{histories[histories['scale_key'].eq(spec.key)]['wandb_run_id'].nunique()}`.",
                (
                    f"  - First common eval: progress `{first['run_progress']:.3f}`, "
                    f"Spearman vs final `{first['rank_spearman_vs_final']:.3f}`, "
                    f"pair-flip fraction `{first['pair_flip_fraction_vs_final']:.3f}`."
                ),
                (
                    f"  - First eval after 80% progress: progress `{late['run_progress']:.3f}`, "
                    f"Spearman `{late['rank_spearman_vs_final']:.3f}`, "
                    f"pair-flip fraction `{late['pair_flip_fraction_vs_final']:.3f}`."
                ),
                f"  - Final step: `{int(final['global_step'])}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The historical 60M and 300M qsplit240 swarms did not have stable early ordering on this metric. "
                "At early common evals, roughly 40% of pairwise order relations disagree with the final checkpoint. "
                "For 300M, rank stability only becomes strong after about 83% progress."
            ),
            "",
        ]
    )
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path


def log_analysis_run(
    *,
    entity: str,
    project: str,
    output_dir: Path,
    histories: pd.DataFrame,
    step_stats: pd.DataFrame,
    final_rankings: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> tuple[str, str, str]:
    """Create a W&B run containing the computed diagnostics."""
    run_name = f"historical_swarm_crossover_diagnostics_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="analysis",
        tags=["swarm-crossover", "qsplit240", "60m", "300m", "agent-generated"],
        config={
            "metric": METRIC,
            "source": "experiments/domain_phase_mix/exploratory/two_phase_many/build_historical_swarm_crossover_wandb_report.py",
        },
    )
    assert run is not None
    for row in step_stats.sort_values(["scale_key", "global_step"]).to_dict(orient="records"):
        scale = str(row["scale_key"])
        wandb.log(
            {
                "crossover/progress": row["run_progress"],
                "crossover/global_step": row["global_step"],
                f"crossover/{scale}_rank_spearman_vs_final": row["rank_spearman_vs_final"],
                f"crossover/{scale}_pair_flip_fraction_vs_final": row["pair_flip_fraction_vs_final"],
                f"crossover/{scale}_top10_overlap_vs_final": row["top10_overlap_vs_final"],
                f"crossover/{scale}_top25_overlap_vs_final": row["top25_overlap_vs_final"],
            }
        )
    image_log = {
        "rank_stability_vs_final": wandb.Image(str(plot_paths["rank_stability"])),
        "topk_overlap_vs_final": wandb.Image(str(plot_paths["topk_overlap"])),
    }
    for spec in SWARMS:
        image_log[f"{spec.key}_loss_curves_final_top8"] = wandb.Image(str(plot_paths[f"{spec.key}_loss_curves"]))
    wandb.log(image_log)
    wandb.log(
        {
            "crossover_step_stats": wandb.Table(dataframe=step_stats),
            "crossover_final_rankings": wandb.Table(dataframe=final_rankings),
        }
    )
    artifact = wandb.Artifact("historical_swarm_crossover_diagnostics", type="analysis")
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            artifact.add_file(str(file_path))
    run.log_artifact(artifact)
    run_url = run.url
    run_id = run.id
    wandb.finish()
    return run_id, run_name, run_url


def report_blocks(analysis_run_name: str, run_url: str, step_stats: pd.DataFrame, summary_text: str) -> list[object]:
    """Build W&B report blocks."""
    first_60m = step_stats[step_stats["scale_key"].eq("60m")].iloc[0]
    first_300m = step_stats[step_stats["scale_key"].eq("300m")].iloc[0]
    return [
        wr.H1("Historical qsplit240 Swarm Crossover Diagnostics"),
        wr.MarkdownBlock(
            "\n".join(
                [
                    "This report checks whether early eval-loss ordering in the historical 60M and 300M qsplit240 swarms matched final ordering.",
                    "",
                    f"Analysis run: [{analysis_run_name}]({run_url})",
                    "",
                    "Headline result: historical ordering was not stable early. "
                    f"At the first common eval, 60M had Spearman {first_60m['rank_spearman_vs_final']:.3f} "
                    f"and 300M had Spearman {first_300m['rank_spearman_vs_final']:.3f} versus final rank.",
                ]
            )
        ),
        wr.TableOfContents(),
        wr.H2("Computed diagnostics"),
        wr.PanelGrid(
            runsets=[
                wr.Runset(
                    entity=ENTITY,
                    project=PROJECT,
                    name="analysis run",
                    query=analysis_run_name,
                )
            ],
            panels=[
                wr.MediaBrowser(
                    title="Diagnostic plots",
                    media_keys=[
                        "rank_stability_vs_final",
                        "topk_overlap_vs_final",
                        "60m_loss_curves_final_top8",
                        "300m_loss_curves_final_top8",
                    ],
                    mode="grid",
                    num_columns=2,
                ),
                wr.LinePlot(
                    title="Rank Spearman vs final",
                    x="crossover/progress",
                    y=[
                        "crossover/60m_rank_spearman_vs_final",
                        "crossover/300m_rank_spearman_vs_final",
                    ],
                    range_y=(0, 1),
                    max_runs_to_show=1,
                    smoothing_type="none",
                ),
                wr.LinePlot(
                    title="Pairwise flip fraction vs final",
                    x="crossover/progress",
                    y=[
                        "crossover/60m_pair_flip_fraction_vs_final",
                        "crossover/300m_pair_flip_fraction_vs_final",
                    ],
                    range_y=(0, 0.5),
                    max_runs_to_show=1,
                    smoothing_type="none",
                ),
            ],
        ),
        wr.H2("Raw W&B run panels"),
        wr.MarkdownBlock(
            "The 300M rows are hard to find by the literal source experiment name because W&B shortened display names "
            "to `ngd3dm2_qsplit2~<hash>`. The local CSV remains the source of truth for exact `wandb_run_id`s."
        ),
        wr.PanelGrid(
            runsets=[
                wr.Runset(
                    entity=ENTITY,
                    project=PROJECT,
                    name="60M qsplit240 core",
                    query="ngd3dm2_qsplit240",
                ),
                wr.Runset(
                    entity=ENTITY,
                    project=PROJECT,
                    name="300M qsplit240 core",
                    query="ngd3dm2_qsplit2",
                ),
            ],
            panels=[
                wr.LinePlot(
                    title=METRIC,
                    x=RUN_PROGRESS,
                    y=[METRIC],
                    max_runs_to_show=300,
                    smoothing_type="none",
                    ignore_outliers=True,
                ),
            ],
        ),
        wr.H2("Repro summary"),
        wr.MarkdownBlock(summary_text),
    ]


def create_report(
    *,
    entity: str,
    project: str,
    analysis_run_name: str,
    run_url: str,
    step_stats: pd.DataFrame,
    summary_text: str,
    draft: bool,
) -> str:
    """Create and save the W&B report."""
    report = wr.Report(
        entity=entity,
        project=project,
        title="Historical qsplit240 swarm eval-loss crossovers",
        description="Rank-stability diagnostics for 60M and 300M qsplit240 eval loss histories.",
        width="fluid",
        blocks=report_blocks(analysis_run_name, run_url, step_stats, summary_text),
    )
    report.save(draft=draft)
    return str(report.url)


def main() -> None:
    """Build local artifacts, W&B analysis run, and W&B report."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_scale = {spec.key: load_swarm_rows(spec) for spec in SWARMS}
    (args.output_dir / "source_rows_summary.json").write_text(
        json.dumps({key: len(value) for key, value in rows_by_scale.items()}, indent=2, sort_keys=True)
    )
    histories = fetch_histories(
        entity=args.entity,
        project=args.project,
        metric=args.metric,
        rows_by_scale=rows_by_scale,
        output_path=args.output_dir / "wandb_histories.csv",
        max_workers=args.max_workers,
        refresh=args.refresh,
    )
    step_stats, final_rankings = analyze_histories(histories, args.metric, args.min_step_coverage)
    step_stats.to_csv(args.output_dir / "crossover_step_stats.csv", index=False)
    final_rankings.to_csv(args.output_dir / "final_rankings.csv", index=False)
    plot_paths = {}
    plot_paths.update(plot_rank_stability(step_stats, args.output_dir))
    plot_paths.update(plot_loss_curves(histories, final_rankings, args.metric, args.output_dir))
    summary_path = write_summary(args.output_dir, histories, step_stats, final_rankings, rows_by_scale)
    run_id, run_name, run_url = log_analysis_run(
        entity=args.entity,
        project=args.project,
        output_dir=args.output_dir,
        histories=histories,
        step_stats=step_stats,
        final_rankings=final_rankings,
        plot_paths=plot_paths,
    )
    report_url = None
    if not args.skip_report:
        report_url = create_report(
            entity=args.entity,
            project=args.project,
            analysis_run_name=run_name,
            run_url=run_url,
            step_stats=step_stats,
            summary_text=summary_path.read_text(),
            draft=args.draft,
        )
        write_summary(args.output_dir, histories, step_stats, final_rankings, rows_by_scale, report_url=report_url)
    (args.output_dir / "wandb_outputs.json").write_text(
        json.dumps(
            {
                "analysis_run_id": run_id,
                "analysis_run_name": run_name,
                "analysis_run_url": run_url,
                "report_url": report_url,
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Analysis run: {run_url}", flush=True)
    if report_url is not None:
        print(f"Report: {report_url}", flush=True)
    print(f"Artifacts: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
