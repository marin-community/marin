# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a dashboard for the completed Grug-MoE mixture scaling tracks.

The non-proportional mixture-generating code is not present in the current
checkout, so this script treats GCS executor metadata as the source of truth:
training configs come from ``gs://marin-us-east5/grug/*/.executor_info`` and
follow-up logprob evals come from ``gs://marin-us-east5/evaluation/grug_logprob``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = Path(
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517"
)

GCS_GRUG_PREFIX = "gs://marin-us-east5/grug"
GCS_EVAL_PREFIX = "gs://marin-us-east5/evaluation/grug_logprob"

TRACKS = ("grug_moe_mix", "grug_moe_mix_v2", "grug_moe_mix_v3", "grug_moe_mix_v4")
TRACK_LABELS = {
    "grug_moe_mix": "Proportional",
    "grug_moe_mix_v2": "v2",
    "grug_moe_mix_v3": "v3",
    "grug_moe_mix_v4": "v4",
}
PATH_ENDPOINT_TRACKS = ("grug_moe_mix_v4",)
PATH_T_BY_SLUG = {
    "t025": 0.25,
    "t050": 0.50,
    "t075": 0.75,
}
PATH_MIXTURE_ORDER = (
    "Proportional",
    "v4 path t=0.25",
    "v4 path t=0.50",
    "v4 path t=0.75",
    "v4",
)
PATH_MIXTURE_COLORS = {
    "Proportional": "#d7191c",
    "v4 path t=0.25": "#fdae61",
    "v4 path t=0.50": "#fee08b",
    "v4 path t=0.75": "#a6d96a",
    "v4": "#1a9641",
}

RUN_RE = re.compile(r"^(grug_moe_mix(?:_v\d+)?)_d(\d+)-([0-9.]+e[+-]\d+)(?:-[0-9a-f]+)?$")
ROOT_RE = re.compile(r"gs://marin-us-east5/grug/(grug_moe_mix(?:_v\d+)?_d\d+-[0-9.]+e[+-]\d+-[0-9a-f]+)/?$")
PATH_RUN_RE = re.compile(r"^(grug_moe_mix_v4_path_r1)_(t\d{3})_d(\d+)-([0-9.]+e[+-]\d+)(?:-[0-9a-f]+)?$")
PATH_ROOT_RE = re.compile(r"gs://marin-us-east5/grug/(grug_moe_mix_v4_path_r1_t\d{3}_d\d+-[0-9.]+e[+-]\d+-[0-9a-f]+)/?$")
TASK_HASH_RE = re.compile(r"-[0-9a-f]{6}$")
NUMERIC_RETRY_RE = re.compile(r"numeric_retry(?P<attempt>\d+)-[0-9a-f]+$")
METRIC_SUFFIX = ",none"


@dataclass(frozen=True)
class GcsObject:
    path: str


def run_text(args: list[str], *, allow_failure: bool = False) -> str:
    """Run a command and return stdout."""
    completed = subprocess.run(args, text=True, capture_output=True, check=False)
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(args)}\nSTDERR:\n{completed.stderr.strip()}"
        )
    return completed.stdout


def gcloud_ls(path: str, *, allow_failure: bool = False) -> list[GcsObject]:
    return [
        GcsObject(line.strip())
        for line in run_text(["gcloud", "storage", "ls", path], allow_failure=allow_failure).splitlines()
        if line.strip()
    ]


def gcloud_cat(path: str, *, allow_failure: bool = False) -> str:
    return run_text(["gcloud", "storage", "cat", path], allow_failure=allow_failure)


def parse_run_key(run_key: str) -> tuple[str, int, str]:
    """Return ``(track, hidden_dim, budget)`` from a run key or prefix."""
    match = RUN_RE.match(run_key)
    if match is None:
        raise ValueError(f"Could not parse run key: {run_key}")
    track, hidden_dim, budget = match.groups()
    if track not in TRACKS:
        raise ValueError(f"Unexpected track: {track}")
    return track, int(hidden_dim), budget


def parse_path_run_key(run_key: str) -> tuple[str, float, int, str]:
    """Return ``(t_slug, t_value, hidden_dim, budget)`` from a path run key."""
    match = PATH_RUN_RE.match(run_key)
    if match is None:
        raise ValueError(f"Could not parse path run key: {run_key}")
    _, t_slug, hidden_dim, budget = match.groups()
    if t_slug not in PATH_T_BY_SLUG:
        raise ValueError(f"Unexpected path t slug: {t_slug}")
    return t_slug, PATH_T_BY_SLUG[t_slug], int(hidden_dim), budget


def scale_label(hidden_dim: int, budget: str) -> str:
    return f"d{hidden_dim}\n{budget}"


def normalize_scale_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize scale key dtypes after CSV roundtrips."""
    if df.empty:
        return df

    out = df.copy()
    if "hidden_dim" in out.columns:
        out["hidden_dim"] = pd.to_numeric(out["hidden_dim"], errors="raise").astype(int)
    if "budget" in out.columns:
        out["budget"] = pd.to_numeric(out["budget"], errors="raise")
    if {"hidden_dim", "budget", "scale_label"}.issubset(out.columns):
        out["scale_label"] = [
            scale_label(int(hidden_dim), f"{float(budget):.2e}")
            for hidden_dim, budget in zip(out["hidden_dim"], out["budget"], strict=True)
        ]
    return out


def domain_family(domain: str) -> str:
    if domain.startswith("dolma3_cc/"):
        return "dolma3_cc"
    if domain.startswith("dolmino_synth_"):
        return "dolmino_synth"
    if domain.startswith("dolmino_"):
        return "dolmino_other"
    if domain.startswith("dolma3_"):
        return "dolma3_other"
    return domain.split("/", maxsplit=1)[0]


def task_group(task_alias: str) -> str:
    if task_alias.startswith("mmlu"):
        return "mmlu"
    if task_alias.startswith(("arc_", "openbookqa", "sciq")):
        return "arc_openbook_sciq"
    if task_alias.startswith(("hellaswag", "swag")):
        return "hellaswag_swag"
    if task_alias.startswith(("boolq", "copa", "csqa", "piqa", "winogrande", "wsc273", "socialiqa")):
        return "commonsense"
    if task_alias.startswith("truthfulqa"):
        return "truthfulqa"
    if task_alias.startswith("logprob_gsm8k"):
        return "gsm8k_logprob"
    if task_alias.startswith("logprob_humaneval"):
        return "humaneval_logprob"
    if task_alias.startswith("medmcqa"):
        return "medmcqa"
    return "other"


def executor_info(path: str) -> dict[str, Any]:
    return json.loads(gcloud_cat(f"{path}/.executor_info"))


def executor_status(path: str) -> str:
    return gcloud_cat(f"{path}/.executor_status", allow_failure=True).strip() or "NO_STATUS"


def collect_training_metadata() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Collect completed training roots and mixture schedules.

    Returns:
        ``runs_df``: one row per completed training root.
        ``weights_df``: one row per domain per track-scale phase.
        ``checkpoint_by_run_id``: mapping from run id to checkpoint root.
    """
    roots = []
    for obj in gcloud_ls(f"{GCS_GRUG_PREFIX}/"):
        match = ROOT_RE.match(obj.path)
        if match is None:
            continue
        root_name = match.group(1)
        try:
            track, hidden_dim, budget = parse_run_key(root_name.rsplit("-", maxsplit=1)[0])
        except ValueError:
            continue
        if track not in TRACKS:
            continue
        roots.append((track, hidden_dim, budget, root_name, obj.path.rstrip("/")))

    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    checkpoint_by_run_id: dict[str, str] = {}

    for track, hidden_dim, budget, root_name, root_path in sorted(roots):
        state = executor_status(root_path)
        if state != "SUCCESS":
            continue
        info = executor_info(root_path)
        config = info["config"]
        run_id = str(config.get("run_id") or root_name.rsplit("-", maxsplit=1)[0])
        weights = config["data"]["train_weights"]
        checkpoint_root = f"{root_path}/checkpoints"
        checkpoint_by_run_id[run_id] = checkpoint_root
        rows.append(
            {
                "track": track,
                "track_label": TRACK_LABELS[track],
                "hidden_dim": hidden_dim,
                "budget": budget,
                "scale_label": scale_label(hidden_dim, budget),
                "root_name": root_name,
                "run_id": run_id,
                "state": state,
                "target_steps": config.get("steps"),
                "checkpoint_root": checkpoint_root,
                "wandb_group": (config.get("tracker") or {}).get("group"),
                "seed": config.get("seed"),
            }
        )

        if isinstance(weights, dict):
            phases = [("constant", 0, weights)]
        elif isinstance(weights, list):
            phases = [
                (f"phase_{idx}", int(boundary), phase_weights) for idx, (boundary, phase_weights) in enumerate(weights)
            ]
        else:
            raise ValueError(f"Unexpected train_weights type for {root_name}: {type(weights)}")

        for phase_name, boundary, phase_weights in phases:
            total = sum(float(value) for value in phase_weights.values())
            for domain, value in phase_weights.items():
                weight_rows.append(
                    {
                        "track": track,
                        "track_label": TRACK_LABELS[track],
                        "hidden_dim": hidden_dim,
                        "budget": budget,
                        "scale_label": scale_label(hidden_dim, budget),
                        "run_id": run_id,
                        "phase": phase_name,
                        "boundary": boundary,
                        "domain": domain,
                        "family": domain_family(domain),
                        "weight": float(value),
                        "phase_weight_sum": total,
                    }
                )

    runs_df = pd.DataFrame(rows)
    weights_df = pd.DataFrame(weight_rows)
    if not runs_df.empty:
        # Some proportional cells have multiple successful retry roots. The
        # dashboard compares logical track-scale cells, so keep one copy per
        # run id and mirror that choice in the weight table.
        runs_df = runs_df.sort_values(["track", "hidden_dim", "budget", "root_name"]).drop_duplicates(
            "run_id", keep="first"
        )
        checkpoint_by_run_id = dict(zip(runs_df["run_id"], runs_df["checkpoint_root"], strict=True))
        weights_df = weights_df.drop_duplicates(["run_id", "phase", "domain"], keep="first")
    return runs_df, weights_df, checkpoint_by_run_id


def collect_path_training_metadata() -> pd.DataFrame:
    """Collect training status for proportional-to-v4 path runs."""
    rows: list[dict[str, Any]] = []
    for obj in gcloud_ls(f"{GCS_GRUG_PREFIX}/"):
        match = PATH_ROOT_RE.match(obj.path)
        if match is None:
            continue
        root_name = match.group(1)
        t_slug, t_value, hidden_dim, budget = parse_path_run_key(root_name.rsplit("-", maxsplit=1)[0])
        root_path = obj.path.rstrip("/")
        state = executor_status(root_path)
        info_text = gcloud_cat(f"{root_path}/.executor_info", allow_failure=True)
        config: dict[str, Any] = {}
        if info_text:
            try:
                config = json.loads(info_text).get("config", {})
            except json.JSONDecodeError:
                config = {}
        rows.append(
            {
                "endpoint_track": "grug_moe_mix_v4",
                "endpoint_label": TRACK_LABELS["grug_moe_mix_v4"],
                "t_slug": t_slug,
                "t": t_value,
                "hidden_dim": hidden_dim,
                "budget": budget,
                "scale_label": scale_label(hidden_dim, budget),
                "root_name": root_name,
                "run_id": str(config.get("run_id") or root_name.rsplit("-", maxsplit=1)[0]),
                "state": state,
                "target_steps": config.get("steps"),
                "checkpoint_root": f"{root_path}/checkpoints",
                "wandb_group": (config.get("tracker") or {}).get("group"),
                "seed": config.get("seed"),
            }
        )

    path_runs_df = pd.DataFrame(rows)
    if path_runs_df.empty:
        return path_runs_df
    return path_runs_df.sort_values(["t", "hidden_dim", "root_name"]).drop_duplicates(
        ["t_slug", "hidden_dim", "budget"], keep="last"
    )


def eval_results_paths() -> list[str]:
    """List per-task Grug logprob result files.

    Top-level combined results are intentionally ignored; per-task child outputs
    have less ambiguity and are the right unit for task coverage accounting.
    """
    paths = []
    for obj in gcloud_ls(f"{GCS_EVAL_PREFIX}/**/results.json"):
        rel = obj.path.split(f"{GCS_EVAL_PREFIX}/", maxsplit=1)[1]
        parent = rel.split("/", maxsplit=1)[0]
        if parent.startswith("grug_moe_mix_v4_path_r1"):
            continue
        if len(rel.split("/")) >= 3:
            paths.append(obj.path)
    return paths


def result_path_attempt_rank(path: str) -> int:
    """Return a monotone retry rank for de-duplicating result files."""
    result_parent = path.rstrip("/").rsplit("/", maxsplit=1)[0].rsplit("/", maxsplit=1)[-1]
    match = NUMERIC_RETRY_RE.match(result_parent)
    if match is None:
        return 0
    return int(match.group("attempt"))


def path_eval_results_paths(parent_run_ids: list[str] | None = None) -> list[str]:
    """List Grug logprob result files for path-test checkpoints, if present."""
    # `gcloud storage ls prefix/**/results.json` has missed nested retry outputs
    # in practice, so explicitly include the direct task-child and one retry
    # level layouts used by the executor.
    if parent_run_ids is None:
        patterns = [
            f"{GCS_EVAL_PREFIX}/grug_moe_mix_v4_path_r1_*/*/results.json",
            f"{GCS_EVAL_PREFIX}/grug_moe_mix_v4_path_r1_*/*/*/results.json",
        ]
    else:
        patterns = []
        for parent_run_id in sorted(set(parent_run_ids)):
            patterns.extend(
                [
                    f"{GCS_EVAL_PREFIX}/{parent_run_id}/*/results.json",
                    f"{GCS_EVAL_PREFIX}/{parent_run_id}/*/*/results.json",
                ]
            )
    paths: dict[str, None] = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(gcloud_ls, pattern, allow_failure=True) for pattern in patterns]
        for future in as_completed(futures):
            for obj in future.result():
                rel = obj.path.split(f"{GCS_EVAL_PREFIX}/", maxsplit=1)[1]
                if len(rel.split("/")) >= 3:
                    paths[obj.path] = None
    return sorted(paths)


def deduplicate_eval_metric_rows(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Keep the latest retry output for each logical task metric cell."""
    if df.empty:
        return df
    out = df.copy()
    out["result_attempt_rank"] = out["result_path"].map(result_path_attempt_rank)
    out = out.sort_values([*key_cols, "result_attempt_rank", "result_path"])
    return out.drop_duplicates(key_cols, keep="last").reset_index(drop=True)


def parse_eval_path(path: str) -> tuple[str, int, str, str]:
    rel = path.split(f"{GCS_EVAL_PREFIX}/", maxsplit=1)[1]
    parent, task_hash, *_ = rel.split("/")
    track, hidden_dim, budget = parse_run_key(parent)
    task_alias = TASK_HASH_RE.sub("", task_hash)
    return track, hidden_dim, budget, task_alias


def parse_path_eval_path(path: str) -> tuple[str, float, int, str, str]:
    rel = path.split(f"{GCS_EVAL_PREFIX}/", maxsplit=1)[1]
    parent, task_hash, *_ = rel.split("/")
    t_slug, t_value, hidden_dim, budget = parse_path_run_key(parent)
    task_alias = TASK_HASH_RE.sub("", task_hash)
    return t_slug, t_value, hidden_dim, budget, task_alias


def aggregate_result_key(task_alias: str, results: dict[str, Any]) -> str:
    if len(results) == 1:
        return next(iter(results))
    candidates = [
        task_alias,
        re.sub(r"_(?:0|5|10)shot$", "", task_alias),
        task_alias.replace("_5shot", ""),
        task_alias.replace("_0shot", ""),
    ]
    for candidate in candidates:
        if candidate in results:
            return candidate
    for key, value in results.items():
        if isinstance(value, dict) and not value.get("alias", "").startswith(" - "):
            return key
    return next(iter(results))


def numeric_metrics(metric_dict: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metric_dict.items():
        if not key.endswith(METRIC_SUFFIX):
            continue
        metric = key[: -len(METRIC_SUFFIX)]
        if metric.endswith("_stderr"):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[metric] = float(value)
    return out


def collect_one_eval(path: str) -> list[dict[str, Any]]:
    track, hidden_dim, budget, task_alias = parse_eval_path(path)
    if track not in TRACKS:
        return []
    data = json.loads(gcloud_cat(path))
    results = data.get("results", {})
    if not isinstance(results, dict) or not results:
        return []
    result_key = aggregate_result_key(task_alias, results)
    metric_dict = results[result_key]
    checkpoint_path = None
    info_path = path.rsplit("/", maxsplit=1)[0] + "/.executor_info"
    info_text = gcloud_cat(info_path, allow_failure=True)
    if info_text:
        try:
            checkpoint_path = json.loads(info_text)["config"].get("checkpoint_path")
        except (KeyError, json.JSONDecodeError):
            checkpoint_path = None
    rows = []
    for metric, value in numeric_metrics(metric_dict).items():
        rows.append(
            {
                "track": track,
                "track_label": TRACK_LABELS[track],
                "hidden_dim": hidden_dim,
                "budget": budget,
                "scale_label": scale_label(hidden_dim, budget),
                "task_alias": task_alias,
                "task_group": task_group(task_alias),
                "result_key": result_key,
                "metric": metric,
                "value": value,
                "result_path": path,
                "checkpoint_path": checkpoint_path,
            }
        )
    return rows


def collect_one_path_eval(path: str) -> list[dict[str, Any]]:
    t_slug, t_value, hidden_dim, budget, task_alias = parse_path_eval_path(path)
    data = json.loads(gcloud_cat(path))
    results = data.get("results", {})
    if not isinstance(results, dict) or not results:
        return []
    result_key = aggregate_result_key(task_alias, results)
    metric_dict = results[result_key]
    checkpoint_path = None
    info_path = path.rsplit("/", maxsplit=1)[0] + "/.executor_info"
    info_text = gcloud_cat(info_path, allow_failure=True)
    if info_text:
        try:
            checkpoint_path = json.loads(info_text)["config"].get("checkpoint_path")
        except (KeyError, json.JSONDecodeError):
            checkpoint_path = None
    rows = []
    for metric, value in numeric_metrics(metric_dict).items():
        rows.append(
            {
                "track": f"grug_moe_mix_v4_path_r1_{t_slug}",
                "track_label": f"v4 path t={t_value:.2f}",
                "endpoint_track": "grug_moe_mix_v4",
                "endpoint_label": TRACK_LABELS["grug_moe_mix_v4"],
                "path_t_slug": t_slug,
                "path_t": t_value,
                "hidden_dim": hidden_dim,
                "budget": budget,
                "scale_label": scale_label(hidden_dim, budget),
                "task_alias": task_alias,
                "task_group": task_group(task_alias),
                "result_key": result_key,
                "metric": metric,
                "value": value,
                "result_path": path,
                "checkpoint_path": checkpoint_path,
            }
        )
    return rows


def collect_eval_metrics() -> pd.DataFrame:
    paths = eval_results_paths()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(collect_one_eval, path) for path in paths]
        for future in as_completed(futures):
            rows.extend(future.result())
    df = pd.DataFrame(rows)
    return deduplicate_eval_metric_rows(df, ["track", "hidden_dim", "budget", "task_alias", "metric"])


def collect_path_eval_metrics(parent_run_ids: list[str] | None = None) -> pd.DataFrame:
    paths = path_eval_results_paths(parent_run_ids)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(collect_one_path_eval, path) for path in paths]
        for future in as_completed(futures):
            rows.extend(future.result())
    df = pd.DataFrame(rows)
    return deduplicate_eval_metric_rows(df, ["track", "hidden_dim", "budget", "task_alias", "metric"])


def preferred_task_values(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Select one oriented smooth metric per task-scale cell."""
    preference = [
        ("choice_prob_norm", 1.0, "choice_prob_norm"),
        ("choice_logprob_norm", 1.0, "choice_logprob_norm"),
        ("choice_logprob", 1.0, "choice_logprob"),
        ("bpb", -1.0, "negative_bpb"),
        ("nll", -1.0, "negative_nll"),
        ("acc_norm", 1.0, "acc_norm"),
        ("acc", 1.0, "acc"),
    ]
    key_cols = ["track", "track_label", "hidden_dim", "budget", "scale_label", "task_alias", "task_group"]
    selected = []
    for key, group in metrics_df.groupby(key_cols, dropna=False):
        metric_values = {str(row.metric): float(row.value) for row in group.itertuples()}
        for metric, direction, preferred_name in preference:
            if metric in metric_values:
                selected.append(
                    {
                        **dict(zip(key_cols, key, strict=True)),
                        "preferred_metric": preferred_name,
                        "raw_metric": metric,
                        "raw_value": metric_values[metric],
                        "oriented_value": direction * metric_values[metric],
                    }
                )
                break
    selected_df = pd.DataFrame(selected)
    if selected_df.empty:
        return selected_df

    cell_cols = ["track", "hidden_dim", "budget"]
    expected_cells = selected_df[cell_cols].drop_duplicates().shape[0]
    cell_counts = selected_df.groupby("task_alias").size()
    counts = selected_df.groupby("task_alias")[cell_cols].nunique()
    common_tasks = counts[(counts["track"] >= len(TRACKS)) & (cell_counts >= expected_cells)].index
    common_df = selected_df[selected_df["task_alias"].isin(common_tasks)].copy()
    common_df["common_task"] = True
    selected_df = selected_df.merge(
        common_df[["track", "hidden_dim", "budget", "task_alias", "common_task"]],
        on=["track", "hidden_dim", "budget", "task_alias"],
        how="left",
    )
    selected_df["common_task"] = selected_df["common_task"].eq(True)
    selected_df["available_cells"] = selected_df["task_alias"].map(cell_counts).astype(int)
    selected_df["expected_cells"] = expected_cells
    selected_df["coverage"] = selected_df["available_cells"].astype(str) + "/" + str(expected_cells)
    return selected_df


def cell_summary(preferred_df: pd.DataFrame) -> pd.DataFrame:
    common = preferred_df[preferred_df["common_task"]].copy()
    if common.empty:
        return pd.DataFrame()
    common["task_mean"] = common.groupby("task_alias")["oriented_value"].transform("mean")
    task_std = common.groupby("task_alias")["oriented_value"].transform("std").replace(0, float("nan"))
    common["task_z"] = (common["oriented_value"] - common["task_mean"]) / task_std

    summary = (
        common.groupby(["track", "track_label", "hidden_dim", "budget", "scale_label"], as_index=False)
        .agg(
            aggregate_z=("task_z", "mean"),
            common_task_count=("task_alias", "nunique"),
            mean_oriented_value=("oriented_value", "mean"),
        )
        .sort_values(["hidden_dim", "aggregate_z"], ascending=[True, False])
    )
    summary["rank_at_scale"] = summary.groupby("hidden_dim")["aggregate_z"].rank(ascending=False, method="min")
    summary["overall_rank"] = summary["aggregate_z"].rank(ascending=False, method="min")
    return summary


def hard_accuracy_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = metrics_df[metrics_df["metric"].isin(["acc_norm", "acc"])].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["metric_priority"] = rows["metric"].map({"acc_norm": 0, "acc": 1})
    rows = rows.sort_values("metric_priority").drop_duplicates(
        ["track", "hidden_dim", "budget", "task_alias"], keep="first"
    )
    return (
        rows.groupby(["track", "track_label", "hidden_dim", "budget", "scale_label"], as_index=False)
        .agg(mean_accuracy=("value", "mean"), accuracy_task_count=("task_alias", "nunique"))
        .sort_values(["hidden_dim", "mean_accuracy"], ascending=[True, False])
    )


def loss_like_task_values(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Select one positive lower-is-better metric per task-scale cell."""
    preference = [
        ("bpb", "identity", "bpb"),
        ("nll", "identity", "nll"),
        ("choice_logprob_norm", "negate", "-choice_logprob_norm"),
        ("choice_logprob", "negate", "-choice_logprob"),
        ("logprob", "negate", "-logprob"),
        ("choice_prob_norm", "one_minus", "1-choice_prob_norm"),
        ("acc_norm", "one_minus", "1-acc_norm"),
        ("acc", "one_minus", "1-acc"),
    ]
    key_cols = ["track", "track_label", "hidden_dim", "budget", "scale_label", "task_alias", "task_group"]
    selected: list[dict[str, Any]] = []
    for key, group in metrics_df.groupby(key_cols, dropna=False):
        metric_values = {str(row.metric): float(row.value) for row in group.itertuples()}
        for metric, transform, loss_metric in preference:
            if metric not in metric_values:
                continue
            raw_value = metric_values[metric]
            if transform == "identity":
                loss_value = raw_value
            elif transform == "negate":
                loss_value = -raw_value
            elif transform == "one_minus":
                loss_value = 1.0 - raw_value
            else:
                raise ValueError(f"Unknown transform: {transform}")
            if math.isfinite(loss_value) and loss_value > 0:
                selected.append(
                    {
                        **dict(zip(key_cols, key, strict=True)),
                        "loss_metric": loss_metric,
                        "raw_metric": metric,
                        "raw_value": raw_value,
                        "loss_value": loss_value,
                    }
                )
                break

    selected_df = pd.DataFrame(selected)
    if selected_df.empty:
        return selected_df
    cell_cols = ["track", "hidden_dim", "budget"]
    expected_cells = selected_df[cell_cols].drop_duplicates().shape[0]
    common_tasks = selected_df.groupby("task_alias").size()
    selected_df["available_cells"] = selected_df["task_alias"].map(common_tasks).astype(int)
    selected_df["expected_cells"] = expected_cells
    selected_df["coverage"] = selected_df["available_cells"].astype(str) + "/" + str(expected_cells)
    selected_df["common_task"] = selected_df["task_alias"].map(common_tasks).eq(expected_cells)
    return selected_df


def power_law_fits(loss_df: pd.DataFrame) -> pd.DataFrame:
    """Fit log10(loss) = intercept + beta log10(training FLOPs)."""
    rows: list[dict[str, Any]] = []
    fit_df = loss_df[loss_df["loss_value"] > 0].copy()
    for (task_alias, track, track_label), group in fit_df.groupby(["task_alias", "track", "track_label"]):
        group = group.sort_values("budget")
        if len(group) < 3:
            continue
        x = pd.to_numeric(group["budget"], errors="coerce").to_numpy(dtype=float)
        y = group["loss_value"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if mask.sum() < 3:
            continue
        beta, intercept = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), deg=1)
        predicted = 10 ** (intercept + beta * np.log10(x[mask]))
        ss_res = float(np.sum((np.log10(y[mask]) - np.log10(predicted)) ** 2))
        ss_tot = float(np.sum((np.log10(y[mask]) - np.mean(np.log10(y[mask]))) ** 2))
        rows.append(
            {
                "task_alias": task_alias,
                "track": track,
                "track_label": track_label,
                "beta": float(beta),
                "intercept": float(intercept),
                "loglog_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "n_points": int(mask.sum()),
                "min_budget": float(x[mask].min()),
                "max_budget": float(x[mask].max()),
            }
        )
    return pd.DataFrame(rows)


def path_t_from_track(track: str) -> float:
    match = re.search(r"_(t\d{3})$", track)
    if match is None:
        raise ValueError(f"Could not infer path t from track: {track}")
    return PATH_T_BY_SLUG[match.group(1)]


def path_task_deltas(preferred_df: pd.DataFrame, path_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build endpoint-path task deltas relative to proportional at the same scale."""
    if preferred_df.empty:
        return pd.DataFrame()

    key_cols = ["hidden_dim", "budget", "scale_label", "task_alias"]
    value_cols = [
        "task_group",
        "preferred_metric",
        "raw_metric",
        "raw_value",
        "oriented_value",
    ]
    baseline = preferred_df[preferred_df["track"].eq("grug_moe_mix")][key_cols + value_cols].rename(
        columns={
            "task_group": "baseline_task_group",
            "preferred_metric": "baseline_preferred_metric",
            "raw_metric": "baseline_raw_metric",
            "raw_value": "baseline_raw_value",
            "oriented_value": "baseline_oriented_value",
        }
    )
    if baseline.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    endpoint_values = preferred_df[preferred_df["track"].isin(PATH_ENDPOINT_TRACKS)].copy()
    if not endpoint_values.empty:
        endpoint_pairs = endpoint_values.merge(baseline, on=key_cols, how="inner")
        endpoint_pairs["endpoint_track"] = endpoint_pairs["track"]
        endpoint_pairs["endpoint_label"] = endpoint_pairs["track"].map(TRACK_LABELS)
        endpoint_pairs["t"] = 1.0
        endpoint_pairs["path_source"] = "endpoint"
        rows.append(endpoint_pairs)

        baseline_pairs = endpoint_pairs.copy()
        baseline_pairs["track"] = "grug_moe_mix"
        baseline_pairs["track_label"] = "Proportional"
        baseline_pairs["task_group"] = baseline_pairs["baseline_task_group"]
        baseline_pairs["preferred_metric"] = baseline_pairs["baseline_preferred_metric"]
        baseline_pairs["raw_metric"] = baseline_pairs["baseline_raw_metric"]
        baseline_pairs["raw_value"] = baseline_pairs["baseline_raw_value"]
        baseline_pairs["oriented_value"] = baseline_pairs["baseline_oriented_value"]
        baseline_pairs["t"] = 0.0
        baseline_pairs["path_source"] = "proportional"
        rows.append(baseline_pairs)

    if not path_metrics_df.empty:
        path_preferred = preferred_task_values(path_metrics_df)
        if not path_preferred.empty:
            path_pairs = path_preferred.merge(baseline, on=key_cols, how="inner")
            path_pairs["endpoint_track"] = "grug_moe_mix_v4"
            path_pairs["endpoint_label"] = TRACK_LABELS["grug_moe_mix_v4"]
            path_pairs["t"] = path_pairs["track"].map(path_t_from_track)
            path_pairs["path_source"] = "intermediate"
            rows.append(path_pairs)

    if not rows:
        return pd.DataFrame()

    path_df = pd.concat(rows, ignore_index=True)
    path_df["delta_oriented"] = path_df["oriented_value"] - path_df["baseline_oriented_value"]
    path_df["benchmark_metric"] = path_df["task_alias"] + " / " + path_df["preferred_metric"]
    path_df = path_df.sort_values(["task_alias", "endpoint_label", "hidden_dim", "t"])
    return path_df.drop_duplicates(["endpoint_track", "hidden_dim", "budget", "task_alias", "t"], keep="last")


def path_task_loss_values(loss_df: pd.DataFrame, path_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build path-test loss-like task values in scaling-law coordinates."""
    if loss_df.empty:
        return pd.DataFrame()

    path_parts: list[pd.DataFrame] = []
    endpoint_df = loss_df[loss_df["track"].isin(["grug_moe_mix", "grug_moe_mix_v4"])].copy()
    if not endpoint_df.empty:
        endpoint_df["endpoint_track"] = "grug_moe_mix_v4"
        endpoint_df["endpoint_label"] = TRACK_LABELS["grug_moe_mix_v4"]
        endpoint_df["t"] = endpoint_df["track"].map({"grug_moe_mix": 0.0, "grug_moe_mix_v4": 1.0})
        endpoint_df["mixture_label"] = endpoint_df["track"].map(
            {"grug_moe_mix": "Proportional", "grug_moe_mix_v4": "v4"}
        )
        endpoint_df["path_source"] = endpoint_df["track"].map(
            {"grug_moe_mix": "proportional", "grug_moe_mix_v4": "endpoint"}
        )
        path_parts.append(endpoint_df)

    if not path_metrics_df.empty:
        path_loss_df = loss_like_task_values(path_metrics_df)
        if not path_loss_df.empty:
            path_loss_df["endpoint_track"] = "grug_moe_mix_v4"
            path_loss_df["endpoint_label"] = TRACK_LABELS["grug_moe_mix_v4"]
            path_loss_df["t"] = path_loss_df["track"].map(path_t_from_track)
            path_loss_df["mixture_label"] = path_loss_df["t"].map(lambda value: f"v4 path t={float(value):.2f}")
            path_loss_df["path_source"] = "intermediate"
            path_parts.append(path_loss_df)

    if not path_parts:
        return pd.DataFrame()
    path_loss_df = pd.concat(path_parts, ignore_index=True)
    path_loss_df["mixture_label"] = pd.Categorical(
        path_loss_df["mixture_label"], categories=PATH_MIXTURE_ORDER, ordered=True
    )
    return path_loss_df.sort_values(["task_alias", "mixture_label", "hidden_dim"])


def write_plot(fig: go.Figure, path: Path) -> str:
    fig.update_layout(template="plotly_white")
    fig.write_html(path, include_plotlyjs="cdn")
    return path.name


def task_loss_scaling_plot(loss_df: pd.DataFrame, fit_df: pd.DataFrame) -> go.Figure:
    """Create per-task small multiples with log-log power-law fits."""
    plot_df = loss_df.copy()
    tasks = sorted(plot_df["task_alias"].unique())
    if not tasks:
        fig = go.Figure()
        fig.update_layout(title="Per-task Grug-MoE scaling: no task metrics available")
        return fig
    n_cols = 4
    n_rows = math.ceil(len(tasks) / n_cols)
    subplot_titles = []
    for task in tasks:
        metric = plot_df.loc[plot_df["task_alias"].eq(task), "loss_metric"].mode().iloc[0]
        coverage = plot_df.loc[plot_df["task_alias"].eq(task), "coverage"].mode().iloc[0]
        coverage_note = "complete" if plot_df.loc[plot_df["task_alias"].eq(task), "common_task"].all() else coverage
        subplot_titles.append(f"{task}<br><sup>{metric}, lower is better; coverage {coverage_note}</sup>")
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, horizontal_spacing=0.055)
    color_map = dict(zip([TRACK_LABELS[t] for t in TRACKS], px.colors.qualitative.D3, strict=False))

    for task_idx, task in enumerate(tasks):
        row = task_idx // n_cols + 1
        col = task_idx % n_cols + 1
        task_df = plot_df[plot_df["task_alias"].eq(task)].copy()
        for track_label in [TRACK_LABELS[t] for t in TRACKS]:
            group = task_df[task_df["track_label"].eq(track_label)].sort_values("budget")
            if group.empty:
                continue
            showlegend = task_idx == 0
            fig.add_trace(
                go.Scatter(
                    x=group["budget"],
                    y=group["loss_value"],
                    mode="markers",
                    name=track_label,
                    legendgroup=track_label,
                    showlegend=showlegend,
                    marker={"color": color_map.get(track_label), "size": 7},
                    customdata=np.stack(
                        [group["hidden_dim"], group["raw_metric"], group["raw_value"], group["coverage"]],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "track=%{fullData.name}<br>"
                        "FLOPs=%{x:.2e}<br>"
                        "loss-like=%{y:.4g}<br>"
                        "hidden_dim=%{customdata[0]}<br>"
                        "raw_metric=%{customdata[1]}<br>"
                        "raw_value=%{customdata[2]:.4g}<br>"
                        "coverage=%{customdata[3]}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            fit = fit_df[(fit_df["task_alias"].eq(task)) & (fit_df["track_label"].eq(track_label))]
            if not fit.empty:
                fit_row = fit.iloc[0]
                xs = np.geomspace(float(fit_row["min_budget"]), float(fit_row["max_budget"]), 32)
                ys = 10 ** (float(fit_row["intercept"]) + float(fit_row["beta"]) * np.log10(xs))
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name=f"{track_label} fit",
                        legendgroup=track_label,
                        showlegend=False,
                        line={"color": color_map.get(track_label), "width": 1.5},
                        hovertemplate=(
                            f"track={track_label}<br>"
                            f"beta={float(fit_row['beta']):.3f}<br>"
                            f"log-log R2={float(fit_row['loglog_r2']):.3f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_xaxes(type="log", row=row, col=col)
        fig.update_yaxes(type="log", row=row, col=col)

    fig.update_layout(
        title="Per-task Grug-MoE scaling: available loss-like metrics vs training FLOPs",
        height=max(760, 260 * n_rows),
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.05, "xanchor": "center", "x": 0.5},
        margin={"t": 95, "b": 80, "l": 45, "r": 20},
    )
    fig.update_annotations(font_size=11)
    return fig


def path_task_scaling_plot(path_loss_df: pd.DataFrame) -> go.Figure:
    """Create per-task path plots in scaling-law coordinates."""
    if path_loss_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Path test scaling: no task metrics available")
        return fig

    plot_df = path_loss_df[path_loss_df["loss_value"] > 0].copy()
    tasks = sorted(plot_df["task_alias"].unique())
    n_cols = 4
    n_rows = math.ceil(len(tasks) / n_cols)
    subplot_titles = []
    for task in tasks:
        task_rows = plot_df[plot_df["task_alias"].eq(task)]
        metric = task_rows["loss_metric"].mode().iloc[0]
        subplot_titles.append(f"{task}<br><sup>{metric}, lower is better</sup>")

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, horizontal_spacing=0.055)
    for task_idx, task in enumerate(tasks):
        row = task_idx // n_cols + 1
        col = task_idx % n_cols + 1
        task_df = plot_df[plot_df["task_alias"].eq(task)].copy()
        for mixture_label in PATH_MIXTURE_ORDER:
            group = task_df[task_df["mixture_label"].astype(str).eq(mixture_label)].sort_values("budget")
            if group.empty:
                continue
            color = PATH_MIXTURE_COLORS[mixture_label]
            showlegend = task_idx == 0
            fig.add_trace(
                go.Scatter(
                    x=group["budget"],
                    y=group["loss_value"],
                    mode="lines+markers",
                    name=mixture_label,
                    legendgroup=mixture_label,
                    showlegend=showlegend,
                    marker={"color": color, "size": 7},
                    line={"color": color, "width": 2.0},
                    customdata=np.stack(
                        [
                            group["hidden_dim"],
                            group["t"],
                            group["raw_metric"],
                            group["raw_value"],
                            group["path_source"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "mixture=%{fullData.name}<br>"
                        "FLOPs=%{x:.2e}<br>"
                        "loss-like=%{y:.4g}<br>"
                        "hidden_dim=%{customdata[0]}<br>"
                        "t=%{customdata[1]:.2f}<br>"
                        "raw_metric=%{customdata[2]}<br>"
                        "raw_value=%{customdata[3]:.4g}<br>"
                        "source=%{customdata[4]}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(type="log", row=row, col=col)
        fig.update_yaxes(type="log", row=row, col=col)

    fig.update_layout(
        title="Path test: available loss-like metrics vs training FLOPs",
        height=max(820, 255 * n_rows),
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.04, "xanchor": "center", "x": 0.5},
        margin={"t": 95, "b": 95, "l": 45, "r": 20},
    )
    fig.update_annotations(font_size=11)
    return fig


def path_task_delta_plot(path_df: pd.DataFrame) -> go.Figure:
    """Create per-task path plots as oriented deltas from proportional."""
    if path_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Path test deltas: no task metrics available")
        return fig

    plot_df = path_df.copy()
    plot_df["mixture_label"] = plot_df["t"].map(
        {
            0.0: "Proportional",
            0.25: "v4 path t=0.25",
            0.50: "v4 path t=0.50",
            0.75: "v4 path t=0.75",
            1.0: "v4",
        }
    )
    tasks = sorted(plot_df["task_alias"].unique())
    n_cols = 4
    n_rows = math.ceil(len(tasks) / n_cols)
    subplot_titles = []
    for task in tasks:
        task_rows = plot_df[plot_df["task_alias"].eq(task)]
        metric = task_rows["preferred_metric"].mode().iloc[0]
        subplot_titles.append(f"{task}<br><sup>{metric}, delta vs proportional; higher is better</sup>")

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, horizontal_spacing=0.055)
    for task_idx, task in enumerate(tasks):
        row = task_idx // n_cols + 1
        col = task_idx % n_cols + 1
        task_df = plot_df[plot_df["task_alias"].eq(task)].copy()
        for mixture_label in PATH_MIXTURE_ORDER:
            group = task_df[task_df["mixture_label"].astype(str).eq(mixture_label)].sort_values("budget")
            if group.empty:
                continue
            color = PATH_MIXTURE_COLORS[mixture_label]
            showlegend = task_idx == 0
            fig.add_trace(
                go.Scatter(
                    x=group["budget"],
                    y=group["delta_oriented"],
                    mode="lines+markers",
                    name=mixture_label,
                    legendgroup=mixture_label,
                    showlegend=showlegend,
                    marker={"color": color, "size": 7},
                    line={"color": color, "width": 2.0},
                    customdata=np.stack(
                        [
                            group["hidden_dim"],
                            group["t"],
                            group["preferred_metric"],
                            group["raw_value"],
                            group["baseline_raw_value"],
                            group["path_source"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "mixture=%{fullData.name}<br>"
                        "FLOPs=%{x:.2e}<br>"
                        "oriented delta=%{y:.4g}<br>"
                        "hidden_dim=%{customdata[0]}<br>"
                        "t=%{customdata[1]:.2f}<br>"
                        "metric=%{customdata[2]}<br>"
                        "raw_value=%{customdata[3]:.4g}<br>"
                        "proportional_raw=%{customdata[4]:.4g}<br>"
                        "source=%{customdata[5]}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        fig.add_hline(y=0.0, line={"color": "#666", "width": 1, "dash": "dot"}, row=row, col=col)
        fig.update_xaxes(type="log", row=row, col=col)

    fig.update_layout(
        title="Path test: oriented task deltas vs proportional at the same scale",
        height=max(820, 255 * n_rows),
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.04, "xanchor": "center", "x": 0.5},
        margin={"t": 95, "b": 95, "l": 45, "r": 20},
    )
    fig.update_annotations(font_size=11)
    return fig


def make_dashboard(
    runs_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    preferred_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    loss_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    path_runs_df: pd.DataFrame,
    path_eval_metrics_df: pd.DataFrame,
    path_df: pd.DataFrame,
    path_loss_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = OUTPUT_DIR / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    track_order = [TRACK_LABELS[t] for t in TRACKS]
    fig_aggregate = px.line(
        summary_df,
        x="scale_label",
        y="aggregate_z",
        color="track_label",
        markers=True,
        category_orders={"track_label": track_order},
        hover_data=["common_task_count", "rank_at_scale"],
        title="Grug-MoE track comparison: common-task smooth aggregate",
        labels={"aggregate_z": "Mean z-score across common smooth task metrics", "scale_label": "Scale"},
    )
    fig_aggregate.update_traces(line={"width": 3}, marker={"size": 9})
    aggregate_html = write_plot(fig_aggregate, img_dir / "aggregate_scaling.html")

    if not accuracy_df.empty:
        fig_accuracy = px.line(
            accuracy_df,
            x="scale_label",
            y="mean_accuracy",
            color="track_label",
            markers=True,
            category_orders={"track_label": track_order},
            hover_data=["accuracy_task_count"],
            title="Mean hard accuracy across available MCQ tasks",
            labels={"mean_accuracy": "Mean accuracy / normalized accuracy", "scale_label": "Scale"},
        )
        fig_accuracy.update_traces(line={"width": 3}, marker={"size": 9})
        accuracy_html = write_plot(fig_accuracy, img_dir / "accuracy_scaling.html")
    else:
        accuracy_html = ""

    task_plot_df = preferred_df.copy()
    task_plot_df["display_value"] = task_plot_df["oriented_value"]
    fig_tasks = px.line(
        task_plot_df,
        x="scale_label",
        y="display_value",
        color="track_label",
        line_group="task_alias",
        facet_col="task_group",
        facet_col_wrap=3,
        markers=True,
        category_orders={"track_label": track_order},
        hover_data=["task_alias", "preferred_metric", "raw_value", "coverage", "common_task"],
        title="Available preferred task metrics by group (higher oriented value is better)",
        labels={"display_value": "Oriented selected metric", "scale_label": "Scale"},
    )
    fig_tasks.update_yaxes(matches=None)
    task_html = write_plot(fig_tasks, img_dir / "task_metric_facets.html")

    fig_loss_scaling = task_loss_scaling_plot(loss_df, fit_df)
    loss_scaling_html = write_plot(fig_loss_scaling, img_dir / "task_loss_scaling_loglog.html")

    fig_path = path_task_scaling_plot(path_loss_df)
    path_html = write_plot(fig_path, img_dir / "path_task_scaling_facets.html")
    fig_delta = path_task_delta_plot(path_df)
    delta_html = write_plot(fig_delta, img_dir / "path_task_delta_facets.html")

    all_domains = sorted(weights_df["domain"].unique())
    heat_df = weights_df.copy()
    heat_df["track_phase"] = heat_df["track_label"] + " / " + heat_df["phase"]
    heat_df["domain"] = pd.Categorical(heat_df["domain"], categories=all_domains, ordered=True)
    fig_heat = px.imshow(
        heat_df.pivot_table(
            index="track_phase",
            columns="domain",
            values="weight",
            aggfunc="mean",
            fill_value=0,
            observed=False,
        ),
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        title="Full mixture weights: all domains",
        labels={"x": "Domain", "y": "Track / phase", "color": "Weight"},
    )
    fig_heat.update_layout(height=max(760, 80 + 28 * heat_df["track_phase"].nunique()), xaxis={"tickangle": -45})
    heat_html = write_plot(fig_heat, img_dir / "mixture_full_domain_heatmap.html")

    family_df = weights_df.groupby(["track_label", "phase", "family"], as_index=False)["weight"].sum()
    fig_family = px.bar(
        family_df,
        x="track_label",
        y="weight",
        color="family",
        facet_col="phase",
        title="Mixture family shares by phase",
        category_orders={"track_label": track_order},
        labels={"track_label": "Track", "weight": "Family weight"},
    )
    family_html = write_plot(fig_family, img_dir / "mixture_family_shares.html")

    best_scale = (
        summary_df.sort_values(["hidden_dim", "aggregate_z"], ascending=[True, False]).groupby("hidden_dim").head(1)
    )
    best_overall = (
        summary_df.groupby(["track", "track_label"], as_index=False)
        .agg(mean_aggregate_z=("aggregate_z", "mean"), median_rank=("rank_at_scale", "median"))
        .sort_values("mean_aggregate_z", ascending=False)
    )
    mixture_wide = (
        weights_df.pivot_table(
            index=["family", "domain"],
            columns=["track_label", "phase"],
            values="weight",
            aggfunc="mean",
            fill_value=0.0,
            observed=False,
        )
        .sort_index()
        .reset_index()
    )
    mixture_wide.columns = [
        " / ".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in mixture_wide.columns
    ]
    task_coverage = (
        loss_df[
            ["task_alias", "task_group", "loss_metric", "available_cells", "expected_cells", "coverage", "common_task"]
        ]
        .drop_duplicates()
        .sort_values(["common_task", "task_group", "task_alias"], ascending=[True, True, True])
    )
    if path_runs_df.empty:
        path_training_status = pd.DataFrame()
    else:
        path_training_status = (
            path_runs_df.groupby(["t", "state"], as_index=False)
            .agg(
                run_count=("run_id", "nunique"),
                min_hidden_dim=("hidden_dim", "min"),
                max_hidden_dim=("hidden_dim", "max"),
            )
            .sort_values(["t", "state"])
        )
    if path_loss_df.empty:
        path_metric_status = pd.DataFrame()
    else:
        path_metric_status = (
            path_loss_df.groupby(["mixture_label", "t", "path_source"], observed=True, as_index=False)
            .agg(task_cells=("task_alias", "count"), tasks=("task_alias", "nunique"), scales=("hidden_dim", "nunique"))
            .sort_values(["t", "mixture_label", "path_source"])
        )
    path_eval_cell_count = 0
    if not path_eval_metrics_df.empty:
        path_eval_cell_count = (
            path_eval_metrics_df[["track", "hidden_dim", "budget", "task_alias"]].drop_duplicates().shape[0]
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Grug-MoE mixture scaling dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #24324a; }}
    h1, h2 {{ color: #172642; }}
    .note {{ max-width: 1100px; line-height: 1.5; }}
    iframe {{ width: 100%; height: 760px; border: 0; margin: 12px 0 32px; }}
    iframe.tall {{ height: 1500px; }}
    table {{ border-collapse: collapse; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #ccd4e0; padding: 6px 10px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #eef3f8; }}
    .table-scroll {{ max-width: 100%; overflow-x: auto; }}
    code {{ background: #eef3f8; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Grug-MoE mixture scaling dashboard</h1>
  <p class="note">
    Source of truth: GCS executor metadata and <code>evaluation/grug_logprob</code> result JSONs.
    The aggregate below is diagnostic: for each common task we choose a smooth metric
    (<code>choice_prob_norm</code> if present, otherwise sign-flipped <code>bpb</code>/<code>nll</code>),
    z-score it across completed cells, then average across tasks. Higher is better.
  </p>

  <h2>Best Track Summary</h2>
  {best_overall.to_html(index=False, float_format=lambda x: f"{x:.4f}")}

  <h2>Best Track At Each Scale</h2>
  {
    best_scale[
        ["hidden_dim", "budget", "track_label", "aggregate_z", "rank_at_scale", "common_task_count"]
    ].to_html(index=False, float_format=lambda x: f"{x:.4f}")
  }

  <h2>Common-Task Smooth Aggregate</h2>
  <iframe src="img/{aggregate_html}"></iframe>

  {"<h2>Hard Accuracy Diagnostic</h2><iframe src='img/" + accuracy_html + "'></iframe>" if accuracy_html else ""}

  <h2>Task-Level Metrics</h2>
  <iframe src="img/{task_html}"></iframe>

  <h2>Per-Task Log-Log Scaling</h2>
  <p class="note">
    These plots include partial task coverage. The coverage label is the number
    of available track/scale cells over the expected full grid, so tasks with
    one missing eval are still shown and visibly marked.
  </p>
  <iframe class="tall" src="img/{loss_scaling_html}"></iframe>

  <h2>Path Test: Proportional to Endpoint Mixtures</h2>
  <p class="note">
    The path plot keeps the same scaling-law coordinates as the main dashboard:
    x is training FLOPs, y is a loss-like task metric, and lower is better. The
    <code>t=0</code> curve is the proportional track, <code>t=1</code> is the
    v4 endpoint track, and intermediate curves are included only when follow-up
    <code>evaluation/grug_logprob</code> results exist for the path checkpoints.
    Current path follow-up eval cells found: <code>{path_eval_cell_count}</code>.
  </p>
  <h3>Path Training Status</h3>
  {
    path_training_status.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    if not path_training_status.empty
    else "<p class='note'>No path training roots found.</p>"
  }
  <h3>Path Metric Coverage</h3>
  {
    path_metric_status.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    if not path_metric_status.empty
    else "<p class='note'>No path metric rows found.</p>"
  }
  <iframe class="tall" src="img/{path_html}"></iframe>
  <h3>Path Deltas vs Proportional</h3>
  <p class="note">
    This view uses oriented task metrics and subtracts the proportional value
    at the same scale. Positive values mean better than proportional.
  </p>
  <iframe class="tall" src="img/{delta_html}"></iframe>

  <h2>Task Coverage</h2>
  <div class="table-scroll">
  {task_coverage.to_html(index=False)}
  </div>

  <h2>Full Mixture Domain Heatmap</h2>
  <iframe src="img/{heat_html}"></iframe>

  <h2>Full Mixture Weights</h2>
  <div class="table-scroll">
  {mixture_wide.to_html(index=False, float_format=lambda x: f"{x:.6f}")}
  </div>

  <h2>Mixture Family Shares</h2>
  <iframe src="img/{family_html}"></iframe>
</body>
</html>
"""
    (OUTPUT_DIR / "dashboard.html").write_text(html)


def cached_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main(*, use_cached_inputs: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if use_cached_inputs:
        runs_df = cached_csv("grug_moe_mix_runs.csv")
        weights_df = cached_csv("grug_moe_mix_weights_long.csv")
        metrics_df = cached_csv("grug_moe_mix_eval_metrics_long.csv")
        if runs_df.empty or weights_df.empty or metrics_df.empty:
            raise FileNotFoundError(
                "--use-cached-inputs requires existing grug_moe_mix_runs.csv, "
                "grug_moe_mix_weights_long.csv, and grug_moe_mix_eval_metrics_long.csv"
            )
        checkpoint_path = OUTPUT_DIR / "checkpoint_by_run_id.json"
        checkpoint_by_run_id = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    else:
        runs_df, weights_df, checkpoint_by_run_id = collect_training_metadata()
        metrics_df = collect_eval_metrics()

    path_runs_df = collect_path_training_metadata()
    path_parent_run_ids = []
    if not path_runs_df.empty:
        path_parent_run_ids = [str(run_id) for run_id in path_runs_df["run_id"].dropna().unique()]
    path_eval_metrics_df = collect_path_eval_metrics(path_parent_run_ids or None)
    runs_df = normalize_scale_columns(runs_df)
    metrics_df = normalize_scale_columns(metrics_df)
    path_runs_df = normalize_scale_columns(path_runs_df)
    path_eval_metrics_df = normalize_scale_columns(path_eval_metrics_df)
    preferred_df = preferred_task_values(metrics_df)
    summary_df = cell_summary(preferred_df)
    accuracy_df = hard_accuracy_summary(metrics_df)
    loss_df = loss_like_task_values(metrics_df)
    fit_df = power_law_fits(loss_df)
    path_df = path_task_deltas(preferred_df, path_eval_metrics_df)
    path_loss_df = path_task_loss_values(loss_df, path_eval_metrics_df)

    runs_df.to_csv(OUTPUT_DIR / "grug_moe_mix_runs.csv", index=False)
    path_runs_df.to_csv(OUTPUT_DIR / "grug_moe_v4_path_training_runs.csv", index=False)
    weights_df.to_csv(OUTPUT_DIR / "grug_moe_mix_weights_long.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "grug_moe_mix_eval_metrics_long.csv", index=False)
    path_eval_metrics_df.to_csv(OUTPUT_DIR / "grug_moe_v4_path_eval_metrics_long.csv", index=False)
    preferred_df.to_csv(OUTPUT_DIR / "grug_moe_mix_preferred_task_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "grug_moe_mix_cell_summary.csv", index=False)
    accuracy_df.to_csv(OUTPUT_DIR / "grug_moe_mix_accuracy_summary.csv", index=False)
    loss_df.to_csv(OUTPUT_DIR / "grug_moe_mix_task_loss_like_metrics.csv", index=False)
    fit_df.to_csv(OUTPUT_DIR / "grug_moe_mix_task_powerlaw_fits.csv", index=False)
    path_df.to_csv(OUTPUT_DIR / "grug_moe_path_task_deltas.csv", index=False)
    path_loss_df.to_csv(OUTPUT_DIR / "grug_moe_path_task_scaling.csv", index=False)
    (OUTPUT_DIR / "checkpoint_by_run_id.json").write_text(json.dumps(checkpoint_by_run_id, indent=2, sort_keys=True))
    make_dashboard(
        runs_df,
        weights_df,
        metrics_df,
        preferred_df,
        summary_df,
        accuracy_df,
        loss_df,
        fit_df,
        path_runs_df,
        path_eval_metrics_df,
        path_df,
        path_loss_df,
    )

    best_overall = (
        summary_df.groupby(["track_label"], as_index=False)
        .agg(mean_aggregate_z=("aggregate_z", "mean"), median_rank=("rank_at_scale", "median"))
        .sort_values("mean_aggregate_z", ascending=False)
    )
    print(f"Wrote dashboard to {OUTPUT_DIR / 'dashboard.html'}")
    if not path_runs_df.empty:
        print(path_runs_df.groupby(["t", "state"]).size().rename("runs").reset_index().to_string(index=False))
    if path_eval_metrics_df.empty:
        print("No Grug path follow-up eval metrics found yet.")
    print(best_overall.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-cached-inputs",
        action="store_true",
        help="Reuse existing base-track CSVs and refresh only path status/eval rows.",
    )
    args = parser.parse_args()
    main(use_cached_inputs=args.use_cached_inputs)
