# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["wandb>=0.19"]
# ///
"""Export the completed 280-row Delphi 3e18 fit swarm from W&B.

The source panel is canonical. W&B supplies immutable run identity and observed
metrics, while the native Table-9 project supplies the 51-component macro BPB.
Every exported row must match one source coordinate exactly and have both
headline targets before the artifact is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import wandb

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    build_delphi_3e18_heldout_registry as heldouts,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_PANEL = (
    SCRIPT_DIR / "reference_outputs/olmo_base_easy_per_component_dsp_kl_sweep_300m_20260628/fit_panel_table9_macro.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_augmented_swarm_3e18_20260714"
TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
FIT_SWARM_TAG = "delphi-3e18-augmented-swarm"
EVAL_GROUP = "olmo_base_eval_table9_delphi_3e18_augmented_swarm"
EXPECTED_ROWS = 280
EXPECTED_GLOBAL_STEP = 3006
WEIGHT_TOLERANCE = 1e-10
TABLE9_NATIVE_PREFIX = "olmo_base_easy/table9"
MMLU_COMPONENT_FIELDS = frozenset({"mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"})

METADATA_FIELDS = (
    "swarm_run_name",
    "training_wandb_run_id",
    "training_wandb_url",
    "training_created_at",
    "training_state",
    "data_seed",
    "trainer_seed",
    "global_step",
    "num_train_steps",
    "phase_boundary_step",
    "phase_0_fraction",
    "parameter_count",
    "realized_train_tokens",
    "uncheatable_bpb",
    "table9_macro_bpb",
    "table9_eval_run_id",
    "table9_eval_run_name",
    "table9_eval_url",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    return parser.parse_args()


def read_source_panel(path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        fields = list(reader.fieldnames)
        domains = [field.removeprefix("phase_0_") for field in fields if field.startswith("phase_0_")]
        rows = list(reader)
    if len(domains) != 39 or len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected 39 domains and {EXPECTED_ROWS} rows, found {len(domains)} and {len(rows)}")
    names = [row["run_name"] for row in rows]
    if len(set(names)) != EXPECTED_ROWS:
        raise ValueError("Source panel run names are not unique")
    return fields, domains, rows


def table9_component_fields(fields: Sequence[str]) -> list[str]:
    components = [
        field for field in fields if field.startswith("olmo_base_eval/easy_bpb/") or field in MMLU_COMPONENT_FIELDS
    ]
    if len(components) != 51:
        raise ValueError(f"Expected 51 Table-9 component fields, found {len(components)}")
    return components


def native_component_key(field: str) -> str:
    if field in MMLU_COMPONENT_FIELDS:
        component = field
    else:
        prefix = "olmo_base_eval/easy_bpb/"
        suffix = "/bpb"
        if not field.startswith(prefix) or not field.endswith(suffix):
            raise ValueError(f"Not a Table-9 component field: {field}")
        component = field.removeprefix(prefix).removesuffix(suffix)
    return f"{TABLE9_NATIVE_PREFIX}/{component}/bpb"


def tag_value(tags: Sequence[str], prefix: str) -> str:
    matches = [tag.removeprefix(prefix) for tag in tags if tag.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {prefix!r} tag, got {matches}")
    return matches[0]


def finite_summary(summary: Mapping[str, object], key: str) -> float:
    value = summary.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Missing finite W&B summary metric {key!r}")
    return float(value)


def collect_table9_attempts(
    api: wandb.Api,
    training_ids: set[str],
    component_fields: Sequence[str],
    batch_size: int,
) -> dict[str, Mapping[str, object]]:
    metadata = api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=1000, lazy=True)
    eval_ids = sorted(run.id for run in metadata)
    by_training: dict[str, list[dict[str, object]]] = {run_id: [] for run_id in training_ids}
    for run in heldouts.batch_runs(api, EVAL_PROJECT, eval_ids, batch_size):
        checkpoint_path = str(run.config.get("checkpoint_path") or "")
        training_id = heldouts.parse_checkpoint_training_run_id(checkpoint_path)
        if training_id not in training_ids:
            continue
        summary = dict(run.summary)
        components: dict[str, float] = {}
        for field in component_fields:
            value = summary.get(native_component_key(field))
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                components[field] = float(value)
        by_training[training_id].append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "run_url": run.url,
                "created_at": str(run.created_at),
                "state": run.state,
                "table9_macro_bpb": heldouts.summary_value(summary, heldouts.TABLE9_KEYS),
                "table9_components": components,
            }
        )
    selected: dict[str, Mapping[str, object]] = {}
    for training_id, attempts in by_training.items():
        complete = [
            row
            for row in attempts
            if row["state"] == "finished"
            and row["table9_macro_bpb"] != ""
            and len(row["table9_components"]) == len(component_fields)
        ]
        if complete:
            selected[training_id] = max(
                complete,
                key=lambda row: (str(row["created_at"]), str(row["run_id"])),
            )
    return selected


def weight_distance(
    source_row: Mapping[str, str],
    phase_0: Sequence[float],
    phase_1: Sequence[float],
    domains: Sequence[str],
) -> float:
    expected_0 = [float(source_row[f"phase_0_{domain}"]) for domain in domains]
    expected_1 = [float(source_row[f"phase_1_{domain}"]) for domain in domains]
    return max(abs(left - right) for left, right in zip([*phase_0, *phase_1], [*expected_0, *expected_1], strict=True))


def collect_rows(
    api: wandb.Api,
    domains: list[str],
    component_fields: list[str],
    source_rows: list[dict[str, str]],
    batch_size: int,
) -> list[dict[str, object]]:
    metadata = api.runs(
        TRAIN_PROJECT,
        filters={"tags": {"$in": [FIT_SWARM_TAG]}},
        per_page=1000,
        lazy=True,
    )
    training_ids = sorted(run.id for run in metadata)
    source_by_name = {row["run_name"]: row for row in source_rows}
    candidates: dict[str, list[dict[str, object]]] = {source_name: [] for source_name in source_by_name}
    for run in heldouts.batch_runs(api, TRAIN_PROJECT, training_ids, batch_size):
        tags = [str(tag) for tag in run.tags or []]
        source_name = tag_value(tags, "source_run=")
        if source_name not in source_by_name:
            raise ValueError(f"Unknown source row {source_name!r} for W&B run {run.name}")
        config = dict(run.config)
        summary = dict(run.summary)
        phase_0, phase_1, phase_count, phase_boundary = heldouts.parse_train_weights(config, domains)
        if phase_count != 2:
            raise ValueError(f"Expected two phases for {run.name}, got {phase_count}")
        source_row = source_by_name[source_name]
        if weight_distance(source_row, phase_0, phase_1, domains) > WEIGHT_TOLERANCE:
            continue
        trainer = config.get("trainer") if isinstance(config.get("trainer"), Mapping) else {}
        num_train_steps = int(trainer.get("num_train_steps") or 0)
        global_step_value = summary.get("global_step")
        if not isinstance(global_step_value, (int, float)) or not math.isfinite(float(global_step_value)):
            continue
        global_step = int(global_step_value)
        if run.state != "finished" or num_train_steps != EXPECTED_GLOBAL_STEP + 1 or global_step != EXPECTED_GLOBAL_STEP:
            continue
        parameter_count = int(float(tag_value(tags, "N=")))
        candidates[source_name].append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "run_url": run.url,
                "created_at": str(run.created_at),
                "config": config,
                "trainer": trainer,
                "summary": summary,
                "phase_boundary": phase_boundary,
                "num_train_steps": num_train_steps,
                "global_step": global_step,
                "parameter_count": parameter_count,
            }
        )
    completed_training_ids = {str(candidate["run_id"]) for rows in candidates.values() for candidate in rows}
    table9 = collect_table9_attempts(api, completed_training_ids, component_fields, batch_size)

    collected: dict[str, dict[str, object]] = {}
    for source_name, source_row in source_by_name.items():
        eligible = [candidate for candidate in candidates[source_name] if candidate["run_id"] in table9]
        if not eligible:
            raise ValueError(f"No completed training and native Table-9 attempt for source row {source_name}")
        candidate = max(eligible, key=lambda row: (str(row["created_at"]), str(row["run_id"])))
        eval_attempt = table9[str(candidate["run_id"])]
        component_values = dict(eval_attempt["table9_components"])
        table9_macro = float(eval_attempt["table9_macro_bpb"])
        component_macro = sum(float(component_values[field]) for field in component_fields) / len(component_fields)
        if not math.isclose(component_macro, table9_macro, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Native component mean {component_macro:.12f} does not match macro {table9_macro:.12f} "
                f"for {candidate['run_name']}"
            )
        config = candidate["config"]
        trainer = candidate["trainer"]
        if not isinstance(config, Mapping) or not isinstance(trainer, Mapping):
            raise TypeError("Stored training candidate config and trainer must be mappings")
        collected[source_name] = {
            **source_row,
            **component_values,
            "swarm_run_name": candidate["run_name"],
            "training_wandb_run_id": candidate["run_id"],
            "training_wandb_url": candidate["run_url"],
            "training_created_at": candidate["created_at"],
            "training_state": "finished",
            "data_seed": config.get("data_seed", ""),
            "trainer_seed": trainer.get("seed", ""),
            "global_step": candidate["global_step"],
            "num_train_steps": candidate["num_train_steps"],
            "phase_boundary_step": candidate["phase_boundary"],
            "phase_0_fraction": float(candidate["phase_boundary"]) / int(candidate["num_train_steps"]),
            "parameter_count": candidate["parameter_count"],
            "realized_train_tokens": int(
                int(candidate["num_train_steps"])
                * int(trainer.get("train_batch_size") or 0)
                * int(config["train_seq_len"])
            ),
            "uncheatable_bpb": finite_summary(candidate["summary"], "eval/uncheatable_eval/bpb"),
            "table9_macro_bpb": table9_macro,
            "table9_eval_run_id": eval_attempt["run_id"],
            "table9_eval_run_name": eval_attempt["run_name"],
            "table9_eval_url": eval_attempt["run_url"],
        }
    return [collected[row["run_name"]] for row in source_rows]


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    source_fields, domains, source_rows = read_source_panel(args.source_panel)
    component_fields = table9_component_fields(source_fields)
    api = wandb.Api(timeout=args.wandb_timeout)
    rows = collect_rows(api, domains, component_fields, source_rows, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "delphi_augmented_swarm_3e18_wide.csv"
    write_csv(output_path, [*source_fields, *METADATA_FIELDS], rows)
    summary = {
        "exported_at": datetime.now(UTC).isoformat(),
        "fit_swarm_tag": FIT_SWARM_TAG,
        "eval_group": EVAL_GROUP,
        "row_count": len(rows),
        "domain_count": len(domains),
        "uncheatable_complete": sum(math.isfinite(float(row["uncheatable_bpb"])) for row in rows),
        "table9_complete": sum(math.isfinite(float(row["table9_macro_bpb"])) for row in rows),
        "source_panel": str(args.source_panel),
        "output_csv": str(output_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
