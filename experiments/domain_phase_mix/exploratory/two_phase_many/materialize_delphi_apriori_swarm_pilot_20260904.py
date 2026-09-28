# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "wandb>=0.21"]
# ///
"""Materialize validated outcomes from the frozen Delphi a-priori swarm pilot.

Each admitted training run must reproduce the frozen row's data seed, trainer
seed, simulated-epoch subset seed, and per-domain pool fractions in its W&B
configuration. The output is consumed by the single-phase registry builder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import launch_delphi_apriori_swarm_3e18 as launcher  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_delphi_bucket_epoch_dose_heldout_20260903 as endpoints,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    prepare_single_phase_heldout_benchmark_20260902 as heldout,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import DOMAIN_NAMES  # noqa: E402

DESIGN_TABLE = launcher.DEFAULT_DESIGN_TABLE
DEFAULT_OUTPUT_DIR = heldout.APRIORI_SWARM_MATERIALIZATION_DIR
TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_GROUPS = tuple(f"{launcher.PROVENANCE_PANEL}_{wave}" for wave in launcher.WAVES)
EVAL_GROUP = launcher.TABLE9_WANDB_GROUP
RESULT_COLUMNS = (
    "run_order",
    "run_name",
    "training_wandb_run_id",
    "training_wandb_url",
    "training_wandb_state",
    "tpu_type",
    "tpu_zone",
    "data_seed",
    "trainer_seed",
    "subset_seed",
    *(f"{heldout.POOL_FRACTION_PREFIX}{domain}" for domain in DOMAIN_NAMES),
    "uncheatable_bpb",
    "uncheatable_provenance",
    "uncheatable_source_uri",
    "uncheatable_component_count",
    "table9_eval_run_id",
    "table9_eval_url",
    "table9_eval_state",
    "table9_macro_bpb",
    "table9_component_count",
)
COMPONENT_COLUMNS = (
    "run_order",
    "run_name",
    "component_position",
    "component",
    "bpb",
    "provenance",
)


def _required_int(value: object, label: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{label} is missing")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not an integer: {value!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} is not an integer: {value!r}")
    return int(numeric)


def read_design(path: Path = DESIGN_TABLE) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        low_memory=False,
        dtype={"data_seed": "Int64", "trainer_seed": "Int64", "subset_seed": "Int64"},
    )
    frame.insert(0, "run_order", np.arange(len(frame), dtype=int))
    return frame


def _pool_fractions_from_config(config: Mapping[str, object]) -> dict[str, float]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("W&B config has no data mapping")
    payload = data.get("simulated_epoch_pool_fractions")
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError("W&B simulated_epoch_pool_fractions is not a mapping")
    unknown = set(payload) - set(DOMAIN_NAMES)
    if unknown:
        raise ValueError(f"W&B pool fractions contain unknown domains: {sorted(unknown)}")
    fractions = {domain: float(payload.get(domain, 1.0)) for domain in DOMAIN_NAMES}
    invalid = {domain: value for domain, value in fractions.items() if not 0.0 < value <= 1.0}
    if invalid:
        raise ValueError(f"W&B pool fractions lie outside (0, 1]: {invalid}")
    return fractions


def validate_runtime_config(
    run: Any,
    design_row: Mapping[str, object],
    *,
    expected_tpu_type: str,
    expected_tpu_zone: str,
) -> dict[str, object]:
    """Validate and return runtime provenance persisted by one training run."""
    config = dict(run.config)
    data = config.get("data")
    trainer = config.get("trainer")
    if not isinstance(data, Mapping) or not isinstance(trainer, Mapping):
        raise ValueError(f"{run.id}: W&B config lacks data or trainer mappings")
    observed = {
        "data_seed": _required_int(config.get("data_seed"), f"{run.id}/data_seed"),
        "trainer_seed": _required_int(trainer.get("seed"), f"{run.id}/trainer.seed"),
        "subset_seed": _required_int(
            data.get("simulated_epoch_subset_seed"), f"{run.id}/data.simulated_epoch_subset_seed"
        ),
    }
    for seed, value in observed.items():
        expected = _required_int(design_row[seed], f"design/{design_row['run_name']}/{seed}")
        if value != expected:
            raise ValueError(f"{run.id}: {seed} {value} differs from frozen design {expected}")

    fractions = _pool_fractions_from_config(config)
    for domain, value in fractions.items():
        expected = float(design_row[f"pool_fraction_{domain}"])
        if not math.isclose(value, expected, abs_tol=1e-12, rel_tol=0.0):
            raise ValueError(f"{run.id}: pool fraction for {domain} is {value}, expected {expected}")

    tags = set(run.tags or ())
    for expected_tag in (f"tpu_type={expected_tpu_type}", f"tpu_zone={expected_tpu_zone}"):
        if expected_tag not in tags:
            raise ValueError(f"{run.id}: missing runtime placement tag {expected_tag}")
    return {
        **observed,
        "tpu_type": expected_tpu_type,
        "tpu_zone": expected_tpu_zone,
        **{f"{heldout.POOL_FRACTION_PREFIX}{domain}": value for domain, value in fractions.items()},
    }


def _matches_design_run(run: Any, run_name: str) -> bool:
    expected_tag = launcher.swarm._wandb_tag(f"source_run={run_name}")
    return expected_tag in set(run.tags or ()) or str(run.name).startswith(f"{run_name}-")


def _selected_training(
    runs: Sequence[Any],
    design_row: Mapping[str, object],
    *,
    expected_tpu_type: str,
    expected_tpu_zone: str,
) -> tuple[dict[str, object], dict[str, object]]:
    run_name = str(design_row["run_name"])
    candidates: list[tuple[dict[str, object], dict[str, object]]] = []
    for run in runs:
        if not _matches_design_run(run, run_name):
            continue
        trainer = dict(run.config.get("trainer") or {})
        expected_step = _required_int(trainer.get("num_train_steps"), f"{run.id}/trainer.num_train_steps") - 1
        endpoint = endpoints._uncheatable_candidate(run, expected_step)
        if endpoint is None:
            continue
        provenance = validate_runtime_config(
            run,
            design_row,
            expected_tpu_type=expected_tpu_type,
            expected_tpu_zone=expected_tpu_zone,
        )
        candidates.append((endpoint, provenance))
    if not candidates:
        raise ValueError(f"{run_name}: no complete, configuration-valid Uncheatable endpoint")
    reference = candidates[0][0]
    for endpoint, _provenance in candidates[1:]:
        if abs(float(endpoint["aggregate"]) - float(reference["aggregate"])) > endpoints.RETRY_TOLERANCE:
            raise ValueError(f"{run_name}: complete training attempts disagree")
        if not np.allclose(endpoint["values"], reference["values"], atol=endpoints.RETRY_TOLERANCE, rtol=0.0):
            raise ValueError(f"{run_name}: complete training attempts disagree")
    return max(candidates, key=lambda candidate: str(candidate[0]["run"].created_at))


def _table9_candidate(run: Any) -> tuple[str, dict[str, object]] | None:
    provenance = dict(run.config.get("provenance") or {})
    run_name = str(provenance.get("swarm_run_name") or provenance.get("source_run_name") or "")
    if not run_name:
        return None
    summary = dict(run.summary)
    keys = heldout.table9_summary_keys()
    if any(keys[component] not in summary for component in heldout.table9_components()):
        return None
    values = np.asarray([summary[keys[component]] for component in heldout.table9_components()], dtype=float)
    aggregate = next(
        (endpoints._finite(summary.get(key)) for key in heldout.TABLE9_AGGREGATE_KEYS if key in summary),
        None,
    )
    if aggregate is None:
        aggregate = float(values.mean())
    aggregate, values = endpoints._validated_payload(
        target="table9",
        aggregate=aggregate,
        component_values=list(values),
        identity=str(run.id),
    )
    return run_name, {"run": run, "aggregate": aggregate, "values": values}


def _table9_by_run_name(runs: Sequence[Any]) -> dict[str, dict[str, object]]:
    candidates: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        candidate = _table9_candidate(run)
        if candidate is not None:
            run_name, payload = candidate
            candidates.setdefault(run_name, []).append(payload)
    selected: dict[str, dict[str, object]] = {}
    for run_name, attempts in candidates.items():
        reference = attempts[0]
        for attempt in attempts[1:]:
            if abs(float(attempt["aggregate"]) - float(reference["aggregate"])) > endpoints.RETRY_TOLERANCE:
                raise ValueError(f"{run_name}: complete Table-9 attempts disagree")
            if not np.allclose(attempt["values"], reference["values"], atol=endpoints.RETRY_TOLERANCE, rtol=0.0):
                raise ValueError(f"{run_name}: complete Table-9 attempts disagree")
        selected[run_name] = max(attempts, key=lambda attempt: str(attempt["run"].created_at))
    return selected


def _runs_in_groups(api: wandb.Api, project: str, groups: Sequence[str]) -> list[Any]:
    by_id: dict[str, Any] = {}
    for group in groups:
        for run in api.runs(project, filters={"group": group}, per_page=500):
            by_id[str(run.id)] = run
    return list(by_id.values())


def materialize(
    output_dir: Path,
    run_names: Sequence[str],
    timeout: int,
    *,
    expected_tpu_type: str = launcher.TARGET_TPU_TYPE,
    expected_tpu_zone: str = launcher.DEFAULT_TPU_ZONE,
) -> dict[str, object]:
    design = read_design()
    requested = list(dict.fromkeys(run_names))
    selected_design = design[design["run_name"].isin(requested)]
    if len(selected_design) != len(requested):
        missing = sorted(set(requested) - set(selected_design["run_name"]))
        raise ValueError(f"Run names are absent from the frozen design: {missing}")
    if not selected_design["source"].eq("new").all():
        raise ValueError("Only new design rows can be materialized")

    api = wandb.Api(timeout=timeout)
    training_runs = _runs_in_groups(api, TRAIN_PROJECT, TRAIN_GROUPS)
    evaluation_runs = _runs_in_groups(api, EVAL_PROJECT, (EVAL_GROUP,))
    table9 = _table9_by_run_name(evaluation_runs)

    result_rows: list[dict[str, object]] = []
    uncheatable_rows: list[dict[str, object]] = []
    table9_rows: list[dict[str, object]] = []
    for design_row in selected_design.sort_values("run_order").to_dict("records"):
        run_name = str(design_row["run_name"])
        training, runtime = _selected_training(
            training_runs,
            design_row,
            expected_tpu_type=expected_tpu_type,
            expected_tpu_zone=expected_tpu_zone,
        )
        training_run = training["run"]
        for position, (component, value) in enumerate(
            zip(heldout.UNCHEATABLE_COMPONENTS, training["values"], strict=True)
        ):
            uncheatable_rows.append(
                {
                    "run_order": int(design_row["run_order"]),
                    "run_name": run_name,
                    "component_position": position,
                    "component": component,
                    "bpb": float(value),
                    "provenance": str(training["provenance"]),
                }
            )

        evaluation = table9.get(run_name)
        if evaluation is None:
            eval_id = eval_url = eval_state = ""
            table9_macro = np.nan
            table9_count = 0
        else:
            eval_run = evaluation["run"]
            eval_id, eval_url, eval_state = str(eval_run.id), str(eval_run.url), str(eval_run.state)
            table9_macro = float(evaluation["aggregate"])
            table9_count = len(heldout.table9_components())
            for position, (component, value) in enumerate(
                zip(heldout.table9_components(), evaluation["values"], strict=True)
            ):
                table9_rows.append(
                    {
                        "run_order": int(design_row["run_order"]),
                        "run_name": run_name,
                        "component_position": position,
                        "component": component,
                        "bpb": float(value),
                        "provenance": "wandb_summary_validated",
                    }
                )

        result_rows.append(
            {
                "run_order": int(design_row["run_order"]),
                "run_name": run_name,
                "training_wandb_run_id": str(training_run.id),
                "training_wandb_url": str(training_run.url),
                "training_wandb_state": str(training_run.state),
                **runtime,
                "uncheatable_bpb": float(training["aggregate"]),
                "uncheatable_provenance": str(training["provenance"]),
                "uncheatable_source_uri": str(training["source_uri"]),
                "uncheatable_component_count": len(heldout.UNCHEATABLE_COMPONENTS),
                "table9_eval_run_id": eval_id,
                "table9_eval_url": eval_url,
                "table9_eval_state": eval_state,
                "table9_macro_bpb": table9_macro,
                "table9_component_count": table9_count,
            }
        )

    results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    uncheatable_components = pd.DataFrame(uncheatable_rows, columns=COMPONENT_COLUMNS)
    table9_components = pd.DataFrame(table9_rows, columns=COMPONENT_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoints._atomic_csv(results, output_dir / "heldout_results.csv")
    endpoints._atomic_csv(uncheatable_components, output_dir / "uncheatable_components.csv")
    endpoints._atomic_csv(table9_components, output_dir / "table9_components.csv")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "observed_at": datetime.now(UTC).isoformat(),
        "frozen_design_sha256": launcher.frozen_design_sha256(DESIGN_TABLE),
        "runtime_placement": {"tpu_type": expected_tpu_type, "tpu_zone": expected_tpu_zone},
        "run_names": requested,
        "counts": {
            "runs": len(results),
            "uncheatable_complete": int(results["uncheatable_component_count"].eq(7).sum()),
            "table9_complete": int(results["table9_component_count"].eq(51).sum()),
        },
        "sources": {
            "training_project": TRAIN_PROJECT,
            "training_groups": list(TRAIN_GROUPS),
            "evaluation_project": EVAL_PROJECT,
            "evaluation_group": EVAL_GROUP,
            "design_table": str(DESIGN_TABLE.relative_to(REPO_ROOT)),
        },
    }
    (output_dir / "heldout_materialization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", action="append", default=[])
    parser.add_argument("--wandb-timeout", type=int, default=180)
    args = parser.parse_args()
    run_names = args.run_name or [launcher.CANARY_RUN_NAME]
    print(json.dumps(materialize(args.output_dir, run_names, args.wandb_timeout)["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
