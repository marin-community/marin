# %%
# This script launches optimizer sweep experiments by reading configuration from JSON files
# The json files contain baseline configurations and sweep grids
# For each optimizer (e.g. AdamW, Muon, Sophia), model size, and target Chinchilla ratio,
# this script:
# 1. Loads the baseline config JSON containing default hyperparameters
# 2. Loads the sweep grids JSON containing parameter ranges to explore
# 3. Launches training runs using the template.py framework with these configurations
import argparse
import json
from pathlib import Path
from typing import Any
from marin.optimizer_sweep.template import template


def _iter_model_size_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


OPT_SWEEP_DIR = Path("experiments/optimizer_sweep")


def _find_jsons_new_layout(
    optimizer: str, chinchilla: int | float, model_size: str | None
) -> list[tuple[Path, Path | None]]:
    """
    New layout:
      baseline_config/{optimizer}/{model_size}/{chinchilla}/*.json
      sweep_grids/{optimizer}/{model_size}/{chinchilla}/*.json
    """
    base_baseline = OPT_SWEEP_DIR / "baseline_config" / optimizer.lower()
    base_grids = OPT_SWEEP_DIR / "sweep_grids" / optimizer.lower()

    candidate_dirs: list[Path]
    if model_size is None:
        candidate_dirs = [d / str(chinchilla) for d in _iter_model_size_dirs(base_baseline)]
    else:
        candidate_dirs = [base_baseline / model_size / str(chinchilla)]

    pairs: list[tuple[Path, Path | None]] = []
    for dir_ in candidate_dirs:
        if not dir_.exists():
            continue
        grids_dir = base_grids / dir_.relative_to(base_baseline)
        for baseline_json in sorted(dir_.glob("*.json")):
            grids_json = grids_dir / baseline_json.name
            pairs.append((baseline_json, grids_json if grids_json.exists() else None))
    return pairs


def _find_jsons_old_layout(optimizer: str, chinchilla: int | float) -> list[tuple[Path, Path | None]]:
    """
    Backward-compatible scan of old layout with Phase level:
      baseline_config/Phase*/{optimizer}/{chinchilla}/*.json
      sweep_grids/Phase*/{optimizer}/{chinchilla}/*.json
    """
    pairs: list[tuple[Path, Path | None]] = []
    baseline_root = OPT_SWEEP_DIR / "baseline_config"
    grids_root = OPT_SWEEP_DIR / "sweep_grids"
    if not baseline_root.exists():
        return []
    for phase_dir in sorted([p for p in baseline_root.iterdir() if p.is_dir()]):
        base = phase_dir / optimizer.lower() / str(chinchilla)
        if not base.exists():
            continue
        grids_dir = grids_root / phase_dir.name / optimizer.lower() / str(chinchilla)
        for baseline_json in sorted(base.glob("*.json")):
            grids_json = grids_dir / baseline_json.name
            pairs.append((baseline_json, grids_json if grids_json.exists() else None))
    return pairs


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch optimizer sweep jobs from extracted JSONs")
    parser.add_argument("optimizer", type=str, help="Optimizer name, e.g., adamw, muon, sophia, etc.")
    parser.add_argument("chinchilla", type=float, help="Target chinchilla ratio (e.g., 1, 2, 4, 8, 16)")
    parser.add_argument(
        "model_size", type=str, default=None, help="Optional model size directory to filter (e.g., 130M, 300M, 1.2B)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print intended launches without executing")
    parser.add_argument("--force-run", action="store_true", help="Pass force_run=True to template")
    parser.add_argument("--tpu-type", type=str, default="v5litepod-128", help="TPU type")
    parser.add_argument("--debug", action="store_true", help="Enable template DEBUG_MODE")
    parser.add_argument("--random-suffix", type=str, default=None, help="Random suffix for template prefix")
    parser.add_argument("--name-filter", type=str, default=None, help="Substring filter on JSON filename stem")

    args = parser.parse_args()

    optimizer = args.optimizer.lower()
    chinchilla = int(args.chinchilla) if args.chinchilla.is_integer() else args.chinchilla  # keep 1 vs 1.0 tidy

    pairs = _find_jsons_new_layout(optimizer, chinchilla, args.model_size)
    if not pairs:
        raise SystemExit("No JSONs found for given inputs.")

    to_launch: list[tuple[str, float | int, str, dict[str, Any], dict[str, Any]]] = []

    for baseline_path, grids_path in pairs:
        if args.name_filter and args.name_filter not in baseline_path.stem:
            continue
        baseline_obj = load_json(baseline_path)
        sweep_obj = load_json(grids_path) if grids_path is not None else {"sweep_grids": {}}

        # Prefer CLI optimizer; sanity check JSON if present
        json_optimizer = str(baseline_obj.get("optimizer_name", optimizer)).lower()
        if json_optimizer != optimizer:
            # Skip mismatched entries silently unless user wants strict matching
            continue

        model_size = baseline_obj.get("model_size")
        target_chinchilla = baseline_obj.get("target_chinchilla", chinchilla)
        baseline_config = baseline_obj.get("baseline_config", {})
        sweep_grids = sweep_obj.get("sweep_grids", {})

        if model_size is None:
            # JSON should always have model_size, skip otherwise
            continue

        to_launch.append((model_size, target_chinchilla, optimizer, baseline_config, sweep_grids))

    if not to_launch:
        raise SystemExit("No matching JSON payloads to launch after filtering.")

    for model_size, target_chinchilla, optimizer_name, baseline_config, sweep_grids in to_launch:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "model_size": model_size,
                        "target_chinchilla": target_chinchilla,
                        "optimizer": optimizer_name,
                        "tpu_type": args.tpu_type,
                        "force_run": args.force_run,
                        "debug": args.debug,
                        "random_suffix": args.random_suffix,
                        "baseline_config": baseline_config,
                        "sweep_grids": sweep_grids,
                    }
                )
            )
        else:
            template(
                model_size,
                target_chinchilla,
                optimizer_name,
                baseline_config,
                sweep_grids,
                tpu_type=args.tpu_type,
                DEBUG_MODE=args.debug,
                random_suffix=args.random_suffix,
                force_run=args.force_run,
            )


if __name__ == "__main__":
    main()
