import json
import os
import pickle
from typing import Dict, List, Tuple, Any

import tqdm

from marin.optimizer_sweep.utils_simp import (
    calculate_data_tag,
    check_baseline_run,
    create_configs,
    grab_best_run,
    grab_run,
)
RESULTS_DIR_DEFAULT = "experiments/optimizer_sweep/Analysis/Results"


def _stringify_key(key: Any) -> str:
    """Convert tuple keys like (param, value) into a readable string 'param=value'."""
    if isinstance(key, tuple) and len(key) == 2:
        return f"{key[0]}={key[1]}"
    return str(key)


def _make_json_friendly(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert result dict so keys are JSON-serializable and easy to read."""
    json_ready: Dict[str, Any] = {}
    for top_key in [
        "result",
        "name",
        "num_left",
        "best_config",
        "min_loss",
        "approximate_best_config_list",
    ]:
        if top_key not in result_dict:
            continue
        value = result_dict[top_key]
        if isinstance(value, dict):
            json_ready[top_key] = {_stringify_key(k): v for k, v in value.items()}
        else:
            json_ready[top_key] = value
    return json_ready

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_to_results_dir(
    optimizer: str,
    model_size: str,
    chinchilla_ratio: int,
    result: Dict[str, Any],
    results_dir: str = RESULTS_DIR_DEFAULT,
) -> str:
    """Save the given result dict to Results/{optimizer}/{model_size}/{chinchilla_ratio}/result.json.

    Returns the path to the written JSON file.
    """
    output_dir = os.path.join(results_dir, str(optimizer), str(model_size), str(chinchilla_ratio))
    _ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "result.json")
    with open(output_path, "w") as f:
        json.dump(_make_json_friendly(result), f, indent=2)
    return output_path
def persist_result(
    optimizer: str,
    model_size: str,
    chinchilla_ratio: int,
    data_size: str,
    result: Dict[str, Any],
    *,
    results_dir: str = RESULTS_DIR_DEFAULT,
) -> Tuple[str, str]:
    """Persist a single best-result payload:
    - Saves human-friendly JSON to Results/{optimizer}/{model_size}/{chinchilla_ratio}/result.json
    - Updates the wandb cache keyed by md5(optimizer, model_size, data_size, chinchilla_ratio)

    Returns (results_path, cache_key).
    """
    results_path = save_to_results_dir(
        optimizer=optimizer,
        model_size=model_size,
        chinchilla_ratio=chinchilla_ratio,
        result=result,
        results_dir=results_dir,
    )
    # No cache implementation yet; return a placeholder for the cache key
    cache_key = None
    return results_path, cache_key




BASELINE_ROOT = os.path.join(
    "experiments", "optimizer_sweep", "baseline_config"
)
SWEEP_ROOT = os.path.join(
    "experiments", "optimizer_sweep", "sweep_grids"
)


def _first_json_in_dir(path: str) -> str:
    for fname in os.listdir(path):
        if fname.endswith(".json"):
            return os.path.join(path, fname)
    raise FileNotFoundError(f"No JSON found in {path}")


def _load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _maybe_get_run_id(config: Dict, tags: Tuple[str, str, str]):
    try:
        run = grab_run(config, tags)
        return getattr(run, "id", None) if run is not None else None
    except Exception:
        return None


def _config_to_loss_and_id(
    baseline_config: Dict, sweep_grids: Dict, target_data: int, tags: Tuple[str, str, str]
) -> Tuple[int, Dict, Dict]:
    target_steps, config_in_dict = create_configs(
        baseline_config, sweep_grids, target_data=target_data
    )
    config_to_loss: Dict = {}
    config_to_name: Dict = {}
    num_left = 0
    for config in config_in_dict:
        exist, loss = check_baseline_run(config, tags, return_loss=True)
        num_left += 0 if exist else 1
        run_id = _maybe_get_run_id(config, tags) if exist else None

        different_key = [key for key in config.keys() if config[key] != baseline_config[key]]
        if len(different_key) == 1:
            key = (different_key[0], config[different_key[0]])
            config_to_loss[key] = loss
            config_to_name[key] = run_id
        else:
            config_to_loss["Baseline"] = loss
            config_to_name["Baseline"] = run_id

    return num_left, config_to_loss, config_to_name


def _augment_result_json_with_ablations(
    results_path: str, result_map: Dict, name_map: Dict
) -> None:
    baseline = None
    if "Baseline" in result_map:
        baseline = {
            "loss": result_map.get("Baseline"),
            "wandb_id": name_map.get("Baseline"),
        }
    ablations = []
    for key, loss in result_map.items():
        if key == "Baseline":
            continue
        if isinstance(key, tuple) and len(key) == 2:
            param, value = key
            ablations.append(
                {
                    "param": str(param),
                    "value": value,
                    "loss": loss,
                    "wandb_id": name_map.get(key),
                }
            )

    existing: Dict = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    merged = dict(existing)
    if baseline is not None:
        merged["baseline"] = baseline
    if ablations:
        merged["ablations"] = ablations

    with open(results_path, "w") as f:
        json.dump(merged, f, indent=2)


def main() -> None:
    combined_results: Dict[str, Dict] = {}

    optimizers = [d for d in os.listdir(BASELINE_ROOT) if os.path.isdir(os.path.join(BASELINE_ROOT, d))]
    optimizers = ["scion"]
    with tqdm.tqdm(total=len(optimizers), desc="Optimizers") as pbar:
        for optimizer in optimizers:
            optimizer_lower = optimizer.lower()
            optimizer_dir = os.path.join(BASELINE_ROOT, optimizer)
            model_sizes = [d for d in os.listdir(optimizer_dir) if os.path.isdir(os.path.join(optimizer_dir, d))]

            collected_for_opt: Dict = {}

            for model_dir in model_sizes:
                chin_dirs = [d for d in os.listdir(os.path.join(optimizer_dir, model_dir)) if os.path.isdir(os.path.join(optimizer_dir, model_dir, d))]
                for chin_ratio_str in chin_dirs:
                    baseline_json_path = _first_json_in_dir(os.path.join(optimizer_dir, model_dir, chin_ratio_str))
                    baseline_payload = _load_json(baseline_json_path)

                    baseline_config = baseline_payload["baseline_config"]
                    model_size = baseline_payload["model_size"]  # e.g., "130m"
                    chin_ratio = int(baseline_payload["target_chinchilla"])  # e.g., 1,2,4,8

                    # Matching sweep grid
                    sweep_json_path = _first_json_in_dir(os.path.join(SWEEP_ROOT, optimizer, model_dir, chin_ratio_str))
                    sweep_payload = _load_json(sweep_json_path)
                    sweep_grids = sweep_payload.get("sweep_grids", {})

                    # Dynamically determine keys from sweep_grids and ensure required fields
                    keys_set = set(sweep_grids.keys())
                    # Always include fields needed to build configs and tags
                    keys_set.update({"warmup", "train_batch_size"})
                    keys = list(keys_set)

                    # Tags and data size
                    target_data, data_size = calculate_data_tag(model_size, chin_ratio)
                    tags = (model_size, data_size, optimizer_lower)

                    # Best runs from W&B
                    current_best_config, approximate_best_config_list, min_loss = grab_best_run(
                        keys, tags, return_loss=True, thshold=5e-3
                    )
                    if not approximate_best_config_list:
                        continue

                    # Evaluate ablations for each near-best config
                    current_num_left = 10 ** 9
                    best_payload = {}
                    for candidate in approximate_best_config_list:
                        # Merge candidate with full baseline so best_config has every key baseline has
                        candidate_full: Dict = dict(baseline_config)
                        if candidate is not None:
                            candidate_full.update(candidate)
                        num_left, cfg_to_loss, cfg_to_name = _config_to_loss_and_id(
                            candidate_full, sweep_grids, target_data, tags
                        )
                        if num_left < current_num_left:
                            current_num_left = num_left
                            best_payload = {
                                "name": cfg_to_name,
                                "result": cfg_to_loss,
                                "num_left": num_left,
                                "best_config": candidate_full,
                                "min_loss": min_loss,
                                "approximate_best_config_list": approximate_best_config_list,
                            }

                    if best_payload:
                        # Persist and also enrich JSON with baseline/ablations
                        results_path, _ = persist_result(
                            optimizer=optimizer_lower,
                            model_size=model_size,
                            chinchilla_ratio=chin_ratio,
                            data_size=data_size,
                            result=best_payload,
                        )
                        _augment_result_json_with_ablations(
                            results_path, best_payload["result"], best_payload["name"]
                        )
                        collected_for_opt[(model_size, chin_ratio, optimizer_lower)] = best_payload

            combined_results[optimizer_lower] = collected_for_opt
            pbar.update(1)



if __name__ == "__main__":
    main()


