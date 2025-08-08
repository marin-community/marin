import hashlib
import json
import os
from typing import Any, Dict, Tuple


CACHE_FILE_DEFAULT = "experiments/optimizer_sweep/Analysis/PhaseI/wandb_cache.json"
RESULTS_DIR_DEFAULT = "experiments/optimizer_sweep/Analysis/Results"


def _stringify_key(key: Any) -> str:
    """Convert tuple keys like (param, value) into a readable string 'param=value'."""
    if isinstance(key, tuple) and len(key) == 2:
        return f"{key[0]}={key[1]}"
    return str(key)


def _make_json_friendly(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert result dict so keys are JSON-serializable and easy to read."""
    json_ready: Dict[str, Any] = {}
    for top_key in ["result", "name", "num_left", "best_config", "min_loss"]:
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


def _get_cache(cache_file: str) -> Dict[str, Any]:
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: Dict[str, Any], cache_file: str) -> None:
    _ensure_dir(os.path.dirname(cache_file))
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def get_cache_key(optimizer: str, model_size: str, data_size: str, target_chinchilla: int) -> str:
    """Generate a unique cache key for the query, matching the logic in 1d_losses."""
    key_str = f"{optimizer}_{model_size}_{data_size}_{target_chinchilla}"
    return hashlib.md5(key_str.encode()).hexdigest()


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
    cache_file: str = CACHE_FILE_DEFAULT,
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

    cache = _get_cache(cache_file)
    cache_key = get_cache_key(optimizer, model_size, data_size, chinchilla_ratio)
    cache[cache_key] = _make_json_friendly(result)
    _save_cache(cache, cache_file)

    return results_path, cache_key

