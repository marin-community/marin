import json
import pickle
import os
from typing import Dict, List, Tuple

import tqdm

from marin.optimizer_sweep.utils_simp import (
    calculate_data_tag,
    check_baseline_run,
    create_configs,
    grab_best_run,
)
from results_io import (
    CACHE_FILE_DEFAULT,
    RESULTS_DIR_DEFAULT,
    get_cache_key,
    persist_result,
    save_to_results_dir,
)


def load_cache(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def config_to_loss(
    baseline_config: Dict,
    sweep_grids: Dict,
    target_data: str,
    tags: Tuple[str, str, str],
) -> Tuple[int, Dict, Dict]:
    """Return (num_left, config_to_loss, config_to_name). Mirrors logic in 1d_losses."""
    target_steps, config_in_dict = create_configs(
        baseline_config, sweep_grids, target_data=target_data
    )
    config_to_loss_map: Dict = {}
    config_to_name: Dict = {}
    num_left = 0
    for config in config_in_dict:
        exist, loss, name = check_baseline_run(config, tags, return_loss=True)
        num_left += 1 if not exist else 0
        different_key = [key for key in config.keys() if config[key] != baseline_config[key]]
        if len(different_key) == 1:
            key = (different_key[0], config[different_key[0]])
            config_to_loss_map[key] = loss
            config_to_name[key] = name
        else:
            config_to_loss_map["Baseline"] = loss
            config_to_name["Baseline"] = name
    return num_left, config_to_loss_map, config_to_name


def _maybe_parse_number(value_str: str):
    try:
        if value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
            return int(value_str)
        # Attempt float parse
        return float(value_str)
    except Exception:
        return value_str


def _rehydrate_json_friendly_payload(payload: Dict) -> Dict:
    """Turn JSON-friendly payload (string keys like 'param=value') back into tuple-keyed dicts.

    This targets the 'result' and 'name' maps only.
    """
    if not payload:
        return payload

    def convert_map(m: Dict) -> Dict:
        restored = {}
        for k, v in m.items():
            if k == "Baseline":
                restored[k] = v
                continue
            if isinstance(k, str) and "=" in k:
                param, val_str = k.split("=", 1)
                restored[(param, _maybe_parse_number(val_str))] = v
            else:
                restored[k] = v
        return restored

    restored_payload = dict(payload)
    if isinstance(restored_payload.get("result"), dict):
        restored_payload["result"] = convert_map(restored_payload["result"])
    if isinstance(restored_payload.get("name"), dict):
        restored_payload["name"] = convert_map(restored_payload["name"])
    return restored_payload


key_of_optimizer: Dict[str, List[str]] = {
    "mars": [
        "learning_rate",
        "weight_decay",
        "min_lr_ratio",
        "warmup",
        "beta1",
        "beta2",
        "gamma",
        "epsilon",
        "max_grad_norm",
        "train_batch_size",
    ],
    "sophia": [
        "learning_rate",
        "weight_decay",
        "warmup",
        "beta1",
        "beta2",
        "gamma",
        "epsilon",
        "train_batch_size",
    ],
    "muon": [
        "learning_rate",
        "weight_decay",
        "min_lr_ratio",
        "warmup",
        "momentum",
        "beta1",
        "beta2",
        "epsilon",
        "muon_epsilon",
        "max_grad_norm",
        "lr_schedule",
        "muon_to_adam_lr",
        "decay",
        "train_batch_size",
    ],
    "lion": [
        "learning_rate",
        "weight_decay",
        "min_lr_ratio",
        "warmup",
        "beta1",
        "beta2",
        "max_grad_norm",
        "train_batch_size",
    ],
    "nadamw": [
        "learning_rate",
        "weight_decay",
        "min_lr_ratio",
        "warmup",
        "beta1",
        "beta2",
        "epsilon",
        "max_grad_norm",
        "nesterov",
        "train_batch_size",
    ],
    "kron": [
        "learning_rate",
        "weight_decay",
        "beta1",
        "preconditioner_lr",
        "preconditioner_init_scale",
        "max_grad_norm",
        "normalize_grads",
        "partition_grads_into_blocks",
        "block_size",
        "preconditioner_update_probability",
        "update_prob_flat_start",
        "warmup",
        "min_lr_ratio",
        "train_batch_size",
    ],
    "scion": [
        "learning_rate",
        "weight_decay",
        "min_lr_ratio",
        "warmup",
        "momentum",
        "beta1",
        "scion_epsilon",
        "max_grad_norm",
        "lr_schedule",
        "scion_to_signum_lr",
        "decay",
        "train_batch_size",
    ],
    "cautious": [
        "learning_rate",
        "weight_decay",
        "min_lr_ratio",
        "warmup",
        "beta1",
        "beta2",
        "epsilon",
        "max_grad_norm",
        "train_batch_size",
    ],
    "soape": [
        "learning_rate",
        "weight_decay",
        "min_lr_ratio",
        "warmup",
        "beta1",
        "beta2",
        "shampoo_beta",
        "precondition_frequency",
        "partition_grads_into_blocks",
        "block_size",
        "epsilon",
        "max_grad_norm",
        "train_batch_size",
    ],
    "adamw": [
        "learning_rate",
        "weight_decay",
        "min_lr_ratio",
        "warmup",
        "beta1",
        "beta2",
        "epsilon",
        "max_grad_norm",
        "nesterov",
        "train_batch_size",
    ],
    "mini": [
        "learning_rate",
        "weight_decay",
        "min_lr_ratio",
        "warmup",
        "beta1",
        "beta2",
        "epsilon",
        "max_grad_norm",
        "train_batch_size",
    ],
}


def process_and_persist(
    optimizer: str,
    model_and_data_size: List[Tuple[str, str, int]],
    sweep_configs: Dict,
    cache: Dict,
) -> Dict:
    keys = key_of_optimizer[optimizer]
    collected: Dict = {}

    for model_size, original_data_size, chinchilla_ratio in tqdm.tqdm(
        model_and_data_size, desc=f"{optimizer}", leave=False
    ):
        # Tags use the original provided data size string, matching existing logic
        tags = (model_size, original_data_size, optimizer)
        target_data, data_size = calculate_data_tag(model_size, chinchilla_ratio)

        cache_key = get_cache_key(optimizer, model_size, data_size, chinchilla_ratio)
        if cache_key in cache and isinstance(cache[cache_key], dict):
            # If cached, just mirror into Results/ for accessibility
            save_to_results_dir(
                optimizer=optimizer,
                model_size=model_size,
                chinchilla_ratio=chinchilla_ratio,
                result=cache[cache_key],
            )
            # Also collect into combined results, rehydrating keys
            collected[(model_size, chinchilla_ratio, optimizer)] = _rehydrate_json_friendly_payload(
                cache[cache_key]
            )
            continue

        optimizer_configs = sweep_configs.get(optimizer.lower(), {})
        model_configs = optimizer_configs.get(model_size, {})
        if not optimizer_configs or not model_configs:
            continue

        current_best_config, approximate_best_config_list, min_loss = grab_best_run(
            keys, tags, return_loss=True, thshold=6e-3
        )
        if not approximate_best_config_list:
            continue

        sweep_config = model_configs.get(str(chinchilla_ratio), [{}])[0]
        sweep_grids = sweep_config.get("sweep_grids", None)
        if not sweep_grids:
            continue

        current_num_left = 10**9
        best_payload: Dict = {}
        for best_config in approximate_best_config_list:
            num_left, cfg_to_loss, cfg_to_name = config_to_loss(
                best_config, sweep_grids, target_data, tags
            )
            if num_left < current_num_left:
                current_num_left = num_left
                best_payload = {
                    "result": cfg_to_loss,
                    "name": cfg_to_name,
                    "num_left": num_left,
                    "best_config": best_config,
                    "min_loss": min_loss,
                }

        if best_payload:
            persist_result(
                optimizer=optimizer,
                model_size=model_size,
                chinchilla_ratio=chinchilla_ratio,
                data_size=data_size,
                result=best_payload,
            )
            collected[(model_size, chinchilla_ratio, optimizer)] = best_payload

    return collected


def main() -> None:
    with open(
        "experiments/optimizer_sweep/Analysis/PhaseI/sweep_configurations.json", "r"
    ) as f:
        sweep_configs = json.load(f)

    model_and_data_size: List[Tuple[str, str, int]] = [
        ("130m", "2B", 1),
        ("130m", "5B", 2),
        ("130m", "10B", 4),
        ("130m", "21B", 8),
        ("300m", "6B", 1),
        ("520m", "10B", 1),
    ]

    cache = load_cache(CACHE_FILE_DEFAULT)
    combined_results: Dict[str, Dict] = {}

    with tqdm.tqdm(total=len(key_of_optimizer), desc="Overall", leave=True) as pbar:
        for optimizer in key_of_optimizer.keys():
            collected = process_and_persist(
                optimizer=optimizer,
                model_and_data_size=model_and_data_size,
                sweep_configs=sweep_configs,
                cache=cache,
            )
            combined_results[optimizer] = collected
            pbar.update(1)

    # Write combined pickle(s) into the Results directory following the Phase structure
    results_root_pkl = os.path.join(RESULTS_DIR_DEFAULT, "1d_losses.pkl")
    results_phase_pkl = os.path.join(RESULTS_DIR_DEFAULT, "PhaseI", "1d_losses.pkl")
    os.makedirs(os.path.dirname(results_root_pkl), exist_ok=True)
    os.makedirs(os.path.dirname(results_phase_pkl), exist_ok=True)
    with open(results_root_pkl, "wb") as f:
        pickle.dump(combined_results, f)
    with open(results_phase_pkl, "wb") as f:
        pickle.dump(combined_results, f)


if __name__ == "__main__":
    main()


