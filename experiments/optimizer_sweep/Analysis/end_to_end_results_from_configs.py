import json
import os
import pickle
from typing import Dict, List, Tuple

import tqdm

from marin.optimizer_sweep.utils_simp import (
    calculate_data_tag,
    check_baseline_run,
    create_configs,
    grab_best_run,
    grab_run,
)
from experiments.optimizer_sweep.Analysis.PhaseI.results_io import (
    CACHE_FILE_DEFAULT,
    RESULTS_DIR_DEFAULT,
    persist_result,
)


BASELINE_ROOT = os.path.join(
    "experiments", "optimizer_sweep", "baseline_config"
)
SWEEP_ROOT = os.path.join(
    "experiments", "optimizer_sweep", "sweep_grids"
)


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
    with tqdm.tqdm(total=len(optimizers), desc="Optimizers") as pbar:
        for optimizer in optimizers:
            optimizer_lower = optimizer.lower()
            if optimizer_lower not in key_of_optimizer:
                pbar.update(1)
                continue

            keys = key_of_optimizer[optimizer_lower]
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

                    # Tags and data size
                    target_data, data_size = calculate_data_tag(model_size, chin_ratio)
                    tags = (model_size, data_size, optimizer_lower)

                    # Best runs from W&B
                    current_best_config, approximate_best_config_list, min_loss = grab_best_run(
                        keys, tags, return_loss=True, thshold=6e-3
                    )
                    if not approximate_best_config_list:
                        continue

                    # Evaluate ablations for each near-best config
                    current_num_left = 10 ** 9
                    best_payload = {}
                    for candidate in approximate_best_config_list:
                        num_left, cfg_to_loss, cfg_to_name = _config_to_loss_and_id(
                            candidate, sweep_grids, target_data, tags
                        )
                        if num_left < current_num_left:
                            current_num_left = num_left
                            best_payload = {
                                "result": cfg_to_loss,
                                "name": cfg_to_name,
                                "num_left": num_left,
                                "best_config": candidate,
                                "min_loss": min_loss,
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

    # Write combined pickle into Results
    results_root_pkl = os.path.join(RESULTS_DIR_DEFAULT, "1d_losses.pkl")
    os.makedirs(os.path.dirname(results_root_pkl), exist_ok=True)
    with open(results_root_pkl, "wb") as f:
        pickle.dump(combined_results, f)


if __name__ == "__main__":
    main()


