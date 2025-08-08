import json
import os
import pickle
from typing import Dict, Tuple

from results_io import RESULTS_DIR_DEFAULT


INPUT_PKL = os.path.join(
    "experiments", "optimizer_sweep", "Analysis", "PhaseI", "1d_losses.pkl"
)


def main() -> None:
    with open(INPUT_PKL, "rb") as f:
        data: Dict[str, Dict[Tuple[str, int, str], Dict]] = pickle.load(f)

    for optimizer_name, optimizer_data in data.items():
        if not optimizer_data:
            continue

        for config_key, config_info in optimizer_data.items():
            # Expecting config_key like (model_size, chinchilla_ratio, optimizer)
            if not isinstance(config_key, tuple) or len(config_key) < 2:
                continue
            model_size = str(config_key[0])
            chinchilla_ratio = str(config_key[1])

            out_dir = os.path.join(
                RESULTS_DIR_DEFAULT, optimizer_name, model_size, chinchilla_ratio
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "result.json")

            # Build ablations from 1d_losses payload
            result_map = config_info.get("result", {})
            name_map = config_info.get("name", {})

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

            # Merge with existing result.json if present
            existing: Dict = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, "r") as f:
                        existing = json.load(f)
                except Exception:
                    existing = {}

            merged = dict(existing)
            if baseline is not None:
                merged["baseline"] = baseline
            if ablations:
                merged["ablations"] = ablations
            # Optionally keep best_config and min_loss if present in pickle
            if "best_config" in config_info and "best_config" not in merged:
                merged["best_config"] = config_info["best_config"]
            if "min_loss" in config_info and "min_loss" not in merged:
                merged["min_loss"] = config_info["min_loss"]

            with open(out_path, "w") as f:
                json.dump(merged, f, indent=2)


if __name__ == "__main__":
    main()


