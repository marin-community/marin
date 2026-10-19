import yaml
import os
from pathlib import Path
from loguru import logger
import json
import re


def check_valid_config(config):
    assert "human_readable_id" in config and isinstance(
        config["human_readable_id"], str
    ), "human_readable_id not found in config"
    assert "model" in config and isinstance(config["model"], str), "model not found in config"
    assert "api" in config and isinstance(config["api"], str), "api not found in config"


def load_configs(root_dir, remove_extension=True):
    root = Path(root_dir)
    # Find all YAML files (supporting both .yaml and .yml extensions) recursively.
    yaml_files = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))

    output_configs = dict()

    for file_path in yaml_files:
        with file_path.open("r") as f:
            config_data = yaml.safe_load(f)
        file_path_remove_config = str(file_path).replace(str(root) + "/", "")
        if remove_extension:
            file_path_remove_config = file_path_remove_config.replace(".yaml", "").replace(".yml", "")
        try:
            check_valid_config(config_data)
        except AssertionError as e:
            raise ValueError(f"Config not correct in {file_path}: {e}")
        output_configs[file_path_remove_config] = config_data

    return output_configs


def exclude_configs(configs, human_readable_ids, exclude_file_path):
    if exclude_file_path is None:
        return configs, human_readable_ids
    with open(exclude_file_path, "r") as f:
        exclude_regexes = [line.strip() for line in f.readlines()]
    for config_path in list(configs.keys()):
        for regex in exclude_regexes:
            if re.match(regex, config_path):
                logger.info(f"Excluding {config_path} due to {regex}")
                del configs[config_path]
                del human_readable_ids[config_path]
                break
    return configs, human_readable_ids


def extract_existing_configs(
    comp,
    root_dir,
    root_dir_configs,
    root_dir_competition_configs,
    exclude_file_path=None,
    allow_non_existing_judgment=False,
):
    with open(f"{root_dir_competition_configs}/{comp}.yaml", "r") as f:
        competition_config = yaml.safe_load(f)

    is_final_answer = competition_config.get("final_answer", True)

    all_configs = load_configs(root_dir_configs)

    configs = dict()
    human_readable_ids = dict()
    for config_path in all_configs:
        if os.path.exists(os.path.join(root_dir, comp, config_path)):
            exists = True
            if not is_final_answer:
                for file in os.listdir(os.path.join(root_dir, comp, config_path)):
                    with open(os.path.join(root_dir, comp, config_path, file), "r") as f:
                        data = json.load(f)
                    if "judgment" not in data and not allow_non_existing_judgment:
                        logger.warning(f"Judgment not found in {file}")
                        exists = False
                        break
            if exists:
                configs[config_path] = all_configs[config_path]
                human_readable_ids[config_path] = all_configs[config_path]["human_readable_id"]

    if len(set(human_readable_ids.values())) != len(human_readable_ids):
        # find for which config the human readable id is duplicated
        duplicated = set()
        for k, v in human_readable_ids.items():
            if v in duplicated:
                logger.error(f"Duplicate human readable id {v} found in {k}")
            duplicated.add(v)
        raise ValueError("Duplicate human readable ids. Website currently does not support this.")

    configs, human_readable_ids = exclude_configs(configs, human_readable_ids, exclude_file_path)
    return configs, human_readable_ids
