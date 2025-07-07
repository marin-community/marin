import json
from typing import Any

from transformers import AutoTokenizer

from .environments.marin_env import MarinEnv
from .environments.math_env import MathEnv
from .environments.swe_bench_env import SWEBenchEnv
from .environments.olym_math_env import OlymMathEnv

# Specify environments here
ENVIRONMENT_NAME_TO_CLASS = {
    "math": MathEnv,
    "olym_math": OlymMathEnv,
    "swe_bench": SWEBenchEnv,
}


def str_to_val(v: str) -> Any:
    """Best effort convert string to bool/int/float, else keep str."""
    v = v.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def load_environment_from_spec(spec: str, tokenizer: AutoTokenizer) -> MarinEnv:
    """
    Instantiate an environment from a spec string like 'OlymMathEnv:difficulty=easy,language=en'
    """
    cls_name, _, arg_str = spec.partition(":")
    if cls_name not in ENVIRONMENT_NAME_TO_CLASS:
        raise ValueError(f"Unknown environment class '{cls_name}'.")

    env_cls = ENVIRONMENT_NAME_TO_CLASS[cls_name]
    kwargs: dict[str, Any] = {}
    if arg_str:
        for pair in arg_str.split(","):
            if not pair.strip():
                continue
            key, value = map(str.strip, pair.split("="))
            kwargs[key] = str_to_val(value)
    return env_cls(tokenizer=tokenizer, **kwargs)


def load_environments_from_config(conf_path: str, tokenizer: AutoTokenizer) -> MarinEnv:
    """
    Load first environment entry from a JSON config file.

    # TODO: support multiple environments. For now, we only load the first one.
    """
    with open(conf_path, "r", encoding="utf-8") as f:
        conf = json.load(f)
    entries = conf.get("entries", [])
    if not entries:
        raise ValueError("'entries' list is empty in environment config.")

    first_entry = entries[0]
    if "environment" not in first_entry:
        raise ValueError("Each entry must have an 'environment' field.")
    return load_environment_from_spec(first_entry["environment"], tokenizer)
