# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any

from transformers import AutoTokenizer

from .environments.aqua_rat_env import AquaRatEnv
from .environments.marin_env import MarinEnv
from .environments.math_env import MathEnv
from .environments.numina_math_env import NuminaMathEnv
from .environments.olym_math_env import OlymMathEnv
from .environments.olympiad_bench_env import OlympiadBenchEnv
from .environments.open_math_reasoning_env import OpenMathReasoningEnv
from .environments.orz_env import ORZEnv
from .environments.svamp_env import SVAMPEnv
from .environments.swe_bench_env import SWEBenchEnv

# Specify environments here
ENVIRONMENT_NAME_TO_CLASS = {
    "aqua_rat": AquaRatEnv,
    "math": MathEnv,
    "numina_math": NuminaMathEnv,
    "olym_math": OlymMathEnv,
    "olympiad_bench": OlympiadBenchEnv,
    "open_math_reasoning": OpenMathReasoningEnv,
    "orz": ORZEnv,
    "svamp": SVAMPEnv,
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


def load_environments_from_config(json_path: str, tokenizer: AutoTokenizer) -> list[tuple[str, MarinEnv]]:
    """
    Load environment entries from a JSON config file.

    Returns:
        List of (environment_name, environment) tuples
    """
    with open(json_path, "r", encoding="utf-8") as f:
        conf = json.load(f)
    entries = conf.get("entries", [])
    if not entries:
        raise ValueError("'entries' list is empty in environment config.")

    envs = []
    for entry in entries:
        if "environment" not in entry:
            raise ValueError("Each entry must have an 'environment' field.")
        envs.append((entry["environment"], load_environment_from_spec(entry["environment"], tokenizer)))
    return envs
