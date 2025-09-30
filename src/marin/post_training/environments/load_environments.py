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
from os import PathLike
from typing import Any

from transformers import AutoTokenizer

from .aqua_rat_env import AquaRatEnv
from .math_env import MathEnv
from .numina_math_env import NuminaMathEnv
from .olym_math_env import OlymMathEnv
from .olympiad_bench_env import OlympiadBenchEnv
from .open_math_reasoning_env import OpenMathReasoningEnv
from .orz_env import ORZEnv
from .svamp_env import SVAMPEnv

try:
    from .swe_bench_env import SWEBenchEnv
except ImportError:
    # not available on mac, ignore for testing.
    SWEBenchEnv = None  # type: ignore

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
}

if SWEBenchEnv is not None:
    ENVIRONMENT_NAME_TO_CLASS["swe_bench"] = SWEBenchEnv


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


def load_environment_from_spec(spec: str, tokenizer: AutoTokenizer):
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


def load_environments_from_config(json_path: PathLike, tokenizer: AutoTokenizer):
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
