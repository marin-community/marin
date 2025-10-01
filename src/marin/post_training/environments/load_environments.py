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
from .prime_intellect_env import PrimeIntellectEnv
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
    "prime_intellect": PrimeIntellectEnv,
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
    or 'prime_intellect:env_id=primeintellect/gsm8k,env_args={num_train_examples=-1,num_eval_examples=-1}'
    """
    cls_name, _, arg_str = spec.partition(":")
    if cls_name not in ENVIRONMENT_NAME_TO_CLASS:
        raise ValueError(f"Unknown environment class '{cls_name}'.")

    env_cls = ENVIRONMENT_NAME_TO_CLASS[cls_name]
    kwargs: dict[str, Any] = {}

    if arg_str:
        # Parse arguments respecting nested structures like {}, [], ()
        pairs = []
        current = []
        depth = 0

        for char in arg_str:
            if char in "{[(":
                depth += 1
                current.append(char)
            elif char in "}])":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                # Split at comma only when not inside nested structures
                if current:
                    pairs.append("".join(current))
                    current = []
            else:
                current.append(char)

        # Don't forget the last pair
        if current:
            pairs.append("".join(current))

        for pair in pairs:
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Invalid key-value pair in spec: '{pair}'")
            key, value = map(str.strip, pair.split("=", 1))

            # Special handling for JSON-like values
            if value.startswith("{") and value.endswith("}"):
                # Parse simplified dictionary format: {key1:val1,key2:val2}
                # Convert to proper Python dict string for literal_eval
                inner = value[1:-1]  # Remove outer braces
                if inner:
                    # Split by comma, respecting nested structures
                    dict_pairs = []
                    current = []
                    depth = 0
                    for char in inner:
                        if char in "{[(":
                            depth += 1
                            current.append(char)
                        elif char in "}])":
                            depth -= 1
                            current.append(char)
                        elif char == "," and depth == 0:
                            if current:
                                dict_pairs.append("".join(current))
                                current = []
                        else:
                            current.append(char)
                    if current:
                        dict_pairs.append("".join(current))

                    # Convert to proper Python dict syntax
                    formatted_pairs = []
                    for dict_pair in dict_pairs:
                        if ":" in dict_pair:
                            k, v = dict_pair.split(":", 1)
                            k = k.strip()
                            v = v.strip()
                            # Add quotes around key if it's not already quoted
                            if not (k.startswith("'") or k.startswith('"')):
                                k = f"'{k}'"
                            formatted_pairs.append(f"{k}: {v}")

                    # Reconstruct as proper Python dict string
                    kwargs[key] = "{" + ", ".join(formatted_pairs) + "}"
                else:
                    kwargs[key] = value
            else:
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
