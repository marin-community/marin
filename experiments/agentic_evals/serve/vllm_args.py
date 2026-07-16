"""Convert a vLLM server-config dict into CLI args + env vars.

Extracted from OT-Agent ``hpc/vllm_utils.py`` (the ``_build_vllm_cli_args``
function + its constant dicts), renamed to ``build_vllm_cli_args`` (public).
No ``hpc.*`` dependencies.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

# Fields from vllm_server/engine config that are for our system, not vLLM.
_OUR_FIELDS = {
    "num_replicas",
    "time_limit",
    "endpoint_json_path",
    "model_path",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "type",
    "max_output_tokens",
    "healthcheck_interval",
    "vllm_local",
    "model",
}

# Fields that map to different vLLM CLI arg names.
_FIELD_RENAMES = {
    "model_path": "model",
}

# Boolean flags (passed as --flag without value when True).
_BOOLEAN_FLAGS = {
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "enable_auto_tool_choice",
    "enable_expert_parallel",
    "trust_remote_code",
    "disable_log_requests",
    "enable_reasoning",
}

# Boolean flags whose vLLM internal default is True but OT-Agent defaults to OFF.
_DEFAULT_OFF_BOOLEAN_FLAGS = {
    "enable_prefix_caching",
}

# Fields that are environment variables, not CLI args.
_ENV_VAR_FIELDS = {
    "use_deep_gemm": "VLLM_USE_DEEP_GEMM",
    "use_flashinfer_sampler": "VLLM_USE_FLASHINFER_SAMPLER",
    "use_flashinfer_moe_fp16": "VLLM_USE_FLASHINFER_MOE_FP16",
    "pynccl_pyspy_on_sigusr1": "VLLM_PYNCCL_PYSPY_ON_SIGUSR1",
    "nccl_cumem_enable": "NCCL_CUMEM_ENABLE",
    "cuda_launch_blocking": "CUDA_LAUNCH_BLOCKING",
}

# Numeric env var fields (value passed through unchanged, str-coerced).
_NUMERIC_ENV_VAR_FIELDS = {
    "pynccl_trace_flush_interval_sec": "VLLM_PYNCCL_TRACE_FLUSH_INTERVAL_SEC",
    "pynccl_faulthandler_interval_sec": "VLLM_PYNCCL_FAULTHANDLER_INTERVAL_SEC",
    "nccl_debug": "NCCL_DEBUG",
    "nccl_debug_subsys": "NCCL_DEBUG_SUBSYS",
    "vllm_ray_extra_env_vars_to_copy": "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY",
}


def build_vllm_cli_args(server_config: dict) -> Tuple[List[str], Dict[str, str]]:
    """Convert vllm_server config dict to CLI args and env vars.

    Returns:
        Tuple of (cli_args list, env_vars dict)
    """
    cli_args: List[str] = []
    env_vars: Dict[str, str] = {}

    for key, value in server_config.items():
        if key in _OUR_FIELDS:
            continue

        if value is None or value == "":
            continue

        if key == "extra_args":
            if isinstance(value, list):
                cli_args.extend(str(v) for v in value)
            continue

        if key in _ENV_VAR_FIELDS:
            env_vars[_ENV_VAR_FIELDS[key]] = "1" if value else "0"
            continue

        if key in _NUMERIC_ENV_VAR_FIELDS:
            env_vars[_NUMERIC_ENV_VAR_FIELDS[key]] = str(value)
            continue

        arg_name = _FIELD_RENAMES.get(key, key)
        arg_name = arg_name.replace("_", "-")

        if key in _BOOLEAN_FLAGS:
            if value:
                cli_args.append(f"--{arg_name}")
            elif key in _DEFAULT_OFF_BOOLEAN_FLAGS:
                cli_args.append(f"--no-{arg_name}")
            continue

        if isinstance(value, bool):
            cli_args.extend([f"--{arg_name}", str(value).lower()])
        elif isinstance(value, dict):
            cli_args.extend([f"--{arg_name}", json.dumps(value)])
        else:
            cli_args.extend([f"--{arg_name}", str(value)])

    for key in _DEFAULT_OFF_BOOLEAN_FLAGS:
        arg_name = key.replace("_", "-")
        pos_flag = f"--{arg_name}"
        neg_flag = f"--no-{arg_name}"
        if pos_flag not in cli_args and neg_flag not in cli_args:
            cli_args.append(neg_flag)

    return cli_args, env_vars
