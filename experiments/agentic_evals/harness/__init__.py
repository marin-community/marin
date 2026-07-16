"""Harbor harness wiring: config load, command build, trial prune, compat shims.

Re-exports the key functions used by the runner and launchers.
"""
from .config import (
    load_harbor_config,
    extract_agent_kwargs_from_config,
    apply_nested_key,
    parse_agent_kwarg_strings,
    merge_agent_kwargs,
    serialize_agent_kwargs,
)
from .command import (
    build_endpoint_meta,
    load_endpoint_metadata,
    merge_harbor_config,
    build_harbor_command,
    run_harbor_cli,
)
from .job_config import load_job_config
from .trial_prune import prune_refire_errored_trials

__all__ = [
    "load_harbor_config",
    "extract_agent_kwargs_from_config",
    "apply_nested_key",
    "parse_agent_kwarg_strings",
    "merge_agent_kwargs",
    "serialize_agent_kwargs",
    "build_endpoint_meta",
    "load_endpoint_metadata",
    "merge_harbor_config",
    "build_harbor_command",
    "run_harbor_cli",
    "load_job_config",
    "prune_refire_errored_trials",
]
