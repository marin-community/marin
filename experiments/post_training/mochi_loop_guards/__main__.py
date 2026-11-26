# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loop-guards stage launcher and analysis subcommands.

The default invocation is the native RL stage CLI, which resolves its declared
credentials file-first (``rigging.secrets``) before ``--run`` submits anything:

    python -m experiments.post_training.mochi_loop_guards \\
        --preset s0-baseline --version 2026.11.25.9 --issue 12264 --run --wait

Analysis subcommands cover the postprocessing chain, the TB2 infrastructure recovery,
conservation and user-override receipts, and the one generic in-region CPU submit path:

    python -m experiments.post_training.mochi_loop_guards postprocess --run-output … --output …
    python -m experiments.post_training.mochi_loop_guards cpu-submit --job-name … \\
        --cpu 16 --memory 64GB --disk 128GB -- python -m experiments.post_training.mochi_loop_guards audit …

The vendored-SkyRL analyses (``counterfactual``, ``observation-census``,
``stitch-tb2-recovery``, ``tb2-repetition-ledger``) run under ``cpu-submit --skyrl``.
"""

import importlib
import os
import sys
from collections.abc import Sequence

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import SecretResolutionError, require_secret

from experiments.post_training import rl, rl_compose
from experiments.post_training.mochi_loop_guards.presets import GUARDS

REGIONAL_ROOT = "s3://marin-us-east-08a/marin"
# Declared stage credentials, resolved file-first before GPU allocation. The RL-pool
# stages (S2) also talk to the dedicated RL Daytona org.
RUN_SECRETS = ("WANDB_API_KEY", "DAYTONA_EVAL_API_KEY", "HF_TOKEN")
RL_POOL_SECRETS = ("DAYTONA_RL_API_KEY",)
RL_POOL_STAGE_PREFIXES = ("s2-",)
WANDB_ENTITY = "marin-community"

# Analysis subcommands may pull the vendored SkyRL environment, so their modules import
# lazily and the stage-launcher path stays light.
ANALYSIS_MODULES = {
    "apply-user-overrides": ("user_overrides", "apply_receipts"),
    "audit": ("audit", "main"),
    "conservation-check": ("conservation", "main"),
    "counterfactual": ("counterfactual", "main"),
    "cpu-submit": ("cpu_submit", "main"),
    "observation-census": ("observation_census", "main"),
    "postprocess": ("postprocess", "postprocess"),
    "publish-user-overrides": ("user_overrides", "publish_receipts"),
    "report-refresh": ("postprocess", "report_refresh"),
    "retain-traces": ("postprocess", "retain_traces"),
    "stitch-tb2-recovery": ("tb2_recovery", "stitch"),
    "tb2-repetition-ledger": ("tb2_recovery", "repetition_ledger"),
    "tb2-recovery-selection": ("tb2_recovery", "recovery_selection"),
}


def selected_preset(arguments: Sequence[str]) -> str | None:
    """The ``--preset`` value of a stage CLI invocation, if any."""
    if "--preset" in arguments:
        return arguments[list(arguments).index("--preset") + 1]
    return None


def resolve_run_secrets(preset_name: str | None) -> tuple[str, ...]:
    """Resolve the stage's declared credentials file-first, before GPU allocation.

    Every name must resolve from a named secret file (``$RIGGING_SECRETS_DIR``, mounted
    in region) or the environment; a missing one fails the launch before any GPU work.
    """
    names = [*RUN_SECRETS, *(RL_POOL_SECRETS if preset_name and preset_name.startswith(RL_POOL_STAGE_PREFIXES) else ())]
    for name in names:
        try:
            os.environ[name] = require_secret(name)
        except SecretResolutionError as error:
            raise SecretResolutionError(
                f"{error}; put {name} in $RIGGING_SECRETS_DIR or export it before --run"
            ) from None
    os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
    return tuple(names)


def run_stage(arguments: Sequence[str]) -> int:
    """The native RL stage CLI against the regional selection root."""
    if "--run" in arguments:
        resolve_run_secrets(selected_preset(arguments))
    configure_coreweave_s3()
    # These experiment selections are published in the execution region, rather than
    # task curation's canonical 02a root. Every stage uses the same regional binding.
    rl_compose.SELECTION_ROOT = f"{REGIONAL_ROOT}/rl-data/task-selection"
    os.environ.setdefault("MARIN_PREFIX", REGIONAL_ROOT)
    os.environ.setdefault("MARIN_TEMP_PREFIX", "s3://marin-us-east-08a/tmp/ttl=3d")
    return rl.main(GUARDS, argv=list(arguments), forward_env=("MARIN_PREFIX", "MARIN_TEMP_PREFIX"))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    entry = ANALYSIS_MODULES.get(arguments[0]) if arguments else None
    if entry is None:
        return run_stage(arguments)
    module_name, function_name = entry
    module = importlib.import_module(f"{__package__}.{module_name}")
    getattr(module, function_name)(arguments[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
