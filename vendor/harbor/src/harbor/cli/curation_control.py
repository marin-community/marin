"""Execute one source-bound physical nop control from a JSON request on stdin."""

import asyncio
import json
import os
import sys
from pathlib import Path

from taskcompendium.harbor.negative_control import NativeInstructionSetup, NegativeControlSource

from harbor.host_capabilities import host_capabilities_from_environment
from harbor.models.trial.config import TrialConfig
from harbor.trial.negative_control import run_negative_control


def main() -> None:
    request = json.load(sys.stdin)
    setup = request["native_setup"]
    record = asyncio.run(
        run_negative_control(
            TrialConfig.model_validate(request["trial"]),
            NegativeControlSource.model_validate(request["source"]),
            Path(request["output_root"]),
            NativeInstructionSetup.model_validate(setup) if setup is not None else None,
            host_capabilities_from_environment(os.environ, model_authentication={}),
        )
    )
    json.dump(record, sys.stdout)


if __name__ == "__main__":
    main()
