# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

from cloud.iris import launch, launch_config
from cloud.iris.config_cli import verify_checkout

verify_checkout(Path(sys.argv[1]), launch, launch_config)
args = launch.create_parser().parse_args(sys.argv[2:])
config = launch.load_launch_config(args.config)
print(
    json.dumps(
        {
            "forward_env": args.forward_env,
            "disk": config.iris.allocation.disk,
            "tokenizer": config.inputs.model.tokenizer_uri,
            "model": config.run.model,
            "attempt": config.run.attempt_id,
            "export": config.artifacts.export_root,
        }
    )
)
