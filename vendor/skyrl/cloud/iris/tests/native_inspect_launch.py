# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect a launch and optionally complete its prepared attempt."""

import json
import sys
from pathlib import Path

from finestore.rl.run_records import LaunchResult

from cloud.iris import launch, launch_config
from cloud.iris.config_cli import verify_checkout


def main() -> None:
    verify_checkout(Path(sys.argv[1]), launch, launch_config)
    parser = launch.create_parser()
    parser.add_argument("--record-response", type=Path)
    args = parser.parse_args(sys.argv[2:])
    config = launch.load_launch_config(args.config)
    info = {
        "forward_env": args.forward_env,
        "disk": config.iris.allocation.disk,
        "tokenizer": config.inputs.model.tokenizer_uri,
        "model": config.run.model,
        "attempt": config.run.attempt_id,
        "export": config.artifacts.export_root,
    }
    if args.record_response is not None:
        response = LaunchResult.model_validate_json(args.record_response.read_text()).model_copy(
            update={"run_id": str(config.run.id), "attempt_id": str(config.run.attempt_id)}
        )
        info["receipt"] = launch._write_attempt_receipt(config, response).model_dump(mode="json")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
