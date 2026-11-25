# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

from cloud.iris import launch
from cloud.iris.config_cli import verify_checkout
from finestore.rl.run_records import LaunchResult, read_attempt


def main() -> None:
    verify_checkout(Path(sys.argv[1]), launch)
    args = launch.create_parser().parse_args(sys.argv[3:])
    config = launch.load_launch_config(args.config)
    launch._record_prepared_attempt(config)
    prepared = read_attempt(str(config.artifacts.export_root), str(config.run.attempt_id))
    assert prepared is not None and prepared.result is None and prepared.index is not None
    assert prepared.native.config["run"]["coordinator_iris_job_id"] == config.run.coordinator_iris_job_id
    response = LaunchResult.model_validate_json(Path(sys.argv[2]).read_text()).model_copy(
        update={"run_id": str(config.run.id), "attempt_id": str(config.run.attempt_id)}
    )
    receipt = launch._write_attempt_receipt(config, response)
    print(json.dumps({"forward_env": args.forward_env, "receipt": receipt.model_dump(mode="json")}))


if __name__ == "__main__":
    main()
