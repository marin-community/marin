# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror both Mac alert producers into the shared Finelog event feed."""

import argparse
import json
from pathlib import Path

from finelog.deploy.config import load_finelog_config
from finelog.deploy.connect import open_client
from finelog.ops_feed import DEFAULT_STATE
from finelog.ops_feed_mac import MacSource, import_alerts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchdog-alerts", type=Path, required=True)
    parser.add_argument("--resource-alerts", type=Path, required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    with open_client(load_finelog_config(args.deployment), args.deployment) as client:
        counts = {
            source: import_alerts(client, source, path, destination=args.deployment, state=args.state)
            for source, path in (
                (MacSource.JOB_WATCHDOG, args.watchdog_alerts),
                (MacSource.RESOURCE_WATCH, args.resource_alerts),
            )
        }
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
