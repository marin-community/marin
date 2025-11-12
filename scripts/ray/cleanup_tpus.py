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

import argparse
import logging
from marin.cluster.cleanup import run_cleanup_loop


def main():
    """CLI entry point for cleanup utilities."""
    parser = argparse.ArgumentParser(description="Automated cluster cleanup utilities")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--zone", required=True, help="GCP zone")
    parser.add_argument("--interval", type=int, default=600, help="Cleanup interval in seconds (default: 600)")
    parser.add_argument("--disk_threshold_pct", type=float, default=1.0, help="Disk threshold percentage (default: 1.0)")

    args = parser.parse_args()

    # Run the cleanup loop
    run_cleanup_loop(
        gcp_project=args.project,
        zone=args.zone,
        interval=args.interval,
        disk_threshold_pct=args.disk_threshold_pct,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
