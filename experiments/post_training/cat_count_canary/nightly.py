# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the CatCount launcher with receiving-controller scheduler receipts."""

import json
import sys
import threading

from iris.cli.connect import rpc_client
from iris.cluster.client.job_info import get_job_info
from iris.rpc import controller_pb2, job_pb2

from .launcher import main

RECEIPT_INTERVAL = 60


def print_scheduler_receipt(controller_address: str, job_id: str) -> None:
    with rpc_client(controller_address) as client:
        state = client.get_scheduler_state(controller_pb2.Controller.GetSchedulerStateRequest())
    receipt = {
        "job_id": job_id,
        "budget": [
            {
                "user": row.user_id,
                "limit": row.budget_limit,
                "spent": row.budget_spent,
                "effective_band": job_pb2.PriorityBand.Name(row.effective_band),
            }
            for row in state.user_budgets
        ],
        "actual_tasks": [
            {"job_id": row.job_id, "band": job_pb2.PriorityBand.Name(row.band), "count": row.count}
            for row in (*state.pending_buckets, *state.running_buckets)
            if row.job_id == job_id or row.job_id.startswith(job_id + "/")
        ],
    }
    print("CAT_COUNT_SCHEDULER " + json.dumps(receipt), flush=True)


def run() -> None:
    info = get_job_info()
    if info is None or not info.controller_address:
        raise RuntimeError("the nightly coordinator requires an Iris task context")
    job_id = info.job_id.to_wire()
    print_scheduler_receipt(info.controller_address, job_id)
    stopped = threading.Event()

    def observe() -> None:
        while not stopped.wait(RECEIPT_INTERVAL):
            print_scheduler_receipt(info.controller_address, job_id)

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    try:
        main(args=sys.argv[1:])
    finally:
        stopped.set()
        observer.join()


if __name__ == "__main__":
    run()
