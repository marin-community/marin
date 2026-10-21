# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Run Marin's evaluation CLI behind an export barrier: evaluation starts only after the HF export job succeeds.

:mod:`experiments.post_training.pivotrl.mixed_rlvr_evals` writes one copy of this file per checkpoint with
``@DEPENDENCY@`` replaced by that checkpoint's export job name, or by ``@VERIFIED@`` when the export was checked by hand
(for example because it ran on another cluster, whose jobs this controller cannot see). The copy runs inside the
evaluation workspace, so it imports only what that workspace provides.
"""

EXPORT_JOB = "@DEPENDENCY@"
EXPORT_TIMEOUT = 24 * 3600
# A federated export can take a while to appear on the peer controller after submission.
EXPORT_VISIBILITY_TIMEOUT = 1800
POLL_INTERVAL = 45


def after_export(batch):
    # The copy runs inside the evaluation workspace's environment; these imports resolve there, not in Marin's.
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415

    from connectrpc.code import Code  # noqa: PLC0415
    from connectrpc.errors import ConnectError  # noqa: PLC0415
    from iris.client.client import iris_ctx  # noqa: PLC0415
    from iris.cluster.types import JobName  # noqa: PLC0415
    from iris.resources.state import JobState, is_job_finished  # noqa: PLC0415
    from marin.evaluation.runner import run_evaluation_batch  # noqa: PLC0415
    from rigging.filesystem.factory import url_to_fs  # noqa: PLC0415
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415

    dependency = None if EXPORT_JOB == "@VERIFIED@" else JobName.from_string(EXPORT_JOB)
    started = time.monotonic()
    while dependency is not None:
        try:
            state = iris_ctx().client.job_state(dependency)
        except ConnectError as error:
            if error.code != Code.NOT_FOUND or time.monotonic() - started > EXPORT_VISIBILITY_TIMEOUT:
                raise
            print("EXPORT_BARRIER " + json.dumps({"dependency": str(dependency), "state": "not_found_yet"}), flush=True)
            time.sleep(POLL_INTERVAL)
            continue
        print(
            "EXPORT_BARRIER "
            + json.dumps({"dependency": str(dependency), "state": str(state), "group_id": batch.group_id}),
            flush=True,
        )
        if state == JobState.SUCCEEDED:
            break
        if is_job_finished(state):
            raise RuntimeError(f"export {dependency} ended {state}; refusing to evaluate incomplete weights")
        if time.monotonic() - started > EXPORT_TIMEOUT:
            raise TimeoutError(f"export {dependency} did not finish within {EXPORT_TIMEOUT} seconds")
        time.sleep(POLL_INTERVAL)
    configure_coreweave_s3()
    fs, prefix = url_to_fs(batch.model.location)
    with fs.open(prefix + "/model.safetensors.index.json", "rb") as stream:
        index = json.load(stream)
    shards = sorted(set(index["weight_map"].values()))
    if not shards:
        raise RuntimeError(f"{batch.model.location} lists no weight shards")
    for filename in ["config.json", "tokenizer_config.json", *shards]:
        if fs.info(prefix + "/" + filename)["size"] <= 0:
            raise RuntimeError(f"{batch.model.location}/{filename} is empty")
    print("EXPORT_VERIFIED " + json.dumps({"location": batch.model.location, "shards": len(shards)}), flush=True)
    return run_evaluation_batch(batch)


def main():
    import marin.evaluation.runner as runner  # noqa: PLC0415

    from experiments.evaluation.cli import main as cli_main  # noqa: PLC0415

    runner.run_evaluation_batch = after_export
    cli_main()


if __name__ == "__main__":
    main()
