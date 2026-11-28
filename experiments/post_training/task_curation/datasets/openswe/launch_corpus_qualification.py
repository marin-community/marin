# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the corpus oracle-qualification shards in-region.

One Iris job per index-range shard of the repaired population; each job runs
the qualification driver's fail-closed no-op + oracle controls at its own
sandbox concurrency. The shards share one Daytona EVAL account, so the sum of
the shards' ``--concurrency`` values is the org's concurrent-sandbox load and
must stay within the coordinator's ceiling. Every shard registers with the
coordinator at submission.
"""

import argparse
import os
import subprocess

from rigging.secrets import resolve_secret_spec

from experiments.post_training.task_curation.datasets.openswe.launch_fidelity_gate import (
    CONDENSED_BINDINGS_URI,
    EVAL_KEY_SPEC,
)

RAW_SCRIPT = (
    "https://raw.githubusercontent.com/marin-community/marin/{revision}"
    "/experiments/post_training/task_curation/run_corpus_qualification.sh"
)


def eval_key() -> str:
    """The Daytona eval account key, from the environment or GSM."""
    value = os.environ.get("DAYTONA_EVAL_API_KEY")
    if not value:
        value = resolve_secret_spec(EVAL_KEY_SPEC).value
    if not value:
        raise SystemExit("the Daytona eval key resolved empty; set DAYTONA_EVAL_API_KEY before launching")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The landed marin revision to run at.")
    parser.add_argument("--job-prefix", required=True, help="The shard job names' prefix.")
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--gate-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--condensed-bindings", default=CONDENSED_BINDINGS_URI)
    parser.add_argument("--concurrency-per-shard", default="8")
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--timeout", default="86400", help="Per-shard job timeout in seconds (default: 24h).")
    args = parser.parse_args()

    script_url = RAW_SCRIPT.format(revision=args.revision)
    key = eval_key()
    for shard_index in range(args.shard_count):
        job_name = f"{args.job_prefix}-shard-{shard_index:02d}"
        command = (
            f"mkdir -p /workspace && curl -fsSL {script_url} -o /workspace/run_corpus_qualification.sh && "
            f"bash /workspace/run_corpus_qualification.sh {args.revision} {args.condensed_bindings} "
            f"{args.gate_root} {shard_index} {args.shard_count} {args.report_root} {args.concurrency_per_shard}"
        )
        argv = [
            "uv", "run", "--no-sync", "iris", f"--cluster={args.cluster}", "job", "run",
            "--secret", "HF_TOKEN", "env:HF_TOKEN",
            "--env-vars", "DAYTONA_API_KEY", key,
            "--env-vars", "DAYTONA_API_URL", "https://app.daytona.io/api",
            "--env-vars", "DAYTONA_TARGET", "us",
            "--no-wait",
            "--job-name", job_name,
            "--replicas", "1",
            "--cpu", "8",
            "--memory", "32GB",
            "--disk", "256GB",
            "--enable-extra-resources",
            "--priority", "interactive",
            "--no-preemptible",
            "--timeout", args.timeout,
            "--max-retries", "0",
            "--", "bash", "-c", command,
        ]
        completed = subprocess.run(argv)
        if completed.returncode:
            raise SystemExit(f"shard {shard_index} submission failed")
        print(f"qualification shard submitted: /benfeuer/{job_name}")


if __name__ == "__main__":
    main()
