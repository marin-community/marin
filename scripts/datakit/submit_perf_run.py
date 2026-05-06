# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a treatment ferry for the zephyr-perf gate.

Maps Gate N (1/2/3) to the iris CLI shape used by
``.github/workflows/marin-canary-datakit-tier<N>.yaml`` so the treatment job
is structurally comparable to the latest scheduled tier-N baseline run. The
treatment job id is printed to stdout as a single JSON line; pair with
``scripts/datakit/collect_perf_metrics.py --job-id <id>`` to produce the
treatment perf report.

Caller is responsible for setting up the working directory at the right SHA
(typically ``git worktree add ../zephyr-perf-treatment <sha>``); this script
does **not** check out code, it only fires ``iris job run`` from ``--cwd``.

Output (one JSON line on stdout)::

    {"job_id": "...", "gate": "1",
     "ferry_module": "experiments.ferries.datakit_ferry",
     "status_path": "gs://...", "run_id": "...", "cwd": "..."}

The status JSON path is the standard ``FERRY_STATUS_PATH`` contract (see
``experiments/ferries/datakit_ferry.py``); the ferry writes ``status`` and
``marin_prefix`` there on completion.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import os
import re
import shlex
import subprocess
import sys

logger = logging.getLogger(__name__)

# `iris job run --no-wait` prints the new job id to stdout. The id is a
# slash-delimited path of segments, each `[a-z0-9-]+` (e.g.
# `marin-prod/zephyr-perf-pr5348-...`); reject anything else so we fail
# loudly if iris ever starts emitting warnings or ANSI on stdout.
_JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9./_-]*$")


@dataclasses.dataclass(frozen=True)
class TierConfig:
    """Iris CLI shape for one tier.

    Mirrors the ``iris job run --no-wait ... -- python -m <ferry>``
    invocation in ``.github/workflows/marin-canary-datakit-tier<N>.yaml``.
    Drift here means the treatment job won't be comparable to the scheduled
    baseline — keep these in lockstep with the workflow YAMLs.
    """

    ferry_module: str
    extra: str
    region: str | None = None
    memory: str | None = None
    disk: str | None = None
    cpu: str | None = None
    priority: str | None = None
    preemptible: bool = True
    enable_extra_resources: bool = False
    extra_env: tuple[tuple[str, str], ...] = ()
    ferry_args: tuple[str, ...] = ()

    def iris_flags(self) -> list[str]:
        flags = [f"--extra={self.extra}"]
        if self.region:
            flags.append(f"--region={self.region}")
        if self.memory:
            flags.append(f"--memory={self.memory}")
        if self.disk:
            flags.append(f"--disk={self.disk}")
        if self.cpu:
            flags.append(f"--cpu={self.cpu}")
        if self.priority:
            flags.append(f"--priority={self.priority}")
        if self.enable_extra_resources:
            flags.append("--enable-extra-resources")
        if not self.preemptible:
            flags.append("--no-preemptible")
        return flags


TIERS: dict[str, TierConfig] = {
    # marin-canary-datakit-tier1.yaml — FineWeb-Edu sample/10BT smoke.
    "1": TierConfig(
        ferry_module="experiments.ferries.datakit_ferry",
        extra="cpu",
        memory="2G",
        disk="4G",
        cpu="1",
        priority="production",
    ),
    # marin-canary-datakit-tier2.yaml — synthetic skewed long-tail stress.
    # Uses iris defaults for memory/disk/cpu/priority (no overrides in the
    # workflow), so we leave them unset here too.
    "2": TierConfig(
        ferry_module="experiments.ferries.datakit_tier2_skewed_ferry",
        extra="cpu",
    ),
    # marin-canary-datakit-tier3.yaml — Nemotron quality=high, max_files=1000.
    # `FERRY_TEST_MAX_FILES=1000` caps the staged shard count to ~10% of the
    # full medium slice; required for parity with the scheduled baseline.
    # `--memory=4G`, `--enable-extra-resources`, and `--no-preemptible` are
    # workarounds for a tokenize cleanup OOM (see workflow YAML); they will
    # go away once the underlying issue is fixed — keep this file in sync
    # when that happens.
    "3": TierConfig(
        ferry_module="experiments.ferries.datakit_nemotron_ferry",
        extra="cpu",
        region="europe-west4",
        memory="4G",
        disk="5G",
        cpu="1",
        priority="production",
        preemptible=False,
        enable_extra_resources=True,
        extra_env=(("FERRY_TEST_MAX_FILES", "1000"),),
    ),
}


def _build_run_id(pr: int, gate: str) -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"zephyr-perf-pr{pr}-g{gate}-treatment-{ts}"


def _build_status_path(pr: int, run_id: str, override: str | None) -> str:
    if override:
        return override
    return f"gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/pr{pr}/{run_id}/ferry_run_status.json"


def submit(
    *,
    gate: str,
    pr: int,
    cwd: str,
    iris_config: str,
    status_path_override: str | None,
    dry_run: bool,
) -> dict[str, object]:
    if gate not in TIERS:
        raise ValueError(f"Unknown gate {gate!r}; expected one of {sorted(TIERS)}")

    cfg = TIERS[gate]
    run_id = _build_run_id(pr, gate)
    status_path = _build_status_path(pr, run_id, status_path_override)

    cmd = [
        ".venv/bin/iris",
        f"--config={iris_config}",
        "job",
        "run",
        "--no-wait",
        *cfg.iris_flags(),
        "-e",
        "SMOKE_RUN_ID",
        run_id,
        "-e",
        "FERRY_STATUS_PATH",
        status_path,
    ]
    for k, v in cfg.extra_env:
        cmd += ["-e", k, v]
    # Forward HF_TOKEN from the calling environment, expanded — `subprocess`
    # does not run a shell, so the literal "$HF_TOKEN" string would otherwise
    # be passed verbatim to iris CLI. Skip the flag entirely when the secret
    # isn't set rather than overriding any inherited value with an empty
    # string. (We don't forward WANDB_* — the datakit pipeline path
    # (download → normalize → minhash → fuzzy_dups → consolidate → tokenize)
    # doesn't read those, even though the tier workflow YAMLs pass them.)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        cmd += ["-e", "HF_TOKEN", hf_token]
    elif not dry_run:
        logger.warning(
            "HF_TOKEN not set in environment; iris job will inherit whatever the "
            "controller has (often unset). HF downloads in tier1/tier2 may fail."
        )
    cmd += ["--", "python", "-m", cfg.ferry_module, *cfg.ferry_args]

    if dry_run:
        logger.info("dry-run cmd (cwd=%s): %s", cwd, " ".join(shlex.quote(c) for c in cmd))
        job_id = "DRY-RUN"
    else:
        out = subprocess.check_output(cmd, cwd=cwd, text=True)
        candidate = out.strip().splitlines()[-1].strip() if out.strip() else ""
        if not _JOB_ID_RE.match(candidate):
            raise RuntimeError(
                f"iris job run did not return a recognisable job id; last line was {candidate!r}. "
                f"Full stdout:\n{out}"
            )
        job_id = candidate

    return {
        "job_id": job_id,
        "gate": gate,
        "ferry_module": cfg.ferry_module,
        "status_path": status_path,
        "run_id": run_id,
        "cwd": cwd,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        required=True,
        choices=sorted(TIERS),
        help="1=tier1 smoke, 2=tier2 long-tail stress, 3=tier3 nemotron.",
    )
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--cwd", required=True, help="Path to a worktree at the PR head SHA.")
    parser.add_argument(
        "--iris-config",
        default="lib/iris/examples/marin.yaml",
        help="Iris cluster config; resolved relative to --cwd.",
    )
    parser.add_argument(
        "--status-out",
        default=None,
        help="Override FERRY_STATUS_PATH. Default lives under gs://marin-us-central1/tmp/ttl=7d/.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = submit(
        gate=args.gate,
        pr=args.pr,
        cwd=args.cwd,
        iris_config=args.iris_config,
        status_path_override=args.status_out,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
