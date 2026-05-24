# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify every Datakit source's raw download and normalized outputs terminated SUCCESS.

For each :class:`marin.datakit.sources.DatakitSource`, two independent
``.executor_status`` checks run against the configured prefix:

* **raw** — the DAG-leaf step(s) of ``normalize_steps`` (steps with no
  ``deps``). These are the upstream download dumps under ``raw/...``
  that the ferry expects to already exist; intermediate ``processed/``
  preprocessing steps are skipped.
* **normalized** — ``source.normalized.output_path`` (the terminal
  normalize step's output — what downstream sample/tokenize consumes).

The staging prefix is pinned via ``MARIN_PREFIX`` so all ``output_path``s
resolve under it regardless of the caller's environment. Defaults to
``gs://marin-us-central1`` (enforced daily as a parallel lane of the
datakit-smoke workflow); pass ``--prefix`` to validate other backends
such as R2 (``s3://marin-na/marin``).
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from marin.datakit.sources import all_sources
from marin.execution.executor_step_status import STATUS_SUCCESS, StatusFile
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_STAGING_PREFIX = "gs://marin-us-central1"
MAX_WORKERS = 16
WORKER_ID = "datakit-smoke-sources-check"


def _configure_r2_env() -> None:
    """Map ``R2_*`` env vars onto the ``AWS_*`` names s3fs/botocore expect."""
    key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("R2_ENDPOINT_URL")
    if not key_id or not secret or not endpoint:
        raise SystemExit(
            "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_ENDPOINT_URL must be set to validate against R2"
        )
    os.environ["AWS_ACCESS_KEY_ID"] = key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret
    os.environ["AWS_ENDPOINT_URL"] = endpoint
    os.environ["AWS_ENDPOINT_URL_S3"] = endpoint


def _raw_leaves(steps: tuple[StepSpec, ...]) -> list[StepSpec]:
    """Return the DAG leaves reachable from ``steps`` (steps with no deps).

    The first entry in ``normalize_steps`` is sometimes a ``processed/`` step
    that depends on a separate ``raw/`` download; we want the upstream raw
    nodes, so walk ``deps`` until we hit a step with none. Identity-dedupes
    shared leaves (e.g. a Nemotron family's single download feeding many
    subsets).
    """
    seen: set[int] = set()
    leaves: list[StepSpec] = []
    stack: list[StepSpec] = list(steps)
    while stack:
        step = stack.pop()
        if id(step) in seen:
            continue
        seen.add(id(step))
        if step.deps:
            stack.extend(step.deps)
        else:
            leaves.append(step)
    return leaves


def _check(output_path: str) -> tuple[str, str]:
    """Return (output_path, status) where status is ``SUCCESS`` or a failure token."""
    status = StatusFile(output_path, worker_id=WORKER_ID).status
    return output_path, status or "MISSING"


def _validate(label: str, paths: list[str], prefix: str) -> list[tuple[str, str]]:
    """Probe every path in parallel; log + return anything not SUCCESS."""
    logger.info("Verifying %d unique %s paths under %s", len(paths), label, prefix)
    bad: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for output_path, status in pool.map(_check, paths):
            if status == STATUS_SUCCESS:
                logger.debug("OK: %s", output_path)
            else:
                logger.error("%s %s: %s", label, status, output_path)
                bad.append((output_path, status))
    if not bad:
        logger.info("All %d %s paths report SUCCESS", len(paths), label)
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default=DEFAULT_STAGING_PREFIX,
        help=(
            "Storage prefix to resolve datakit output_paths against (sets MARIN_PREFIX). "
            f"Defaults to {DEFAULT_STAGING_PREFIX}. Use e.g. s3://marin-na/marin to validate R2."
        ),
    )
    args = parser.parse_args()

    configure_logging()
    if args.prefix.startswith("s3://"):
        _configure_r2_env()
    # Pin the staging prefix before building the registry — StepSpec caches
    # output_path on first access, so MARIN_PREFIX must be set before
    # all_sources() materializes any chain.
    os.environ["MARIN_PREFIX"] = args.prefix

    sources = all_sources()
    raw_paths = sorted({leaf.output_path for s in sources.values() for leaf in _raw_leaves(s.normalize_steps)})
    normalized_paths = sorted({s.normalized.output_path for s in sources.values()})

    bad_raw = _validate("raw", raw_paths, args.prefix)
    bad_norm = _validate("normalized", normalized_paths, args.prefix)

    if bad_raw or bad_norm:
        raise SystemExit(
            f"{len(bad_raw)}/{len(raw_paths)} raw and "
            f"{len(bad_norm)}/{len(normalized_paths)} normalized paths not SUCCESS under {args.prefix}"
        )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
