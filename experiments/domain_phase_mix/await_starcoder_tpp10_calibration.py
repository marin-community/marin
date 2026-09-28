# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Submit the authorized calibration after the reviewed preparation succeeds."""

import argparse
import csv
import json
import logging
from pathlib import Path

import fsspec
from iris.client.client import iris_ctx
from iris.cluster.types import JobName

from experiments.domain_phase_mix import launch_starcoder_tpp10 as training
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as preparation
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.launch_starcoder_epoch_matching import persist_submission_plan
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    experiment.require_central1()
    authorization = json.loads(args.authorization.read_text())
    design = experiment.load_design()
    plan, _ = training.build_plan(design, "calibration")
    if (
        authorization["approved"] is not True
        or authorization["stage"] != "calibration"
        or authorization["plan_sha256"] != plan["plan_sha256"]
        or authorization["design_sha256"] != design["design_sha256"]
        or not authorization["reviewer"]
    ):
        raise ValueError("Authorization does not release this frozen calibration plan")

    context = iris_ctx()
    if context.client is None:
        raise RuntimeError("Calibration coordinator must run under Iris")
    prerequisite = JobName.from_wire(authorization["preparation_job"])
    logger.info("Waiting for successful preparation: %s", prerequisite)
    context.client.job(prerequisite).wait(timeout=43200, poll_interval=60, raise_on_failure=True)

    audit = preparation.verify_caches(design, preparation.data_steps(design), experiment.PREFIX)
    digest = canonical_sha256(audit)
    audit_uri = f"{experiment.PREFIX}/experiments/starcoder_tpp10/data_audits/{digest}/cache_audit.json"
    with fsspec.open(audit_uri, "rt") as handle:
        if json.load(handle) != audit:
            raise ValueError("The preparation audit differs from the current caches")
    release = {
        "approved": True,
        "reviewer": authorization["reviewer"],
        "stage": "calibration",
        "plan_sha256": plan["plan_sha256"],
        "cache_audit_sha256": digest,
        "preparation_job": authorization["preparation_job"],
        "authorization_sha256": canonical_sha256(authorization),
    }
    release_uri = f"{experiment.PREFIX}/experiments/starcoder_tpp10/{plan['plan_sha256']}/calibration_release.json"
    persist_submission_plan(release, release_uri)
    args.output.mkdir(parents=True, exist_ok=True)
    release_path = args.output / "calibration_release.json"
    release_path.write_text(json.dumps(release, indent=2) + "\n")
    logger.info("Verified caches: %s; recorded release: %s", audit_uri, release_uri)
    training.main(
        [
            "--stage",
            "calibration",
            "--release",
            str(release_path),
            "--plan-path",
            str(args.output / "calibration_plan.json"),
            "--max-concurrent",
            "8",
            "--submit",
        ]
    )
    with (args.output / "calibration_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    measurements = {row["run_name"]: float(row["value"]) for row in rows}
    results = {
        "plan_sha256": plan["plan_sha256"],
        "primary_metric": plan["primary_metric"],
        "rows": rows,
        "summary": experiment.calibration_summary(plan, measurements),
    }
    results_uri = f"{experiment.PREFIX}/experiments/starcoder_tpp10/{plan['plan_sha256']}/calibration_results.json"
    persist_submission_plan(results, results_uri)
    logger.info("Calibration complete: %s; later stages require a separate release", results_uri)


if __name__ == "__main__":
    main()
