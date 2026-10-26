# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a rights-free synthetic open-ended authoring lifecycle through one Zephyr map."""

import argparse
from datetime import UTC, datetime

from taskcompendium.models import Source
from taskcompendium.runtime.resources import inline_resource
from taskforge.spec.draft import requirements

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedTaskBrief, OpenEndedTaskBundle
from experiments.post_training.capability_driven_envs.open_ended_packet_schema import OpenEndedReleaseStatus
from experiments.post_training.capability_driven_envs.open_ended_zephyr import (
    OpenAIModelPort,
    OpenEndedZephyrConfig,
    run_open_ended_zephyr,
)
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL


def synthetic_bundle() -> OpenEndedTaskBundle:
    """Return a tiny source-free curator input with no pre-authored TaskSpec fields."""
    return OpenEndedTaskBundle(
        bundle_id="synthetic-two-group-rate-smoke",
        split_group="synthetic-open-ended-smoke",
        brief=OpenEndedTaskBrief(
            goal="Compare success rates in two supplied groups.",
            expected_deliverable=(
                "Only the observed rates, a transparent arithmetic comparison, and a small-sample limitation."
            ),
            question_family="synthetic_rate_comparison",
            avoid=(
                "Infer causality from descriptive counts.",
                "Require hypothesis tests, confidence intervals, or inferential values absent from private evidence.",
            ),
        ),
        solver_resources=(inline_resource("workspace/data/counts.csv", b"group,success,total\nA,8,10\nB,2,10\n"),),
        author_evidence=(
            inline_resource(
                "private/evidence.txt",
                (
                    b"The source counts imply A=8/10=80% and B=2/10=20%. The descriptive difference is 60 "
                    b"percentage points and the rate ratio is 4. The samples are small and support no causal claim. "
                    b"Require transparent arithmetic, a descriptive conclusion, and a sample-size limitation. Do not "
                    b"require or report p-values, confidence intervals, hypothesis tests, or other inferential values."
                ),
            ),
        ),
        source=Source(
            dataset="synthetic-open-ended-smoke",
            revision="1",
            row="two-group-rate",
            importer_revision="1",
        ),
        environment_requirements=requirements(image=None),
        release_status=OpenEndedReleaseStatus.UNREVIEWED_DO_NOT_PUBLISH,
        tags=("synthetic", "open-ended-smoke"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API root ending in /v1")
    parser.add_argument("--relay-job", default=DEFAULT_GLM_RELAY_JOB)
    parser.add_argument("--relay-service", default=None)
    parser.add_argument("--model", default=GLM_MODEL)
    parser.add_argument("--api-key-env", default=GLM_BULK_TOKEN_ENV)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--preemptible", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    port = (
        OpenAIModelPort(args.model, args.api_key_env, base_url=args.base_url)
        if args.base_url
        else OpenAIModelPort(
            args.model,
            args.api_key_env,
            relay_job=args.relay_job,
            relay_service=args.relay_service or args.model,
        )
    )
    output = args.output or (
        "gs://marin-us-central1/marin/taskforge/open-ended-smoke/"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}/result.jsonl"
    )
    result = run_open_ended_zephyr(
        [synthetic_bundle()],
        config=OpenEndedZephyrConfig(port),
        output=output,
        max_workers=1,
        logical_shards=1,
        region=args.region or None,
        preemptible=args.preemptible,
    )
    print(f"output={output}")
    print(result)


if __name__ == "__main__":
    main()
