# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select Pulumi service rollouts from changed repository paths."""

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Rollout:
    name: str
    stack: str
    work_dir: str
    service_account: str
    source_roots: tuple[str, ...] = ()
    timeout_minutes: int = 60
    cloudflare_secret_name: str = ""
    deploy_generation_key: str = ""
    test_path: str = ""


ROLLOUTS = (
    Rollout(
        name="ducky",
        stack="ducky-marin",
        work_dir="infra/ducky",
        service_account="iris-ci-smoke@hai-gcp-models.iam.gserviceaccount.com",
        source_roots=("lib/ducky", "infra/pulumi/src/iac/iris"),
        timeout_minutes=30,
        deploy_generation_key="ducky:deploy_generation",
    ),
    Rollout(
        name="echo",
        stack="marin-echo",
        work_dir="infra/echo",
        service_account="marin-cd-cloud-run-deploy@hai-gcp-models.iam.gserviceaccount.com",
        source_roots=(
            "infra/pulumi/src/iac/gcp/cloud_run.py",
            "infra/pulumi/src/iac/gcp/cloud_run_job.py",
            "lib/rigging/src/rigging/auth.py",
        ),
        timeout_minutes=90,
        cloudflare_secret_name="cloudflare-oa-dns-token",
    ),
    Rollout(
        name="grafana",
        stack="marin-grafana",
        work_dir="infra/grafana",
        service_account="marin-cd-cloud-run-deploy@hai-gcp-models.iam.gserviceaccount.com",
        source_roots=("infra/pulumi/src/iac/gcp/cloud_run.py", "lib/rigging/src/rigging/auth.py"),
        cloudflare_secret_name="cloudflare-oa-dns-token",
        test_path="infra/grafana",
    ),
    Rollout(
        name="xprof",
        stack="xprof-marin",
        work_dir="infra/xprof",
        service_account="iris-ci-smoke@hai-gcp-models.iam.gserviceaccount.com",
        source_roots=("infra/pulumi/src/iac/iris", "lib/rigging/src/rigging/filesystem"),
        timeout_minutes=30,
        deploy_generation_key="xprof:deploy_generation",
    ),
)


def path_is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def rollouts_for_paths(paths: Iterable[str]) -> tuple[Rollout, ...]:
    """Return each rollout affected by at least one changed path."""
    changed_paths = tuple(paths)
    return tuple(
        rollout
        for rollout in ROLLOUTS
        if any(path_is_under(path, root) for path in changed_paths for root in (rollout.work_dir, *rollout.source_roots))
    )


def rollout_for_service(name: str) -> Rollout:
    for rollout in ROLLOUTS:
        if rollout.name == name:
            return rollout
    choices = ", ".join(rollout.name for rollout in ROLLOUTS)
    raise ValueError(f"unknown rollout {name!r}; choose one of: {choices}")


def rollout_item(rollout: Rollout, deploy_generation: str = "") -> dict[str, object]:
    item = asdict(rollout)
    item.pop("source_roots")
    deploy_generation_key = item.pop("deploy_generation_key")
    if deploy_generation and not deploy_generation_key:
        raise ValueError(f"{rollout.name} does not support a deploy-generation override")
    item["config_map"] = (
        json.dumps({deploy_generation_key: {"value": deploy_generation}}, separators=(",", ":"))
        if deploy_generation
        else ""
    )
    return item


def workflow_payload(rollouts: Iterable[Rollout], deploy_generation: str = "") -> dict[str, object]:
    items = [rollout_item(rollout, deploy_generation) for rollout in rollouts]
    return {
        "deploy": {"include": items},
        "test": {"include": [item for item in items if item["test_path"]]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", help="select one rollout instead of reading changed paths from stdin")
    parser.add_argument("--deploy-generation", default="", help="force a supported service to redeploy")
    args = parser.parse_args()

    try:
        if args.service:
            selected = (rollout_for_service(args.service),)
        else:
            if args.deploy_generation:
                raise ValueError("--deploy-generation requires --service")
            selected = rollouts_for_paths(line.strip() for line in sys.stdin if line.strip())
        print(json.dumps(workflow_payload(selected, args.deploy_generation), separators=(",", ":")))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
