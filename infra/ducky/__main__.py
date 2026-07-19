# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for the ducky service.

Deploys ducky as an always-on Iris job through the reusable
``iac.iris.service.IrisService`` component. The job identity, shape, and DUCKY_* task
environment come from the stack config (``Pulumi.<stack>.yaml``); the port and endpoint
come from ``ducky.config`` so the stack cannot drift from what the server registers.

The Vue dashboard must be built before ``pulumi up`` (CI does this; locally run
``npm ci && npm run build`` in lib/ducky/dashboard) — the deploy fails when ``dist/``
is missing rather than shipping without the dashboard.

Runs on the shared repo venv; ``uv sync --all-packages --extra deploy`` first. See
infra/iac/README.md for the state backend and passphrase.
"""

import pulumi
from ducky.config import ENDPOINT_NAME, PORT_NAME
from iac.iris.service import IrisService, IrisServiceArgs
from iris.cluster.types import ResourceSpec, tpu_device

# The built SPA is gitignored; this glob re-includes it into the Iris bundle (and into
# the redeploy code hash, as built bytes).
DASHBOARD_DIST_INCLUDE = "lib/ducky/dashboard/dist/**/*"


def main() -> None:
    config = pulumi.Config()
    tpu = config.get("tpu")
    service = IrisService(
        "ducky",
        IrisServiceArgs(
            cluster=config.require("cluster"),
            name="ducky",
            user=config.require("user"),
            entrypoint=("python", "-m", "ducky.server"),
            resources=ResourceSpec(
                cpu=float(config.require("cpu")),
                memory=config.require("memory"),
                device=tpu_device(tpu) if tpu else None,
            ),
            regions=(config.require("region"),),
            port=PORT_NAME,
            endpoint=ENDPOINT_NAME,
            env=dict(config.get_object("env") or {}),
            secret_env=dict(config.get_object("secret_env") or {}),
            deploy_generation=config.get_int("deploy_generation") or 0,
            code_paths=("lib/ducky",),
            extra_bundle_includes=(DASHBOARD_DIST_INCLUDE,),
        ),
    )
    pulumi.export("job_id", service.job_id)
    pulumi.export("url", service.url)


main()
