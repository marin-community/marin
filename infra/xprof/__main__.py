# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pulumi
from config import ENDPOINT_NAME, HEALTH_PATH, PORT_NAME, XPROF_PACKAGE
from iac.iris.service import IrisService, IrisServiceArgs
from iris.cluster.types import ResourceSpec


def main() -> None:
    config = pulumi.Config()
    service = IrisService(
        "xprof",
        IrisServiceArgs(
            cluster=config.require("cluster"),
            name="xprof",
            user=config.require("user"),
            entrypoint=("python", "-m", "infra.xprof.server"),
            resources=ResourceSpec(
                cpu=float(config.require("cpu")),
                memory=config.require("memory"),
                disk=config.require("disk"),
            ),
            regions=(config.require("region"),),
            port=PORT_NAME,
            endpoint=ENDPOINT_NAME,
            health_path=HEALTH_PATH,
            env=dict(config.get_object("env") or {}),
            secret_env=dict(config.get_object("secret_env") or {}),
            pip_packages=(XPROF_PACKAGE,),
            sync_packages=("marin-iris", "marin-rigging"),
            deploy_generation=config.get_int("deploy_generation") or 0,
            code_paths=(
                "infra/xprof",
                "infra/pulumi",
                "lib/rigging/src/rigging/filesystem",
            ),
        ),
    )
    pulumi.export("job_id", service.job_id)
    pulumi.export("url", service.url)


main()
