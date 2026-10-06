# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pulumi
from config import CHEROOT_PACKAGE, ENDPOINT_NAME, HEALTH_PATH, PORT_NAME
from iac.iris.service import IrisService, IrisServiceArgs
from iris.cluster.types import ResourceSpec
from release import XPROF_RS_BINARY_PATH


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
            pip_packages=(CHEROOT_PACKAGE,),
            sync_packages=("marin-iris", "marin-rigging"),
            build_commands=(".venv/bin/python -m infra.xprof.download",),
            extra_bundle_includes=(XPROF_RS_BINARY_PATH,),
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
