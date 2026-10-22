# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave-hosted OpenBao application; deployment never initializes the vault."""

import os

import pulumi
import pulumi_kubernetes as k8s
from iac.config import load_iris_config
from iac.kubernetes.openbao import OpenBaoDeployment, openbao_manifests


def main() -> None:
    config = OpenBaoDeployment.model_validate(pulumi.Config().require_object("deployment"))
    cluster = load_iris_config(config.cluster)
    coreweave = cluster.platform.coreweave
    if coreweave is None or not coreweave.kube_context or not os.environ.get("KUBECONFIG"):
        raise ValueError("OpenBao requires the reviewed CW context and ambient KUBECONFIG")
    provider = k8s.Provider("openbao-k8s", kubeconfig="", context=coreweave.kube_context)
    k8s.yaml.v2.ConfigGroup(
        "openbao",
        objs=openbao_manifests(config),
        # A sealed bootstrap server deliberately remains unready until its ceremony.
        skip_await=True,
        opts=pulumi.ResourceOptions(provider=provider, protect=True),
    )
    pulumi.export("endpoint", f"https://{config.hostname}")
    pulumi.export("namespace", config.namespace)
    pulumi.export("cluster", config.cluster)
    pulumi.export("exposure", config.exposure.value)


main()
