# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the service authorization proof on CW with synthetic identities."""

import pulumi
import pulumi_kubernetes as k8s

config = pulumi.Config("marin-auth-trial")
context = config.require("kubernetes_context")
image = config.require("image")
if context != "marin_US-WEST-04A":
    raise ValueError("This test rollout is restricted to the approved us-west4 cluster")
if "@sha256:" not in image:
    raise ValueError("The proof image must be pinned by digest")
provider = k8s.Provider("trial", kubeconfig="", context=context)
options = pulumi.ResourceOptions(provider=provider)
namespace = k8s.core.v1.Namespace(
    "trial",
    metadata={
        "name": "marin-auth-trial",
        "labels": {
            "pod-security.kubernetes.io/enforce": "restricted",
            "app.kubernetes.io/part-of": "marin-auth-trial",
        },
    },
    opts=options,
)
scoped = pulumi.ResourceOptions(provider=provider, depends_on=[namespace])
network = k8s.networking.v1.NetworkPolicy(
    "trial-isolation",
    metadata={"namespace": namespace.metadata["name"]},
    spec={"pod_selector": {}, "policy_types": ["Ingress", "Egress"], "ingress": [], "egress": []},
    opts=scoped,
)
job = k8s.batch.v1.Job(
    "proof",
    metadata={"namespace": namespace.metadata["name"], "name": "marin-auth-proof"},
    spec={
        "backoff_limit": 0,
        "active_deadline_seconds": 600,
        "ttl_seconds_after_finished": 86400,
        "template": {
            "metadata": {"labels": {"app.kubernetes.io/name": "marin-auth-proof"}},
            "spec": {
                "restart_policy": "Never",
                "automount_service_account_token": False,
                "security_context": {
                    "run_as_non_root": True,
                    "run_as_user": 1000,
                    "run_as_group": 1000,
                    "fs_group": 1000,
                    "seccomp_profile": {"type": "RuntimeDefault"},
                },
                "containers": [
                    {
                        "name": "proof",
                        "image": image,
                        "security_context": {
                            "allow_privilege_escalation": False,
                            "read_only_root_filesystem": True,
                            "capabilities": {"drop": ["ALL"]},
                        },
                        "env": [{"name": "HOME", "value": "/tmp"}],
                        "resources": {
                            "requests": {"cpu": "1", "memory": "1Gi", "ephemeral-storage": "1Gi"},
                            "limits": {"cpu": "4", "memory": "4Gi", "ephemeral-storage": "8Gi"},
                        },
                        "volume_mounts": [{"name": "scratch", "mount_path": "/tmp"}],
                    }
                ],
                "volumes": [{"name": "scratch", "empty_dir": {"size_limit": "2Gi"}}],
            },
        },
    },
    opts=pulumi.ResourceOptions(provider=provider, depends_on=[network]),
)
pulumi.export("namespace", namespace.metadata["name"])
pulumi.export("job", job.metadata["name"])
pulumi.export("image", image)
