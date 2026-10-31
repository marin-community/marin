# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operator-owned deployment with Pod reads and exec, and no signing key."""

GATEWAY_NAME = "iris-exec-gateway"
GATEWAY_PORT = 10002
PUBLIC_KEY_PATH = "/etc/iris-exec/public.pem"


def gateway_manifests(*, namespace: str, image: str, gateway_id: str, issuer: str, public_key: str) -> list[dict]:
    """Render separate gateway identity, pinned verification material, and service."""
    metadata = {"name": GATEWAY_NAME, "namespace": namespace}
    labels = {"app": GATEWAY_NAME}
    return [
        {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": metadata},
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": metadata,
            "rules": [
                {"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]},
                {"apiGroups": [""], "resources": ["pods/exec"], "verbs": ["get", "create"]},
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": metadata,
            "subjects": [{"kind": "ServiceAccount", "name": GATEWAY_NAME, "namespace": namespace}],
            "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": GATEWAY_NAME},
        },
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": metadata, "data": {"public.pem": public_key}},
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": metadata,
            "spec": {
                # Sessions belong to one process; the service must not randomly
                # route Send/Receive to another replica. Scale by gateway identity.
                "replicas": 1,
                "strategy": {"type": "Recreate"},
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "serviceAccountName": GATEWAY_NAME,
                        "securityContext": {"runAsNonRoot": True, "runAsUser": 65532, "runAsGroup": 65532},
                        "containers": [
                            {
                                "name": "gateway",
                                "image": image,
                                "env": [{"name": "HOME", "value": "/nonexistent"}],
                                "command": [
                                    "iris",
                                    "exec-gateway",
                                    "serve",
                                    "--namespace",
                                    namespace,
                                    "--gateway-id",
                                    gateway_id,
                                    "--issuer",
                                    issuer,
                                    "--public-key",
                                    PUBLIC_KEY_PATH,
                                    "--host",
                                    "0.0.0.0",
                                    "--port",
                                    str(GATEWAY_PORT),
                                ],
                                "ports": [{"name": "rpc", "containerPort": GATEWAY_PORT}],
                                "resources": {"requests": {"cpu": "1", "memory": "512Mi"}, "limits": {"memory": "1Gi"}},
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "readOnlyRootFilesystem": True,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "volumeMounts": [
                                    {"name": "verification", "mountPath": "/etc/iris-exec", "readOnly": True}
                                ],
                                "readinessProbe": {"tcpSocket": {"port": "rpc"}},
                                "livenessProbe": {"tcpSocket": {"port": "rpc"}},
                            }
                        ],
                        "volumes": [{"name": "verification", "configMap": {"name": GATEWAY_NAME}}],
                    },
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": metadata,
            "spec": {"selector": labels, "ports": [{"name": "rpc", "port": GATEWAY_PORT, "targetPort": "rpc"}]},
        },
    ]
