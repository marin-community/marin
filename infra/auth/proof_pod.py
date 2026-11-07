# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the isolated staging proof Pod; apply its JSON only after reviewing it."""

import argparse
import json
import re

NAMESPACE = "marin-auth-staging"
POD_NAME = "auth-flow-proof-10950"
TOKEN_SECRETS = ("broker-token", "openfga-token", "task-canary-token")


def proof_pod(image: str, identity_secret: str) -> dict:
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("The proof image requires an explicit sha256 digest")
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": POD_NAME,
            "namespace": NAMESPACE,
            "labels": {"app.kubernetes.io/name": "auth-staging-proof", "marin.dev/auth-client": "true"},
        },
        "spec": {
            "serviceAccountName": "auth-staging-proof",
            "automountServiceAccountToken": True,
            "restartPolicy": "Never",
            "activeDeadlineSeconds": 3600,
            "terminationGracePeriodSeconds": 10,
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 1000,
                "runAsGroup": 1000,
                "fsGroup": 1000,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "proof",
                    "image": image,
                    "workingDir": "/opt/marin-source",
                    "command": ["sleep", "3600"],
                    "env": [{"name": "HOME", "value": "/proof"}],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "readOnlyRootFilesystem": True,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "resources": {"requests": {"cpu": "1", "memory": "1Gi"}, "limits": {"cpu": "4", "memory": "4Gi"}},
                    "volumeMounts": [
                        {"name": "proof", "mountPath": "/proof"},
                        {"name": "tmp", "mountPath": "/tmp"},
                        {"name": "policy", "mountPath": "/etc/marin-auth", "readOnly": True},
                        {"name": "ca", "mountPath": "/var/run/marin-auth/ca", "readOnly": True},
                        {"name": "database", "mountPath": "/var/run/marin-auth/database", "readOnly": True},
                        {"name": "identities", "mountPath": "/var/run/marin-auth/proof", "readOnly": True},
                        *(
                            {"name": name, "mountPath": f"/var/run/marin-auth/{name}", "readOnly": True}
                            for name in TOKEN_SECRETS
                        ),
                    ],
                }
            ],
            "volumes": [
                {"name": "proof", "emptyDir": {"sizeLimit": "2Gi"}},
                {"name": "tmp", "emptyDir": {"sizeLimit": "1Gi"}},
                {
                    "name": "policy",
                    "configMap": {"name": "marin-auth-policy", "items": [{"key": "policy.yaml", "path": "policy.yaml"}]},
                },
                {
                    "name": "ca",
                    "secret": {
                        "secretName": "auth-ca",
                        "defaultMode": 0o440,
                        "items": [{"key": "ca.crt", "path": "ca.crt"}],
                    },
                },
                {
                    "name": "database",
                    "secret": {
                        "secretName": "grants-database",
                        "defaultMode": 0o440,
                        "items": [{"key": "url", "path": "url"}],
                    },
                },
                {
                    "name": "identities",
                    "secret": {
                        "secretName": identity_secret,
                        "defaultMode": 0o440,
                        "items": [
                            {"key": name, "path": name} for name in ("launcher.jwt", "controller.jwt", "grants.pem")
                        ],
                    },
                },
                *(
                    {
                        "name": name,
                        "secret": {
                            "secretName": name,
                            "defaultMode": 0o440,
                            "items": [{"key": "token", "path": "token"}],
                        },
                    }
                    for name in TOKEN_SECRETS
                ),
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--identity-secret", default="auth-proof-identities")
    args = parser.parse_args()
    print(json.dumps(proof_pod(args.image, args.identity_secret), indent=2))


if __name__ == "__main__":
    main()
