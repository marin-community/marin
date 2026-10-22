# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Namespaced broker authority and private controller transport."""


def credential_secret_rbac(namespace: str, service_account: str) -> tuple[dict, dict]:
    metadata = {"name": "iris-task-sessions", "namespace": namespace}
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": metadata,
        "rules": [
            {
                "apiGroups": [""],
                "resources": ["secrets"],
                "verbs": ["get", "list", "create", "update", "patch", "delete"],
            }
        ],
    }
    binding = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": metadata,
        "subjects": [{"kind": "ServiceAccount", "name": service_account, "namespace": namespace}],
        "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": metadata["name"]},
    }
    return role, binding


def controller_certificate_manifests(namespace: str, service: str, tls_secret: str) -> list[dict]:
    """Use cert-manager's CA issuer; only the public CA is projected into tasks."""

    def resource(kind: str, name: str, spec: dict) -> dict:
        return {
            "apiVersion": "cert-manager.io/v1",
            "kind": kind,
            "metadata": {"name": name, "namespace": namespace},
            "spec": spec,
        }

    return [
        resource("Issuer", "iris-bootstrap", {"selfSigned": {}}),
        resource(
            "Certificate",
            "iris-controller-ca",
            {
                "isCA": True,
                "commonName": f"Iris controller CA ({namespace})",
                "secretName": "iris-controller-ca-key",
                "duration": "87600h",
                "issuerRef": {"name": "iris-bootstrap", "kind": "Issuer"},
                "privateKey": {"algorithm": "ECDSA", "size": 256},
            },
        ),
        resource("Issuer", "iris-controller-ca", {"ca": {"secretName": "iris-controller-ca-key"}}),
        resource(
            "Certificate",
            "iris-controller",
            {
                "secretName": tls_secret,
                "duration": "2160h",
                "renewBefore": "720h",
                "issuerRef": {"name": "iris-controller-ca", "kind": "Issuer"},
                "privateKey": {"algorithm": "ECDSA", "size": 256, "rotationPolicy": "Always"},
                "dnsNames": [
                    service,
                    f"{service}.{namespace}",
                    f"{service}.{namespace}.svc",
                    f"{service}.{namespace}.svc.cluster.local",
                    "localhost",
                ],
                # Port-forward clients verify loopback against this cluster's own CA.
                "ipAddresses": ["127.0.0.1", "::1"],
            },
        ),
    ]


def controller_tls_sidecar(port: int, native_port: int) -> tuple[dict, str]:
    configuration = f"""events {{ worker_connections 8192; }}
http {{
  access_log off;
  client_max_body_size 0;
  limit_req_zone $binary_remote_addr zone=task_renew:10m rate=20r/s;
  server {{
    listen {port} ssl;
    ssl_certificate /etc/iris-tls/tls.crt;
    ssl_certificate_key /etc/iris-tls/tls.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    location = /iris.cluster.ControllerService/RenewTaskSession {{
      limit_req zone=task_renew burst=100 nodelay;
      limit_req_status 429;
      proxy_pass http://127.0.0.1:{native_port};
      proxy_set_header Host $http_host;
      proxy_set_header X-Forwarded-Proto https;
      proxy_set_header X-Forwarded-For $remote_addr;
    }}
    location / {{
      proxy_pass http://127.0.0.1:{native_port};
      proxy_http_version 1.1;
      proxy_set_header Host $http_host;
      proxy_set_header X-Forwarded-Proto https;
      proxy_set_header X-Forwarded-For $remote_addr;
      proxy_buffering off;
      proxy_request_buffering off;
      proxy_read_timeout 3600s;
    }}
  }}
}}
"""
    return {
        "name": "controller-tls",
        "image": "nginx:stable-alpine",
        "command": [
            "/bin/sh",
            "-ec",
            "nginx -g 'daemon off;' & proxy_pid=$!; trap 'kill -TERM $proxy_pid' TERM INT; "
            "while kill -0 $proxy_pid 2>/dev/null; do sleep 60 & wait $!; nginx -s reload; done; wait $proxy_pid",
        ],
        "ports": [{"containerPort": port}],
        "volumeMounts": [
            {"name": "controller-tls", "mountPath": "/etc/iris-tls", "readOnly": True},
            {
                "name": "tls-proxy-config",
                "mountPath": "/etc/nginx/nginx.conf",
                "subPath": "nginx.conf",
                "readOnly": True,
            },
        ],
        "resources": {"requests": {"cpu": "250m", "memory": "128Mi"}, "limits": {"memory": "512Mi"}},
    }, configuration
