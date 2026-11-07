# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A disposable CloudNativePG deployment, restricted to the approved CW trial."""

import pulumi
import pulumi_kubernetes as k8s

CHART_VERSION = "0.29.1"
OPERATOR_IMAGE = "1.30.1@sha256:923c267ec29636db3bee20f993d0ec4973fa22998e1adad37da79e4d32b5bc07"
POSTGRES_IMAGE = (
    "ghcr.io/cloudnative-pg/postgresql:17@sha256:70664ebcfa1100361b5bdc28bbf06fdbe08db2dc4ad7bd14de33c5e05fe8ea8e"
)


def scope_operator(resource: dict, _options: pulumi.ResourceOptions) -> None:
    """Limit admission and mutable cluster permissions to this installation."""
    if resource["kind"] in {"MutatingWebhookConfiguration", "ValidatingWebhookConfiguration"}:
        for webhook in resource["webhooks"]:
            webhook["namespaceSelector"] = {"matchLabels": {"kubernetes.io/metadata.name": NAMESPACE}}
    if resource["kind"] == "ClusterRole":
        for rule in resource.get("rules", []):
            if "admissionregistration.k8s.io" in rule.get("apiGroups", []):
                rule["resourceNames"] = ["cnpg-mutating-webhook-configuration", "cnpg-validating-webhook-configuration"]


config = pulumi.Config("marin-postgres-trial")
NAMESPACE = config.get("namespace") or "marin-pg-trial"
CLUSTER = config.get("cluster") or "auth-postgres"
if NAMESPACE not in {"marin-pg-trial", "marin-auth-db-staging"}:
    raise ValueError("Only isolated trial and auth staging namespaces are supported")
retained = NAMESPACE == "marin-auth-db-staging"
context = config.require("kubernetes_context")
if context != "marin_US-WEST-04A":
    raise ValueError("The PostgreSQL prototype is restricted to the approved us-west4 cluster")
api_cidrs = config.require_object("kubernetes_api_cidrs")
dns_cidrs = config.require_object("dns_cidrs")
storage_class = config.require("storage_class")
provider = k8s.Provider("trial", kubeconfig="", context=context)
namespace = k8s.core.v1.Namespace(
    "trial",
    metadata={
        "name": NAMESPACE,
        "labels": {"pod-security.kubernetes.io/enforce": "restricted", "app.kubernetes.io/part-of": NAMESPACE},
    },
    opts=pulumi.ResourceOptions(provider=provider, protect=retained, retain_on_delete=retained),
)
scoped = pulumi.ResourceOptions(provider=provider, depends_on=[namespace])
network = k8s.networking.v1.NetworkPolicy(
    "trial-isolation",
    metadata={"namespace": NAMESPACE},
    spec={
        "pod_selector": {},
        "policy_types": ["Ingress", "Egress"],
        "ingress": [{"from": [{"pod_selector": {}}]}],
        "egress": [
            {"to": [{"pod_selector": {}}]},
            {
                "to": [{"ip_block": {"cidr": cidr}} for cidr in api_cidrs],
                "ports": [{"protocol": "TCP", "port": 443}, {"protocol": "TCP", "port": 6443}],
            },
            {
                "to": [
                    *({"ip_block": {"cidr": cidr}} for cidr in dns_cidrs),
                    {
                        "namespace_selector": {"match_labels": {"kubernetes.io/metadata.name": "kube-system"}},
                        "pod_selector": {"match_labels": {"k8s-app": "node-local-dns"}},
                    },
                ],
                "ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}],
            },
        ],
    },
    opts=scoped,
)
if retained:
    k8s.networking.v1.NetworkPolicy(
        "auth-database-clients",
        metadata={"namespace": NAMESPACE},
        spec={
            "pod_selector": {"match_labels": {"cnpg.io/cluster": CLUSTER}},
            "policy_types": ["Ingress"],
            "ingress": [
                {
                    "from": [
                        {"namespace_selector": {"match_labels": {"kubernetes.io/metadata.name": "marin-auth-staging"}}}
                    ],
                    "ports": [{"protocol": "TCP", "port": 5432}],
                }
            ],
        },
        opts=scoped,
    )
admission = k8s.networking.v1.NetworkPolicy(
    "admission",
    metadata={"namespace": NAMESPACE},
    spec={
        "pod_selector": {"match_labels": {"app.kubernetes.io/name": "cloudnative-pg"}},
        "policy_types": ["Ingress"],
        "ingress": [{"ports": [{"protocol": "TCP", "port": 9443}]}],
    },
    opts=scoped,
)
operator = k8s.helm.v3.Chart(
    "operator",
    k8s.helm.v3.ChartOpts(
        chart="cloudnative-pg",
        version=CHART_VERSION,
        namespace=NAMESPACE,
        fetch_opts=k8s.helm.v3.FetchOpts(repo="https://cloudnative-pg.github.io/charts"),
        values={
            "fullnameOverride": NAMESPACE,
            "config": {"clusterWide": False},
            "image": {"tag": OPERATOR_IMAGE},
            "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"cpu": "1", "memory": "512Mi"}},
        },
        transformations=[scope_operator],
    ),
    opts=pulumi.ResourceOptions(provider=provider, depends_on=[network, admission]),
)
database = k8s.apiextensions.CustomResource(
    "database",
    api_version="postgresql.cnpg.io/v1",
    kind="Cluster",
    metadata={"name": CLUSTER, "namespace": NAMESPACE},
    spec={
        "instances": 3,
        "imageName": POSTGRES_IMAGE,
        "enableSuperuserAccess": False,
        "bootstrap": {"initdb": {"database": "authproof", "owner": "authproof"}},
        **(
            {
                "managed": {
                    "roles": [
                        {
                            "name": name,
                            "ensure": "present",
                            "login": True,
                            "passwordSecret": {"name": f"{name}-database"},
                        }
                        for name in ("openfga", "grants")
                    ]
                }
            }
            if retained
            else {}
        ),
        "storage": {"storageClass": storage_class, "size": "5Gi"},
        "resources": {"requests": {"cpu": "500m", "memory": "512Mi"}, "limits": {"cpu": "2", "memory": "1Gi"}},
        "affinity": {
            "podAntiAffinityType": "preferred",
            "topologyKey": "topology.kubernetes.io/zone",
            "tolerations": [{"key": "is_gpu", "operator": "Equal", "value": "true", "effect": "PreferNoSchedule"}],
        },
        "topologySpreadConstraints": [
            {
                "maxSkew": 1,
                "minDomains": 2,
                "topologyKey": "topology.kubernetes.io/zone",
                "whenUnsatisfiable": "DoNotSchedule",
                "labelSelector": {"matchLabels": {"cnpg.io/cluster": CLUSTER, "cnpg.io/podRole": "instance"}},
            }
        ],
        "primaryUpdateStrategy": "supervised",
        "failoverDelay": 5,
        "postgresql": {
            "synchronous": {"method": "any", "number": 1, "dataDurability": "required", "failoverQuorum": True},
            "pg_hba": ["hostnossl all all 0.0.0.0/0 reject", "hostnossl all all ::/0 reject"],
        },
    },
    opts=pulumi.ResourceOptions(
        provider=provider, depends_on=operator.ready, protect=retained, retain_on_delete=retained
    ),
)
if retained:
    for name in ("openfga", "grants", "openfga-recovery", "grants-recovery"):
        k8s.apiextensions.CustomResource(
            f"{name}-database",
            api_version="postgresql.cnpg.io/v1",
            kind="Database",
            metadata={"name": name, "namespace": NAMESPACE},
            spec={"name": name, "owner": name.removesuffix("-recovery"), "cluster": {"name": CLUSTER}},
            opts=pulumi.ResourceOptions(provider=provider, depends_on=[database], protect=True, retain_on_delete=True),
        )
pulumi.export("namespace", NAMESPACE)
pulumi.export("cluster", CLUSTER)
pulumi.export("primary_service", f"{CLUSTER}-rw.{NAMESPACE}.svc")
pulumi.export("credentials_secret", f"{CLUSTER}-app")
pulumi.export("ca_secret", f"{CLUSTER}-ca")

proof_image = config.get("proof_image")
if proof_image is not None:
    if "@sha256:" not in proof_image:
        raise ValueError("The failover proof image must be pinned by digest")
    account = k8s.core.v1.ServiceAccount(
        "proof", metadata={"name": "postgres-proof", "namespace": NAMESPACE}, opts=scoped
    )
    role = k8s.rbac.v1.Role(
        "proof",
        metadata={"namespace": NAMESPACE},
        rules=[
            {
                "api_groups": [""],
                "resources": ["pods"],
                "resource_names": [f"{CLUSTER}-{index}" for index in range(1, 4)],
                "verbs": ["get", "delete"],
            },
            {
                "api_groups": ["postgresql.cnpg.io"],
                "resources": ["clusters"],
                "resource_names": [CLUSTER],
                "verbs": ["get", "patch"],
            },
        ],
        opts=scoped,
    )
    binding = k8s.rbac.v1.RoleBinding(
        "proof",
        metadata={"namespace": NAMESPACE},
        role_ref={"api_group": "rbac.authorization.k8s.io", "kind": "Role", "name": role.metadata["name"]},
        subjects=[{"kind": "ServiceAccount", "name": account.metadata["name"], "namespace": NAMESPACE}],
        opts=scoped,
    )
    proof = k8s.batch.v1.Job(
        "proof",
        metadata={"namespace": NAMESPACE, "name": "postgres-auth-proof"},
        spec={
            "backoff_limit": 0,
            "active_deadline_seconds": 900,
            "ttl_seconds_after_finished": 86400,
            "template": {
                "spec": {
                    "service_account_name": account.metadata["name"],
                    "restart_policy": "Never",
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
                            "image": proof_image,
                            "command": ["python", "-m", "infra.postgres.prove"],
                            "env": [{"name": "HOME", "value": "/tmp"}],
                            "security_context": {
                                "allow_privilege_escalation": False,
                                "read_only_root_filesystem": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "512Mi"},
                                "limits": {"cpu": "2", "memory": "2Gi"},
                            },
                            "volume_mounts": [
                                {"name": "scratch", "mount_path": "/tmp"},
                                {
                                    "name": "credentials",
                                    "mount_path": "/var/run/postgres/credentials",
                                    "read_only": True,
                                },
                                {"name": "ca", "mount_path": "/var/run/postgres/ca", "read_only": True},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "scratch", "empty_dir": {"size_limit": "1Gi"}},
                        {
                            "name": "credentials",
                            "secret": {
                                "secret_name": f"{CLUSTER}-app",
                                "default_mode": 0o440,
                                "items": [
                                    {"key": "username", "path": "username"},
                                    {"key": "password", "path": "password"},
                                ],
                            },
                        },
                        {
                            "name": "ca",
                            "secret": {"secret_name": f"{CLUSTER}-ca", "items": [{"key": "ca.crt", "path": "ca.crt"}]},
                        },
                    ],
                }
            },
        },
        opts=pulumi.ResourceOptions(provider=provider, depends_on=[database, binding], delete_before_replace=True),
    )
    pulumi.export("proof_job", proof.metadata["name"])
