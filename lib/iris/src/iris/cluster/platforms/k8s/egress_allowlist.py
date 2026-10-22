# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CiliumNetworkPolicy manifests for ``EGRESS_POLICY_ALLOWLIST`` pods.

An allowlisted pod carries the label ``iris.egress=allowlist`` and is selected
by up to two policies; Cilium admits the union of their rules and denies all
other egress. Ingress is not restricted.

- ``iris-egress-allowlist``, one per Iris namespace, applied at
  ``iris cluster start``: DNS, every pod in the namespace (controller, finelog,
  other tasks), cluster nodes, the configured ``endpoints``, and Iris's base
  allowlist plus the cluster's ``allow`` rules.
- ``<pod>-egress``, one per task attempt whose job has its own rules, applied
  by the controller before the pod and deleted with the pod's other
  resources.

Domains become ``toFQDNs`` selectors and DNS rules. Cilium's DNS proxy answers
every name the pod's policies do not list with REFUSED, so looking up an
unlisted name fails at once instead of a connection timing out. ``toFQDNs``
admits only addresses outside the cluster; in-cluster destinations need an
endpoint rule even when reached by name.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from iris.cluster.config import EgressAllowlistConfig
from iris.cluster.egress import BASE_EGRESS_ALLOWLIST, is_domain_pattern, parse_egress_rules
from iris.cluster.platforms.k8s.constants import EGRESS_LABEL
from iris.cluster.runtime.sandbox import TaskNetwork
from iris.rpc import job_pb2

ALLOWLIST_POLICY_NAME = "iris-egress-allowlist"
TASK_POLICY_SUFFIX = "-egress"

_API_VERSION = "cilium.io/v2"
_KIND = "CiliumNetworkPolicy"
_NAMESPACE_LABEL = "k8s:io.kubernetes.pod.namespace"
_DNS_NAMESPACE = "kube-system"
# Cilium's "*" matches one DNS label, so cluster-domain names get one pattern
# per depth. glibc stops walking the resolv.conf search list at the first
# REFUSED answer, so every search candidate of an in-cluster name
# ("svc.namespace" becomes "svc.namespace.namespace.svc.cluster.local") must
# resolve rather than be refused.
_CLUSTER_DOMAIN_PATTERNS = tuple("*." * depth + "cluster.local" for depth in range(1, 5))
_NODE_ENTITIES = ("host", "remote-node")


def allowlist_policy_manifest(namespace: str, allowlist: EgressAllowlistConfig) -> dict:
    """The namespace-wide policy every ALLOWLIST pod in ``namespace`` receives."""
    rules = parse_egress_rules([*BASE_EGRESS_ALLOWLIST, *allowlist.allow])
    endpoints = [{}]
    for target in allowlist.endpoints:
        labels = {_NAMESPACE_LABEL: target.namespace}
        labels.update({f"k8s:{key}": value for key, value in target.labels.items()})
        endpoints.append({"matchLabels": labels})
    egress = [
        _dns_section([*_CLUSTER_DOMAIN_PATTERNS, *_domains(rules)]),
        *_destination_sections(rules),
        {"toEndpoints": endpoints},
        {"toEntities": list(_NODE_ENTITIES)},
    ]
    return _policy(ALLOWLIST_POLICY_NAME, namespace, {EGRESS_LABEL: TaskNetwork.ALLOWLIST.value}, egress)


def task_policy_name(pod_name: str) -> str:
    return f"{pod_name}{TASK_POLICY_SUFFIX}"


def task_policy_manifest(
    pod_name: str,
    namespace: str,
    pod_labels: Mapping[str, str],
    rules: Sequence[job_pb2.EgressRule],
) -> dict:
    """The policy adding a job's own ``rules`` to the pod that ``pod_labels`` select."""
    egress = list(_destination_sections(rules))
    domains = _domains(rules)
    if domains:
        egress.insert(0, _dns_section(domains))
    selector = {EGRESS_LABEL: TaskNetwork.ALLOWLIST.value, **pod_labels}
    return _policy(task_policy_name(pod_name), namespace, selector, egress)


def _policy(name: str, namespace: str, selector: Mapping[str, str], egress: list[dict]) -> dict:
    return {
        "apiVersion": _API_VERSION,
        "kind": _KIND,
        "metadata": {"name": name, "namespace": namespace},
        "spec": {"endpointSelector": {"matchLabels": dict(selector)}, "egress": egress},
    }


def _domains(rules: Iterable[job_pb2.EgressRule]) -> list[str]:
    return list(dict.fromkeys(rule.domain for rule in rules if rule.WhichOneof("destination") == "domain"))


def _name_selector(domain: str) -> dict:
    return {"matchPattern": domain} if is_domain_pattern(domain) else {"matchName": domain}


def _dns_section(names: Iterable[str]) -> dict:
    return {
        "toEndpoints": [{"matchLabels": {_NAMESPACE_LABEL: _DNS_NAMESPACE}}],
        "toPorts": [
            {
                "ports": [{"port": "53", "protocol": "ANY"}],
                "rules": {"dns": [_name_selector(name) for name in names]},
            }
        ],
    }


def _destination_sections(rules: Iterable[job_pb2.EgressRule]) -> list[dict]:
    """One ``toFQDNs`` and one ``toCIDRSet`` section per distinct port set."""
    domains: dict[tuple[int, ...], list[str]] = defaultdict(list)
    cidrs: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for rule in rules:
        ports = tuple(sorted(rule.ports))
        if rule.WhichOneof("destination") == "domain":
            domains[ports].append(rule.domain)
        else:
            cidrs[ports].append(rule.cidr)
    sections = []
    for ports, names in domains.items():
        section: dict = {"toFQDNs": [_name_selector(name) for name in dict.fromkeys(names)]}
        sections.append(_with_ports(section, ports))
    for ports, networks in cidrs.items():
        section = {"toCIDRSet": [{"cidr": cidr} for cidr in dict.fromkeys(networks)]}
        sections.append(_with_ports(section, ports))
    return sections


def _with_ports(section: dict, ports: tuple[int, ...]) -> dict:
    if ports:
        section["toPorts"] = [{"ports": [{"port": str(port), "protocol": "TCP"} for port in ports]}]
    return section
