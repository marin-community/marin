# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CiliumNetworkPolicy manifests for ``EGRESS_POLICY_ALLOWLIST`` pods.

An allowlisted pod carries the label ``iris.egress=allowlist`` and is selected
by up to two policies; Cilium admits the union of their rules and denies all
other egress. Ingress is not restricted.

- ``iris-egress-allowlist``, one per Iris namespace, applied at
  ``iris cluster start``: DNS, every pod in the namespace (controller, finelog,
  other tasks), cluster nodes, ``kubernetes_provider.cilium.egress_endpoints``,
  and Iris's base allowlist plus ``defaults.egress_allowlist.allow``.
- ``<pod>-egress``, one per task attempt whose job has its own rules, applied
  by the controller before the pod and owned by it, so Kubernetes deletes it
  with the pod.

Domains become ``toFQDNs`` selectors and DNS rules. Cilium's DNS proxy answers
every name the pod's policies do not list with REFUSED, so looking up an
unlisted name fails at once instead of a connection timing out. ``toFQDNs``
admits only addresses outside the cluster; in-cluster destinations need an
endpoint rule even when reached by name.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from iris.cluster.config import EgressEndpointsConfig
from iris.cluster.egress import BASE_EGRESS_ALLOWLIST, is_domain_pattern, parse_egress_rules, rule_domains
from iris.cluster.platforms.k8s.constants import EGRESS_LABEL
from iris.cluster.runtime.sandbox import TaskNetwork
from iris.rpc import job_pb2

ALLOWLIST_POLICY_NAME = "iris-egress-allowlist"

_API_VERSION = "cilium.io/v2"
_KIND = "CiliumNetworkPolicy"
_NAMESPACE_LABEL = "k8s:io.kubernetes.pod.namespace"
_DNS_NAMESPACE = "kube-system"
# Every search candidate a lookup tries must reach CoreDNS rather than be
# refused: musl, which static binaries such as uv use, fails the whole lookup
# at the first REFUSED answer. With the pod's "ndots:5", a name of up to five
# labels is first tried under "<namespace>.svc.cluster.local", giving nine
# labels. Cilium's "*" matches one label, so each depth gets a pattern.
_CLUSTER_DOMAIN_PATTERNS = tuple("*." * depth + "cluster.local" for depth in range(1, 8))
_NODE_ENTITIES = ("host", "remote-node")


def allowlist_policy_manifest(
    namespace: str,
    allow: Sequence[str],
    endpoints: Sequence[EgressEndpointsConfig],
) -> dict:
    """The namespace-wide policy every ALLOWLIST pod in ``namespace`` receives.

    ``allow`` holds the cluster's rules, which add to Iris's base allowlist, and
    ``endpoints`` the pods outside ``namespace`` that tasks may reach.
    """
    rules = [*parse_egress_rules(BASE_EGRESS_ALLOWLIST), *parse_egress_rules(allow)]
    selectors: list[dict] = [{}]
    for target in endpoints:
        labels = {_NAMESPACE_LABEL: target.namespace}
        labels.update({f"k8s:{key}": value for key, value in target.labels.items()})
        selectors.append({"matchLabels": labels})
    egress = [
        _dns_section([*_CLUSTER_DOMAIN_PATTERNS, *rule_domains(rules)]),
        *_destination_sections(rules),
        {"toEndpoints": selectors},
        {"toEntities": list(_NODE_ENTITIES)},
    ]
    return _policy(ALLOWLIST_POLICY_NAME, namespace, {EGRESS_LABEL: TaskNetwork.ALLOWLIST.value}, egress)


def task_policy_manifest(
    pod_name: str,
    namespace: str,
    pod_labels: Mapping[str, str],
    rules: Sequence[job_pb2.EgressRule],
) -> dict:
    """The policy adding a job's own ``rules`` to the pod that ``pod_labels`` select."""
    egress = list(_destination_sections(rules))
    domains = rule_domains(rules)
    if domains:
        egress.insert(0, _dns_section(domains))
    selector = {EGRESS_LABEL: TaskNetwork.ALLOWLIST.value, **pod_labels}
    return _policy(f"{pod_name}-egress", namespace, selector, egress)


def _policy(name: str, namespace: str, selector: Mapping[str, str], egress: list[dict]) -> dict:
    return {
        "apiVersion": _API_VERSION,
        "kind": _KIND,
        "metadata": {"name": name, "namespace": namespace},
        "spec": {"endpointSelector": {"matchLabels": dict(selector)}, "egress": egress},
    }


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
