# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Egress allowlist rules for ``EGRESS_POLICY_ALLOWLIST`` jobs.

A rule names one destination, a DNS name, a one-label wildcard pattern or a
CIDR network, and optionally the TCP ports it admits. Users write rules as
strings (``iris job run --egress-allow``, ``IrisClient.submit(egress_allow=)``,
``defaults.egress_allowlist.allow``); :func:`parse_egress_rule`
turns one into a ``job_pb2.EgressRule``, and the controller re-validates a
submitted job's rules with :func:`validate_egress_rules`.

Syntax: ``DEST[:PORT[,PORT...]]``, where ``DEST`` is ``api.openai.com``,
``*.openai.com``, ``203.0.113.0/24`` or a bare address. An IPv6 destination
with ports goes in brackets: ``[2001:db8::/32]:443``.
"""

import ipaddress
import re
from collections.abc import Iterable, Sequence

from iris.rpc import job_pb2

# Destinations every ALLOWLIST task may reach on any cluster that enables the
# allowlist: package indexes and toolchains, source and model hubs, Google
# Cloud APIs and storage, and experiment tracking. A "*" stands for one DNS
# label, so deeper names are listed explicitly.
BASE_EGRESS_ALLOWLIST: tuple[str, ...] = (
    # Python packages and wheels.
    "pypi.org",
    "files.pythonhosted.org",
    "download.pytorch.org",
    "pypi.nvidia.com",
    "wheels.vllm.ai",
    "flashinfer.ai",
    # Source code, release assets and uv's managed Python builds.
    "github.com",
    "*.github.com",
    "*.githubusercontent.com",
    "ghcr.io",
    # Other package ecosystems and OS packages.
    "registry.npmjs.org",
    "registry.yarnpkg.com",
    "conda.anaconda.org",
    "repo.anaconda.com",
    "crates.io",
    "index.crates.io",
    "static.crates.io",
    "static.rust-lang.org",
    "sh.rustup.rs",
    "deb.debian.org",
    "security.debian.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "developer.download.nvidia.com",
    # Hugging Face models and datasets, including its Xet storage.
    "huggingface.co",
    "*.huggingface.co",
    "hf.co",
    "*.hf.co",
    "*.xethub.hf.co",
    "*.*.cdn.hf.co",
    # Google Cloud APIs, Cloud Storage and Artifact Registry.
    "*.googleapis.com",
    "*.pkg.dev",
    # Weights & Biases.
    "wandb.ai",
    "*.wandb.ai",
)

# Upper bound on a job's own rules, and on the cluster's. Each rule becomes a
# selector in a CiliumNetworkPolicy; a job needing more should use the cluster
# network.
MAX_EGRESS_RULES = 64

_DNS_LABEL = re.compile(r"^[a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?$")
_WILDCARD = "*"
_MAX_DOMAIN_LENGTH = 253
# A pattern must fix at least this many trailing labels, so "*.com" is rejected.
_MIN_FIXED_LABELS = 2


def parse_egress_rule(text: str) -> job_pb2.EgressRule:
    """Parse ``DEST[:PORT[,PORT...]]`` into a validated rule.

    Raises:
        ValueError: The text is not a valid domain, pattern, address or network,
            or a port is not an integer in 1..65535.
    """
    destination, ports = _split_ports(text.strip())
    rule = job_pb2.EgressRule(ports=ports)
    if _looks_like_address(destination):
        rule.cidr = _normalize_cidr(destination)
    else:
        rule.domain = destination.lower().rstrip(".")
    validate_egress_rule(rule)
    return rule


def parse_egress_rules(texts: Iterable[str]) -> list[job_pb2.EgressRule]:
    """Parse rules in order, rejecting a list longer than ``MAX_EGRESS_RULES``."""
    rules = [parse_egress_rule(text) for text in texts]
    _check_rule_count(rules)
    return rules


def validate_egress_rules(rules: Sequence[job_pb2.EgressRule]) -> None:
    """Raise ValueError for more than ``MAX_EGRESS_RULES`` rules or any invalid rule."""
    _check_rule_count(rules)
    for rule in rules:
        validate_egress_rule(rule)


def _check_rule_count(rules: Sequence[job_pb2.EgressRule]) -> None:
    if len(rules) > MAX_EGRESS_RULES:
        raise ValueError(f"At most {MAX_EGRESS_RULES} egress rules are allowed; got {len(rules)}")


def validate_egress_rule(rule: job_pb2.EgressRule) -> None:
    """Raise ValueError unless ``rule`` names one valid destination and valid ports."""
    which = rule.WhichOneof("destination")
    if which == "domain":
        _validate_domain(rule.domain)
    elif which == "cidr":
        network = ipaddress.ip_network(rule.cidr, strict=True)
        if network.prefixlen == 0:
            raise ValueError(f"Egress rule {rule.cidr} admits every address; use the cluster egress policy instead")
    else:
        raise ValueError("Egress rule names no destination")
    for port in rule.ports:
        if not 1 <= port <= 65535:
            raise ValueError(f"Egress rule port {port} is outside 1..65535")
    if len(set(rule.ports)) != len(rule.ports):
        raise ValueError(f"Egress rule {format_egress_rule(rule)} repeats a port")


def format_egress_rule(rule: job_pb2.EgressRule) -> str:
    """Render ``rule`` in the syntax :func:`parse_egress_rule` accepts."""
    destination = rule.domain if rule.WhichOneof("destination") == "domain" else rule.cidr
    if not rule.ports:
        return destination
    if ":" in destination:
        destination = f"[{destination}]"
    return f"{destination}:{','.join(str(port) for port in rule.ports)}"


def is_domain_pattern(domain: str) -> bool:
    return _WILDCARD in domain.split(".")


def rule_domains(rules: Iterable[job_pb2.EgressRule]) -> list[str]:
    """The distinct domains and patterns of ``rules``, in order."""
    return list(dict.fromkeys(rule.domain for rule in rules if rule.WhichOneof("destination") == "domain"))


def _split_ports(text: str) -> tuple[str, list[int]]:
    if text.startswith("["):
        destination, bracket, rest = text[1:].partition("]")
        if not bracket:
            raise ValueError(f"Egress rule {text!r} has an unclosed '['")
        if not rest:
            return destination, []
        if not rest.startswith(":"):
            raise ValueError(f"Egress rule {text!r} must continue with ':PORT' after ']'")
        return destination, _parse_ports(rest[1:], text)
    # More than one colon is an unbracketed IPv6 destination without ports.
    if text.count(":") == 1:
        destination, _, ports = text.partition(":")
        return destination, _parse_ports(ports, text)
    return text, []


def _parse_ports(ports: str, text: str) -> list[int]:
    try:
        return [int(port) for port in ports.split(",")]
    except ValueError:
        raise ValueError(f"Egress rule {text!r} has a non-integer port") from None


def _looks_like_address(destination: str) -> bool:
    if "/" in destination or ":" in destination:
        return True
    try:
        ipaddress.ip_address(destination)
    except ValueError:
        return False
    return True


def _normalize_cidr(destination: str) -> str:
    if "/" in destination:
        return str(ipaddress.ip_network(destination, strict=True))
    return str(ipaddress.ip_network(ipaddress.ip_address(destination)))


def _validate_domain(domain: str) -> None:
    if domain != domain.lower() or domain.endswith("."):
        raise ValueError(f"Egress domain {domain!r} must be lowercase without a trailing dot")
    if len(domain) > _MAX_DOMAIN_LENGTH:
        raise ValueError(f"Egress domain {domain!r} is longer than {_MAX_DOMAIN_LENGTH} characters")
    labels = domain.split(".")
    fixed = 0
    for label in reversed(labels):
        if label == _WILDCARD:
            break
        fixed += 1
    if any(label != _WILDCARD for label in labels[: len(labels) - fixed]):
        raise ValueError(f"Egress domain {domain!r} may use '*' only for its leftmost labels")
    if fixed < _MIN_FIXED_LABELS:
        raise ValueError(
            f"Egress domain {domain!r} needs at least {_MIN_FIXED_LABELS} labels after its wildcards, "
            "such as '*.example.com'"
        )
    for label in labels[len(labels) - fixed :]:
        if not _DNS_LABEL.match(label):
            raise ValueError(f"Egress domain {domain!r} has an invalid label {label!r}")
