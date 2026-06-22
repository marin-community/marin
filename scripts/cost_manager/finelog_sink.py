# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where cost events go: stdout for dry runs, or the finelog ``cost.events`` table.

:func:`open_sink` is a context manager that yields a :class:`Sink` for the
configured destination:

- ``--dry-run`` → :class:`StdoutSink` (prints a table; never connects).
- ``finelog.url`` set → connect a ``LogClient`` directly (e.g. a tunnel a
  caller already opened, or a local server).
- ``finelog.config`` set → load the finelog deploy config and open an SSH
  (GCE) or ``kubectl port-forward`` (k8s) tunnel to the server. This is the CI
  default: it works headlessly with a service-account + SSH key, with no
  interactive IAP login.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from typing import Any, Protocol

from finelog.client import FlushResult, LogClient, StoragePolicy
from finelog.deploy.config import FinelogConfig, load_finelog_config
from rigging.tunnel import GcpSshForwardTarget, K8sPortForwardTarget, TunnelTarget, open_tunnel

from scripts.cost_manager.cost_event import COST_EVENTS_MAX_BYTES, COST_EVENTS_NAMESPACE, CostEvent

logger = logging.getLogger(__name__)

FLUSH_TIMEOUT = 30.0


class Sink(Protocol):
    def write(self, events: list[CostEvent]) -> None: ...

    def flush(self) -> None: ...


class StdoutSink:
    """Prints events as a table plus per-provider totals; writes nowhere."""

    def write(self, events: list[CostEvent]) -> None:
        if not events:
            print("(no cost events)")
            return
        print(f"{'usage_date':<12} {'provider':<10} {'category':<22} {'detail':<40} {'cost':>12} kind")
        for e in sorted(events, key=lambda x: (x.usage_date, x.provider, x.category, x.detail)):
            print(
                f"{e.usage_date:<12} {e.provider:<10} {e.category[:22]:<22} "
                f"{e.detail[:40]:<40} {e.cost:>12.4f} {e.currency} {e.amount_kind}"
            )
        totals: dict[str, float] = defaultdict(float)
        for e in events:
            totals[e.provider] += e.cost
        print("\n-- totals by provider --")
        for provider, total in sorted(totals.items()):
            print(f"{provider:<10} {total:>12.2f}")

    def flush(self) -> None:
        pass


class FinelogSink:
    """Appends events to the finelog ``cost.events`` namespace."""

    def __init__(self, client: LogClient, namespace: str) -> None:
        self._table = client.get_table(
            namespace,
            CostEvent,
            storage_policy=StoragePolicy(max_bytes=COST_EVENTS_MAX_BYTES),
        )

    def write(self, events: list[CostEvent]) -> None:
        self._table.write(events)

    def flush(self) -> None:
        result = self._table.flush(timeout=FLUSH_TIMEOUT)
        if result is not FlushResult.SUCCEEDED:
            raise RuntimeError(f"finelog flush did not complete within {FLUSH_TIMEOUT:.0f}s (result={result})")


@contextmanager
def open_sink(finelog_cfg: Mapping[str, Any], *, dry_run: bool, tunnel_timeout: float = 60.0) -> Iterator[Sink]:
    if dry_run:
        yield StdoutSink()
        return

    namespace = finelog_cfg.get("namespace", COST_EVENTS_NAMESPACE)
    url = finelog_cfg.get("url")
    if url:
        logger.info("Connecting to finelog at %s", url)
        with closing(LogClient.connect(url)) as client:
            yield FinelogSink(client, namespace)
        return

    config_name = finelog_cfg.get("config")
    if not config_name:
        raise ValueError("finelog config must set 'config' or 'url' (or pass --dry-run)")
    cfg = load_finelog_config(config_name)
    target = _tunnel_target(cfg)
    logger.info("Opening tunnel to finelog '%s' (%s)", cfg.name, type(target).__name__)
    with open_tunnel(target, timeout=tunnel_timeout) as tunnel_url, closing(LogClient.connect(tunnel_url)) as client:
        yield FinelogSink(client, namespace)


def _tunnel_target(cfg: FinelogConfig) -> TunnelTarget:
    """Build a rigging tunnel target from a finelog deployment block."""
    if cfg.deployment.gcp is not None:
        gcp = cfg.deployment.gcp
        return GcpSshForwardTarget(
            project=gcp.project,
            zone=gcp.zone,
            instance=cfg.name,
            port=cfg.port,
            impersonate_service_account=gcp.service_account,
        )
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    return K8sPortForwardTarget(namespace=k8s.namespace, service=cfg.name, port=cfg.port)
