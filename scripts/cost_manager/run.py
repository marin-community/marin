# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch provider costs and write them to finelog.

Loads ``config.yaml``, runs each enabled provider backend over a trailing UTC
window, and appends the resulting :class:`CostEvent` rows to the finelog
``cost.events`` namespace (or prints them with ``--dry-run``).

One provider failing (missing key, API/permission error) does not abort the
run: the remaining backends still write, and the process exits non-zero so CI
surfaces the failure.

Examples::

    # Local smoke: fetch + print, never connect to finelog.
    uv run python -m scripts.cost_manager.run --dry-run

    # Production daily run (CI): tunnel to the 'marin' finelog server.
    uv run python -m scripts.cost_manager.run
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import click
import yaml

from scripts.cost_manager.backends import BACKENDS
from scripts.cost_manager.cost_event import CostEvent, CostFetchError, DateWindow, stamp_collected
from scripts.cost_manager.finelog_sink import open_sink

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_LOOKBACK_DAYS = 3


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a yaml mapping at top level")
    return raw


def _run_backends(
    providers: list[dict[str, Any]], window: DateWindow, only: set[str]
) -> tuple[list[CostEvent], list[str]]:
    """Run each enabled (and selected) backend; return (events, failed_providers)."""
    events: list[CostEvent] = []
    failed: list[str] = []
    for provider_cfg in providers:
        name = provider_cfg["name"]
        if only and name not in only:
            continue
        if not provider_cfg.get("enabled", False):
            logger.info("Skipping disabled provider %s", name)
            continue
        fetcher = BACKENDS.get(name)
        if fetcher is None:
            logger.error("Unknown provider %r (known: %s)", name, ", ".join(sorted(BACKENDS)))
            failed.append(name)
            continue
        try:
            provider_events = fetcher(provider_cfg, window)
        except CostFetchError as exc:
            logger.error("Provider %s failed: %s", name, exc)
            failed.append(name)
            continue
        except Exception:
            logger.exception("Provider %s raised an unexpected error", name)
            failed.append(name)
            continue
        logger.info("Provider %s: %d events, $%.2f", name, len(provider_events), sum(e.cost for e in provider_events))
        events.extend(provider_events)
    return events, failed


@click.command(help=__doc__)
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG, show_default=True)
@click.option("--lookback-days", type=int, default=None, help="Override config lookback_days (trailing UTC days).")
@click.option("--provider", "providers_filter", multiple=True, help="Restrict to these providers (repeatable).")
@click.option("--today", "today_override", default=None, help="UTC 'today' as YYYY-MM-DD (default: now). For backfill.")
@click.option("--finelog-url", default=None, help="Override finelog.url (direct connect, e.g. a pre-opened tunnel).")
@click.option("--finelog-config", default=None, help="Override finelog.config (deploy config name to tunnel to).")
@click.option("--dry-run/--no-dry-run", default=False, help="Fetch and print; never connect to finelog.")
def main(
    config_path: Path,
    lookback_days: int | None,
    providers_filter: tuple[str, ...],
    today_override: str | None,
    finelog_url: str | None,
    finelog_config: str | None,
    dry_run: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = _load_config(config_path)

    today = dt.date.fromisoformat(today_override) if today_override else dt.datetime.now(dt.UTC).date()
    days = lookback_days if lookback_days is not None else config.get("lookback_days", DEFAULT_LOOKBACK_DAYS)
    window = DateWindow.trailing(days, today=today)
    logger.info("Fetching costs for UTC window %s..%s", window.start, window.end)

    providers = config.get("providers", [])
    events, failed = _run_backends(providers, window, set(providers_filter))

    collected_at = dt.datetime.now(dt.UTC)
    events = stamp_collected(events, collected_at)

    finelog_cfg = dict(config.get("finelog", {}))
    if finelog_url:
        finelog_cfg["url"] = finelog_url
    if finelog_config:
        finelog_cfg["config"] = finelog_config

    with open_sink(finelog_cfg, dry_run=dry_run) as sink:
        sink.write(events)
        sink.flush()
    logger.info("Wrote %d cost events (dry_run=%s)", len(events), dry_run)

    if failed:
        raise SystemExit(f"providers failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
