"""Which sandbox provider runs container trials, and what gate receipts may claim.

The pipeline's sandbox code is provider-neutral: every client comes from the
staged ``dt.py``'s ``client()``, and silo ships a drop-in ``dt.py`` whose client
reads the same Daytona parameter objects.  What must NOT be neutral is what a
receipt claims.  A silo run recording ``daytona-network-block-all`` would be a
false statement in the gate evidence, so the claimed adapter and isolation
mechanism are derived from the provider that actually ran.

``submit.sh --sandbox silo`` exports ``CAPABILITY_SANDBOX_PROVIDER=silo``; the
default remains Daytona so every existing path is unchanged.
"""

from __future__ import annotations

import os

DAYTONA = "daytona"
SILO = "silo"
PROVIDERS = (DAYTONA, SILO)

_ADAPTER = {DAYTONA: "taskcompendium-daytona", SILO: "taskcompendium-silo"}
_ISOLATION = {DAYTONA: "daytona-network-block-all", SILO: "silo-netns-none"}

# Checkers accept either mechanism: both are network-blocked sandboxes whose
# isolation is proven (Daytona by configuration and receipts, silo by an
# in-sandbox egress probe with an outside positive control).  Evidence written
# by an earlier Daytona run therefore stays valid.
KNOWN_ADAPTERS = frozenset(_ADAPTER.values())
NETWORK_BLOCKED_ISOLATION = frozenset(_ISOLATION.values())


def provider() -> str:
    value = os.environ.get("CAPABILITY_SANDBOX_PROVIDER", DAYTONA)
    if value not in PROVIDERS:
        raise RuntimeError(f"CAPABILITY_SANDBOX_PROVIDER must be one of {PROVIDERS}, got {value!r}")
    return value


def adapter_id() -> str:
    return _ADAPTER[provider()]


def isolation_id() -> str:
    return _ISOLATION[provider()]


def credentials_present() -> bool:
    """True when the selected provider's credentials are in the environment."""
    if provider() == SILO:
        return bool(os.environ.get("SILO_API_TOKEN")) and bool(
            os.environ.get("SILO_BROKER_RESOLVE_URL") or os.environ.get("SILO_BROKER_URL")
        )
    return bool(os.environ.get("DAYTONA_API_KEY"))


def credentials_hint() -> str:
    if provider() == SILO:
        return "SILO_API_TOKEN and SILO_BROKER_RESOLVE_URL (or SILO_BROKER_URL)"
    return "DAYTONA_API_KEY"
