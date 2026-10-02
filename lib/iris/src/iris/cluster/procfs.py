# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse Linux process records shared by local and Kubernetes runtimes."""


def stat_fields_after_comm(raw: str) -> list[str]:
    """Return ``/proc/PID/stat`` fields starting at state (field 3).

    The parenthesized process name in field 2 may contain spaces and parentheses.
    """
    rclose = raw.rfind(")")
    return raw[rclose + 2 :].split() if rclose != -1 else []
