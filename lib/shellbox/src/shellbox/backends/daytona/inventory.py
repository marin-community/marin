# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the shared Daytona sandbox inventory for account operations."""

from daytona import AsyncDaytona, AsyncSandbox

INVENTORY_REQUEST_TIMEOUT = 30.0


async def inventory_sandboxes(client: AsyncDaytona) -> list[AsyncSandbox]:
    """Read every sandbox in the authenticated account across all pages."""
    return [sandbox async for sandbox in client.list(request_timeout=INVENTORY_REQUEST_TIMEOUT)]
