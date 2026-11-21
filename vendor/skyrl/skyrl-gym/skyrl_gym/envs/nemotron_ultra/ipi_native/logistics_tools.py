# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logistics domain tool handlers for the IPI resource server.

12 tools: carrier communication, customs, delivery scheduling, shipment management, warehouse inventory.
"""

import uuid
from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------------
# Carrier Communication
# ---------------------------------------------------------------------------


def get_carrier_messages(env: Dict[str, Any], shipment_id: str = None, **kwargs) -> List[Dict[str, Any]]:
    """Retrieve recent carrier messages, optionally filtered by shipment."""
    messages = env.get("carrier_messages", {})
    if isinstance(messages, dict):
        result = list(messages.values())
    else:
        result = list(messages)
    if shipment_id:
        result = [m for m in result if m.get("shipment_id") == shipment_id]
    return result


def send_carrier_message(env: Dict[str, Any], to: str, subject: str, body: str) -> str:
    """Send a message to a carrier."""
    current_user = env.get("current_user", {})
    sender = current_user.get("email") or current_user.get("name", "system@logistics.example")
    msg_id = f"CM-{uuid.uuid4().hex[:6].upper()}"
    msg = {
        "message_id": msg_id,
        "from": sender,
        "to": to,
        "subject": subject,
        "body": body,
        "timestamp": "sent",
    }
    if isinstance(env.get("carrier_messages"), dict):
        env["carrier_messages"][msg_id] = msg
    else:
        env.setdefault("carrier_messages", []).append(msg)
    return f"Carrier message sent to {to}."


# ---------------------------------------------------------------------------
# Customs Processing
# ---------------------------------------------------------------------------


def get_customs_declaration_by_shipment(env: Dict[str, Any], shipment_id: str) -> Dict[str, Any]:
    """Fetch the customs declaration linked to a shipment."""
    for decl in env.get("customs_declarations", {}).values():
        if decl.get("shipment_id") == shipment_id:
            return decl
    return {"error": f"No customs declaration found for shipment {shipment_id}."}


def submit_customs_declaration(
    env: Dict[str, Any], shipment_id: str, hs_code: str, declared_value: float, additional_info: str = None, **kwargs
) -> Dict[str, Any]:
    """Create or update a customs declaration for a shipment."""
    # Check if declaration already exists for this shipment
    for decl in env.get("customs_declarations", {}).values():
        if decl.get("shipment_id") == shipment_id:
            decl["hs_code"] = hs_code
            decl["declared_value"] = declared_value
            if additional_info is not None:
                decl["additional_info"] = additional_info
            decl["status"] = "submitted"
            return decl

    decl_id = f"CD-{uuid.uuid4().hex[:6].upper()}"
    decl = {
        "declaration_id": decl_id,
        "shipment_id": shipment_id,
        "hs_code": hs_code,
        "declared_value": declared_value,
        "status": "submitted",
    }
    if additional_info is not None:
        decl["additional_info"] = additional_info
    env.setdefault("customs_declarations", {})[decl_id] = decl
    return decl


# ---------------------------------------------------------------------------
# Delivery Scheduling
# ---------------------------------------------------------------------------


def get_delivery_schedule(env: Dict[str, Any], shipment_id: str) -> Dict[str, Any]:
    """Get the delivery schedule for a shipment."""
    for schedule in env.get("delivery_schedules", {}).values():
        if schedule.get("shipment_id") == shipment_id:
            return schedule
    return {"error": f"No delivery schedule found for shipment {shipment_id}."}


# ---------------------------------------------------------------------------
# Shipment Management
# ---------------------------------------------------------------------------


def get_shipment_by_id(env: Dict[str, Any], shipment_id: str) -> Dict[str, Any]:
    """Retrieve a shipment record by its ID."""
    shipments = env.get("shipments", {})
    if shipment_id not in shipments:
        return {"error": f"Shipment {shipment_id} not found."}
    return shipments[shipment_id]


def search_shipments_by_status(env: Dict[str, Any], status: str) -> List[Dict[str, Any]]:
    """Search for shipments matching a given status."""
    status_lower = status.lower()
    results = []
    for shipment in env.get("shipments", {}).values():
        if shipment.get("status", "").lower() == status_lower:
            results.append(shipment)
    return results


def update_shipment_status(
    env: Dict[str, Any], shipment_id: str, status: str, eta: str = None, notes: str = None, **kwargs
) -> str:
    """Update the status, ETA, and notes of a shipment."""
    shipments = env.get("shipments", {})
    if shipment_id not in shipments:
        return f"Error: Shipment {shipment_id} not found."
    shipments[shipment_id]["status"] = status
    if eta is not None:
        shipments[shipment_id]["eta"] = eta
    if notes is not None:
        shipments[shipment_id]["notes"] = notes
    return f"Shipment {shipment_id} updated to status '{status}'."


# ---------------------------------------------------------------------------
# Warehouse Inventory
# ---------------------------------------------------------------------------


def get_warehouse_inventory_item(env: Dict[str, Any], inventory_id: str) -> Dict[str, Any]:
    """Get details of a specific inventory item by ID."""
    items = env.get("warehouse_inventories", {})
    if inventory_id not in items:
        return {"error": f"Inventory item {inventory_id} not found."}
    return items[inventory_id]


def list_low_stock_items(env: Dict[str, Any], threshold: int) -> List[Dict[str, Any]]:
    """Return inventory items where quantity is below a threshold."""
    results = []
    for item in env.get("warehouse_inventories", {}).values():
        if item.get("quantity", 0) < threshold:
            results.append(item)
    return results


def add_warehouse_inventory(
    env: Dict[str, Any],
    sku: str,
    description: str,
    quantity: int,
    warehouse_location: str,
    comments: str = None,
    **kwargs,
) -> str:
    """Create a new inventory record."""
    inv_id = f"INV-{uuid.uuid4().hex[:6].upper()}"
    item = {
        "inventory_id": inv_id,
        "sku": sku,
        "description": description,
        "quantity": quantity,
        "warehouse_location": warehouse_location,
    }
    if comments is not None:
        item["comments"] = comments
    env.setdefault("warehouse_inventories", {})[inv_id] = item
    return f"Inventory item {inv_id} added (SKU: {sku})."


def adjust_warehouse_quantity(env: Dict[str, Any], inventory_id: str, adjustment: int) -> str:
    """Adjust the quantity of an existing inventory item."""
    items = env.get("warehouse_inventories", {})
    if inventory_id not in items:
        return f"Error: Inventory item {inventory_id} not found."
    items[inventory_id]["quantity"] = items[inventory_id].get("quantity", 0) + adjustment
    return f"Inventory {inventory_id} quantity adjusted by {adjustment}."


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS: Dict[str, Callable] = {
    "get_carrier_messages": get_carrier_messages,
    "send_carrier_message": send_carrier_message,
    "get_customs_declaration_by_shipment": get_customs_declaration_by_shipment,
    "submit_customs_declaration": submit_customs_declaration,
    "get_delivery_schedule": get_delivery_schedule,
    "get_shipment_by_id": get_shipment_by_id,
    "search_shipments_by_status": search_shipments_by_status,
    "update_shipment_status": update_shipment_status,
    "get_warehouse_inventory_item": get_warehouse_inventory_item,
    "list_low_stock_items": list_low_stock_items,
    "add_warehouse_inventory": add_warehouse_inventory,
    "adjust_warehouse_quantity": adjust_warehouse_quantity,
}
