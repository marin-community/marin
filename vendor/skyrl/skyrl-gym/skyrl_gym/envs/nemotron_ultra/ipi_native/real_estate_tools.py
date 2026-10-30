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

"""Real Estate domain tool handlers for the IPI resource server.

12 tools: client CRM, communications, inspections, property listings.
"""

import uuid
from typing import Any, Callable, Dict, List

from skyrl_gym.envs.nemotron_ultra.ipi_native.common_tools import get_messages, send_message


# ---------------------------------------------------------------------------
# Client CRM
# ---------------------------------------------------------------------------


def list_clients(env: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
    """Return a list of clients."""
    return list(env.get("clients", {}).values())


def get_client_details(env: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """Get full client record by client_id."""
    clients = env.get("clients", {})
    if client_id not in clients:
        return {"error": f"Client {client_id} not found."}
    return clients[client_id]


def add_client_note(env: Dict[str, Any], client_id: str, notes: str) -> str:
    """Append or replace the notes field for a client record."""
    clients = env.get("clients", {})
    if client_id not in clients:
        return f"Error: Client {client_id} not found."
    clients[client_id]["notes"] = notes
    return f"Notes updated for client {client_id}."


# ---------------------------------------------------------------------------
# Inspections
# ---------------------------------------------------------------------------


def schedule_inspection(env: Dict[str, Any], property_id: str, inspector_name: str, inspection_date: str) -> str:
    """Create a new inspection report shell for a property."""
    report_id = f"IR-{uuid.uuid4().hex[:6].upper()}"
    env.setdefault("inspection_reports", {})[report_id] = {
        "report_id": report_id,
        "property_id": property_id,
        "inspector_name": inspector_name,
        "inspection_date": inspection_date,
        "findings": "",
        "recommendations": "",
        "overall_condition": "",
    }
    return f"Inspection {report_id} scheduled for property {property_id}."


def get_inspection_report(env: Dict[str, Any], report_id: str) -> Dict[str, Any]:
    """Fetch an inspection report by report_id."""
    reports = env.get("inspection_reports", {})
    if report_id not in reports:
        return {"error": f"Inspection report {report_id} not found."}
    return reports[report_id]


def submit_inspection_findings(env: Dict[str, Any], report_id: str, findings: str, recommendations: str) -> str:
    """Update the findings and recommendations fields of an existing inspection report."""
    reports = env.get("inspection_reports", {})
    if report_id not in reports:
        return f"Error: Inspection report {report_id} not found."
    reports[report_id]["findings"] = findings
    reports[report_id]["recommendations"] = recommendations
    return f"Findings submitted for inspection report {report_id}."


# ---------------------------------------------------------------------------
# Property Listings
# ---------------------------------------------------------------------------


def search_properties(
    env: Dict[str, Any],
    keywords: str = None,
    min_price: float = None,
    max_price: float = None,
    location: str = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Search property listings by keywords, price range, or location."""
    results = []
    for prop in env.get("property_listings", {}).values():
        if min_price is not None and prop.get("price", 0) < min_price:
            continue
        if max_price is not None and prop.get("price", 0) > max_price:
            continue
        if location and location.lower() not in prop.get("address", "").lower():
            continue
        if keywords:
            kw_lower = keywords.lower()
            addr_match = kw_lower in prop.get("address", "").lower()
            desc_match = kw_lower in prop.get("description", "").lower()
            if not (addr_match or desc_match):
                continue
        results.append(prop)
    return results


def get_property_details(env: Dict[str, Any], property_id: str) -> Dict[str, Any]:
    """Retrieve full details of a property by property_id."""
    properties = env.get("property_listings", {})
    if property_id not in properties:
        return {"error": f"Property {property_id} not found."}
    return properties[property_id]


def create_property(
    env: Dict[str, Any],
    address: str,
    price: float,
    bedrooms: int,
    bathrooms: int,
    description: str = None,
    status: str = "active",
    listing_date: str = None,
    **kwargs,
) -> str:
    """Create a new property listing."""
    prop_id = f"PROP-{uuid.uuid4().hex[:6].upper()}"
    prop = {
        "property_id": prop_id,
        "address": address,
        "price": price,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "status": status,
    }
    if description:
        prop["description"] = description
    if listing_date:
        prop["listing_date"] = listing_date
    env.setdefault("property_listings", {})[prop_id] = prop
    return f"Property {prop_id} created at {address}."


def update_property_description(env: Dict[str, Any], property_id: str, description: str) -> str:
    """Update the description field of an existing property listing."""
    properties = env.get("property_listings", {})
    if property_id not in properties:
        return f"Error: Property {property_id} not found."
    properties[property_id]["description"] = description
    return f"Property {property_id} description updated."


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS: Dict[str, Callable] = {
    "list_clients": list_clients,
    "get_client_details": get_client_details,
    "add_client_note": add_client_note,
    "get_messages": get_messages,
    "send_message": send_message,
    "schedule_inspection": schedule_inspection,
    "get_inspection_report": get_inspection_report,
    "submit_inspection_findings": submit_inspection_findings,
    "search_properties": search_properties,
    "get_property_details": get_property_details,
    "create_property": create_property,
    "update_property_description": update_property_description,
}
