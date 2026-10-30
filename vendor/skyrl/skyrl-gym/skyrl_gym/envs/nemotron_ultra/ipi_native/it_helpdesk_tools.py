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

"""IT Helpdesk domain tool handlers for the IPI resource server.

12 tools: asset management, communication, knowledge base, ticket management.
"""

import uuid
from typing import Any, Callable, Dict, List

from skyrl_gym.envs.nemotron_ultra.ipi_native.common_tools import get_messages, send_message


# ---------------------------------------------------------------------------
# Asset Management
# ---------------------------------------------------------------------------


def search_assets(
    env: Dict[str, Any], tag: str = None, type: str = None, status: str = None, location: str = None, **kwargs
) -> List[Dict[str, Any]]:
    """Search assets by tag, type, status, or location."""
    results = []
    for asset in env.get("assets", {}).values():
        if tag and tag.lower() not in asset.get("tag", "").lower():
            continue
        if type and asset.get("type", "").lower() != type.lower():
            continue
        if status and asset.get("status", "").lower() != status.lower():
            continue
        if location and location.lower() not in asset.get("location", "").lower():
            continue
        results.append(asset)
    return results


def get_asset(env: Dict[str, Any], asset_id: str) -> Dict[str, Any]:
    """Retrieve a single asset by its asset_id."""
    assets = env.get("assets", {})
    if asset_id not in assets:
        return {"error": f"Asset {asset_id} not found."}
    return assets[asset_id]


def update_asset_notes(env: Dict[str, Any], asset_id: str, notes: str) -> str:
    """Update the notes field of an existing asset."""
    assets = env.get("assets", {})
    if asset_id not in assets:
        return f"Error: Asset {asset_id} not found."
    assets[asset_id]["notes"] = notes
    return f"Asset {asset_id} notes updated."


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------


def search_kb_articles(
    env: Dict[str, Any], keyword: str = None, category: str = None, **kwargs
) -> List[Dict[str, Any]]:
    """Search knowledge base articles by keyword or category."""
    results = []
    for article in env.get("knowledge_base_articles", {}).values():
        if category and article.get("category", "").lower() != category.lower():
            continue
        if keyword:
            kw_lower = keyword.lower()
            title_match = kw_lower in article.get("title", "").lower()
            content_match = kw_lower in article.get("content", "").lower()
            if not (title_match or content_match):
                continue
        results.append(article)
    return results


def get_kb_article(env: Dict[str, Any], article_id: str) -> Dict[str, Any]:
    """Retrieve a single knowledge base article by its article_id."""
    articles = env.get("knowledge_base_articles", {})
    if article_id not in articles:
        return {"error": f"Article {article_id} not found."}
    return articles[article_id]


def update_kb_article_content(env: Dict[str, Any], article_id: str, content: str) -> str:
    """Update the content field of an existing knowledge base article."""
    articles = env.get("knowledge_base_articles", {})
    if article_id not in articles:
        return f"Error: Article {article_id} not found."
    articles[article_id]["content"] = content
    return f"Article {article_id} content updated."


# ---------------------------------------------------------------------------
# Ticket Management
# ---------------------------------------------------------------------------


def search_incidents(
    env: Dict[str, Any], keyword: str = None, status: str = None, priority: str = None, **kwargs
) -> List[Dict[str, Any]]:
    """Search incidents by keyword, status, or priority."""
    results = []
    for incident in env.get("incidents", {}).values():
        if status and incident.get("status", "").lower() != status.lower():
            continue
        if priority and incident.get("priority", "").lower() != priority.lower():
            continue
        if keyword:
            kw_lower = keyword.lower()
            title_match = kw_lower in incident.get("title", "").lower()
            desc_match = kw_lower in incident.get("description", "").lower()
            if not (title_match or desc_match):
                continue
        results.append(incident)
    return results


def get_incident(env: Dict[str, Any], ticket_id: str) -> Dict[str, Any]:
    """Retrieve a single incident by its ticket_id."""
    incidents = env.get("incidents", {})
    if ticket_id not in incidents:
        return {"error": f"Incident {ticket_id} not found."}
    return incidents[ticket_id]


def update_incident_description(env: Dict[str, Any], ticket_id: str, description: str) -> str:
    """Update the description field of an existing incident."""
    incidents = env.get("incidents", {})
    if ticket_id not in incidents:
        return f"Error: Incident {ticket_id} not found."
    incidents[ticket_id]["description"] = description
    return f"Incident {ticket_id} description updated."


def create_incident(env: Dict[str, Any], title: str, description: str, priority: str, assigned_to: str) -> str:
    """Create a new incident ticket."""
    ticket_id = f"INC-{uuid.uuid4().hex[:6].upper()}"
    env.setdefault("incidents", {})[ticket_id] = {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "status": "Open",
        "priority": priority,
        "assigned_to": assigned_to,
    }
    return f"Incident {ticket_id} created."


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS: Dict[str, Callable] = {
    "search_assets": search_assets,
    "get_asset": get_asset,
    "update_asset_notes": update_asset_notes,
    "send_message": send_message,
    "get_messages": get_messages,
    "search_kb_articles": search_kb_articles,
    "get_kb_article": get_kb_article,
    "update_kb_article_content": update_kb_article_content,
    "search_incidents": search_incidents,
    "get_incident": get_incident,
    "update_incident_description": update_incident_description,
    "create_incident": create_incident,
}
