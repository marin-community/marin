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

"""Ecommerce domain tool handlers for the IPI resource server.

11 tools: order management, product catalog, refund processing, support tickets.
"""

import uuid
from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------------
# Order Management
# ---------------------------------------------------------------------------


def search_orders(env: Dict[str, Any], customer_id: str = None, status: str = None, **kwargs) -> List[Dict[str, Any]]:
    """Search orders by customer_id or status."""
    results = []
    for order in env.get("orders", {}).values():
        if customer_id and order.get("customer_id") != customer_id:
            continue
        if status and order.get("status", "").lower() != status.lower():
            continue
        results.append(order)
    return results


def get_order_details(env: Dict[str, Any], order_id: str) -> Dict[str, Any]:
    """Get full order details by order ID."""
    orders = env.get("orders", {})
    if order_id not in orders:
        return {"error": f"Order {order_id} not found."}
    return orders[order_id]


def update_order_status(env: Dict[str, Any], order_id: str, status: str) -> str:
    """Update the status of an order."""
    orders = env.get("orders", {})
    if order_id not in orders:
        return f"Error: Order {order_id} not found."
    orders[order_id]["status"] = status
    return f"Order {order_id} status updated to '{status}'."


def add_order_note(env: Dict[str, Any], order_id: str, note: str) -> str:
    """Add internal notes to an order."""
    orders = env.get("orders", {})
    if order_id not in orders:
        return f"Error: Order {order_id} not found."
    existing = orders[order_id].get("notes", "")
    orders[order_id]["notes"] = existing + "\n" + note if existing else note
    return f"Note added to order {order_id}."


# ---------------------------------------------------------------------------
# Product Catalog
# ---------------------------------------------------------------------------


def search_products(env: Dict[str, Any], query: str = None, category: str = None, **kwargs) -> List[Dict[str, Any]]:
    """Filter products by text in names/descriptions and by category."""
    results = []
    for product in env.get("products", {}).values():
        if category and product.get("category", "").lower() != category.lower():
            continue
        if query:
            query_lower = query.lower()
            name_match = query_lower in product.get("name", "").lower()
            desc_match = query_lower in product.get("description", "").lower()
            if not (name_match or desc_match):
                continue
        results.append(product)
    return results


def get_product_details(env: Dict[str, Any], product_id: str) -> Dict[str, Any]:
    """Get full product details by product ID."""
    products = env.get("products", {})
    if product_id not in products:
        return {"error": f"Product {product_id} not found."}
    return products[product_id]


def update_product_stock(env: Dict[str, Any], product_id: str, stock_quantity: int) -> str:
    """Update stock quantity for a product."""
    products = env.get("products", {})
    if product_id not in products:
        return f"Error: Product {product_id} not found."
    products[product_id]["stock_quantity"] = stock_quantity
    return f"Stock updated for product {product_id} to {stock_quantity}."


# ---------------------------------------------------------------------------
# Refund Processing
# ---------------------------------------------------------------------------


def process_refund(
    env: Dict[str, Any], refund_id: str, amount: float, reason: str = None, admin_notes: str = None
) -> str:
    """Process a refund request."""
    refunds = env.get("refunds", {})
    if refund_id in refunds:
        refunds[refund_id]["status"] = "processed"
        refunds[refund_id]["amount"] = amount
        if reason:
            refunds[refund_id]["reason"] = reason
        if admin_notes:
            refunds[refund_id]["admin_notes"] = admin_notes
    else:
        refund = {"refund_id": refund_id, "amount": amount, "status": "processed"}
        if reason:
            refund["reason"] = reason
        if admin_notes:
            refund["admin_notes"] = admin_notes
        env.setdefault("refunds", {})[refund_id] = refund
    return f"Refund {refund_id} processed for ${amount}."


# ---------------------------------------------------------------------------
# Support Tickets
# ---------------------------------------------------------------------------


def create_support_ticket(
    env: Dict[str, Any],
    customer_id: str,
    subject: str,
    description: str,
    order_id: str = None,
    priority: str = "medium",
) -> str:
    """Create a new support ticket."""
    ticket_id = f"T-{uuid.uuid4().hex[:6].upper()}"
    ticket = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "subject": subject,
        "description": description,
        "status": "open",
        "priority": priority,
    }
    if order_id:
        ticket["order_id"] = order_id
    env.setdefault("support_tickets", {})[ticket_id] = ticket
    return f"Support ticket {ticket_id} created."


def get_ticket_details(env: Dict[str, Any], ticket_id: str) -> Dict[str, Any]:
    """Get full support ticket details by ticket ID."""
    tickets = env.get("support_tickets", {})
    if ticket_id not in tickets:
        return {"error": f"Ticket {ticket_id} not found."}
    return tickets[ticket_id]


def update_ticket_status(env: Dict[str, Any], ticket_id: str, status: str, priority: str = None) -> str:
    """Update the status and priority of a support ticket."""
    tickets = env.get("support_tickets", {})
    if ticket_id not in tickets:
        return f"Error: Ticket {ticket_id} not found."
    tickets[ticket_id]["status"] = status
    if priority:
        tickets[ticket_id]["priority"] = priority
    return f"Ticket {ticket_id} updated to status '{status}'."


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS: Dict[str, Callable] = {
    "search_orders": search_orders,
    "get_order_details": get_order_details,
    "update_order_status": update_order_status,
    "add_order_note": add_order_note,
    "search_products": search_products,
    "get_product_details": get_product_details,
    "update_product_stock": update_product_stock,
    "process_refund": process_refund,
    "create_support_ticket": create_support_ticket,
    "get_ticket_details": get_ticket_details,
    "update_ticket_status": update_ticket_status,
}
