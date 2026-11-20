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

"""Insurance domain tool handlers for the IPI resource server.

12 tools: adjuster operations, claims processing, communications, fraud, payouts, policy management.
"""

import uuid
from typing import Any, Callable, Dict, List

from skyrl_gym.envs.nemotron_ultra.ipi_native.common_tools import send_message


# ---------------------------------------------------------------------------
# Adjuster Operations
# ---------------------------------------------------------------------------


def add_adjuster_note(env: Dict[str, Any], claim_id: str, adjuster_id: str, note_text: str) -> Dict[str, Any]:
    """Add a field note from an adjuster to a claim."""
    note_id = f"AN-{uuid.uuid4().hex[:6].upper()}"
    note = {
        "note_id": note_id,
        "claim_id": claim_id,
        "adjuster_id": adjuster_id,
        "note_text": note_text,
        "timestamp": "created",
    }
    env.setdefault("adjuster_notes", {})[note_id] = note
    return note


def get_adjuster_notes_for_claim(env: Dict[str, Any], claim_id: str) -> List[Dict[str, Any]]:
    """Retrieve all adjuster notes for a given claim."""
    results = []
    for note in env.get("adjuster_notes", {}).values():
        if note.get("claim_id") == claim_id:
            results.append(note)
    return results


# ---------------------------------------------------------------------------
# Claims Processing
# ---------------------------------------------------------------------------


def create_claim(
    env: Dict[str, Any],
    policy_id: str,
    claim_number: str,
    date_of_loss: str,
    description: str,
    claim_type: str,
    amount_requested: float,
    adjuster_id: str,
) -> Dict[str, Any]:
    """Create a new claim linked to a policy."""
    claim_id = f"CL-{uuid.uuid4().hex[:6].upper()}"
    claim = {
        "claim_id": claim_id,
        "policy_id": policy_id,
        "claim_number": claim_number,
        "date_of_loss": date_of_loss,
        "description": description,
        "claim_type": claim_type,
        "amount_requested": amount_requested,
        "status": "open",
        "adjuster_id": adjuster_id,
    }
    env.setdefault("claims", {})[claim_id] = claim
    return claim


def get_claim_by_id(env: Dict[str, Any], claim_id: str) -> Dict[str, Any]:
    """Fetch a claim by its claim_id."""
    claims = env.get("claims", {})
    if claim_id not in claims:
        return {"error": f"Claim {claim_id} not found."}
    return claims[claim_id]


def update_claim_description(env: Dict[str, Any], claim_id: str, description: str) -> str:
    """Modify the description field of an existing claim."""
    claims = env.get("claims", {})
    if claim_id not in claims:
        return f"Error: Claim {claim_id} not found."
    claims[claim_id]["description"] = description
    return f"Claim {claim_id} description updated."


# ---------------------------------------------------------------------------
# Communications
# ---------------------------------------------------------------------------


def get_messages_for_claim(env: Dict[str, Any], claim_id: str) -> List[Dict[str, Any]]:
    """Get all messages linked to a specific claim."""
    messages = env.get("messages", [])
    if isinstance(messages, dict):
        messages = list(messages.values())
    return [m for m in messages if m.get("related_claim_id") == claim_id or m.get("claim_id") == claim_id]


# ---------------------------------------------------------------------------
# Fraud Investigation
# ---------------------------------------------------------------------------


def flag_fraud_investigation(
    env: Dict[str, Any], claim_id: str, investigator_id: str, flag_reason: str, severity: str
) -> Dict[str, Any]:
    """Create a fraud investigation flag on a claim."""
    flag_id = f"FF-{uuid.uuid4().hex[:6].upper()}"
    flag = {
        "flag_id": flag_id,
        "claim_id": claim_id,
        "investigator_id": investigator_id,
        "flag_reason": flag_reason,
        "severity": severity,
        "status": "open",
    }
    env.setdefault("fraud_investigation_flags", {})[flag_id] = flag
    return flag


# ---------------------------------------------------------------------------
# Payouts
# ---------------------------------------------------------------------------


def process_payout(
    env: Dict[str, Any], claim_id: str, amount: float, date: str, method: str, notes: str
) -> Dict[str, Any]:
    """Record a payout for a claim."""
    payout_id = f"PO-{uuid.uuid4().hex[:6].upper()}"
    payout = {
        "payout_id": payout_id,
        "claim_id": claim_id,
        "amount": amount,
        "date": date,
        "method": method,
        "status": "processed",
        "notes": notes,
    }
    env.setdefault("payouts", {})[payout_id] = payout
    return payout


# ---------------------------------------------------------------------------
# Policy Management
# ---------------------------------------------------------------------------


def get_policy_by_id(env: Dict[str, Any], policy_id: str) -> Dict[str, Any]:
    """Retrieve a policy record by its policy_id."""
    policies = env.get("policies", {})
    if policy_id not in policies:
        return {"error": f"Policy {policy_id} not found."}
    return policies[policy_id]


def search_policies_by_holder(env: Dict[str, Any], policyholder_name: str) -> List[Dict[str, Any]]:
    """Search for policies by policyholder name."""
    name_lower = policyholder_name.lower()
    results = []
    for policy in env.get("policies", {}).values():
        if name_lower in policy.get("policyholder_name", "").lower():
            results.append(policy)
    return results


def update_policy_notes(env: Dict[str, Any], policy_id: str, notes: str) -> str:
    """Update the notes field of a policy record."""
    policies = env.get("policies", {})
    if policy_id not in policies:
        return f"Error: Policy {policy_id} not found."
    policies[policy_id]["notes"] = notes
    return f"Policy {policy_id} notes updated."


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS: Dict[str, Callable] = {
    "add_adjuster_note": add_adjuster_note,
    "get_adjuster_notes_for_claim": get_adjuster_notes_for_claim,
    "create_claim": create_claim,
    "get_claim_by_id": get_claim_by_id,
    "update_claim_description": update_claim_description,
    "send_message": send_message,
    "get_messages_for_claim": get_messages_for_claim,
    "flag_fraud_investigation": flag_fraud_investigation,
    "process_payout": process_payout,
    "get_policy_by_id": get_policy_by_id,
    "search_policies_by_holder": search_policies_by_holder,
    "update_policy_notes": update_policy_notes,
}
