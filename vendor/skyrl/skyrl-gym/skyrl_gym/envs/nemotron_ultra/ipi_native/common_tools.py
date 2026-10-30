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

"""Shared tool implementations for messaging across multiple IPI domains.

send_message and get_messages have identical semantics across healthcare,
education, insurance, IT helpdesk, legal, and real estate domains but differ
in parameter names. This module provides a single adaptive implementation.
"""

import uuid
from typing import Any, Dict, List, Union


def send_message(env: Dict[str, Any], **kwargs) -> str:
    """Send a message. Works across all domain env shapes.

    Accepts flexible kwargs:
      - recipient (healthcare)
      - to (education, insurance, IT helpdesk, legal)
      - recipients (real estate, may be a list)
      - from_ (education, insurance, IT helpdesk, legal) or inferred from current_user
      - subject, body
      - case_id, related_claim_id (optional domain-specific fields)
    """
    current_user = env.get("current_user", {})
    sender = kwargs.get("from_") or current_user.get("email") or current_user.get("name", "system@example.com")

    # Normalize recipient: different domains use different kwarg names
    recipient = kwargs.get("recipient") or kwargs.get("to") or kwargs.get("recipients")
    if isinstance(recipient, list):
        recipient_str = ", ".join(str(r) for r in recipient)
    else:
        recipient_str = str(recipient) if recipient else "unknown"

    msg: Dict[str, Any] = {
        "from": sender,
        "to": recipient_str,
        "subject": kwargs.get("subject", ""),
        "body": kwargs.get("body", ""),
        "timestamp": "sent",
    }

    # Preserve optional domain-specific fields
    if "case_id" in kwargs:
        msg["case_id"] = kwargs["case_id"]
    if "related_claim_id" in kwargs:
        msg["related_claim_id"] = kwargs["related_claim_id"]

    messages = env.get("messages")
    if isinstance(messages, list):
        messages.append(msg)
    elif isinstance(messages, dict):
        msg_id = f"MSG-{uuid.uuid4().hex[:6].upper()}"
        msg["message_id"] = msg_id
        messages[msg_id] = msg
    else:
        env.setdefault("messages", []).append(msg)

    return f"Message sent to {recipient_str}."


def get_messages(env: Dict[str, Any], **kwargs) -> Union[List[Dict[str, Any]], Dict]:
    """Get messages. Works across all domain env shapes.

    Supports optional filters used by the legal domain:
      - case_id, from_, to, is_read
    """
    messages = env.get("messages", [])
    if isinstance(messages, dict):
        result = list(messages.values())
    else:
        result = list(messages)

    # Apply optional filters (legal domain)
    if kwargs.get("case_id"):
        result = [m for m in result if m.get("case_id") == kwargs["case_id"]]
    if kwargs.get("from_"):
        result = [m for m in result if m.get("from", "").lower() == kwargs["from_"].lower()]
    if kwargs.get("to"):
        result = [m for m in result if m.get("to", "").lower() == kwargs["to"].lower()]
    if kwargs.get("is_read") is not None:
        result = [m for m in result if m.get("is_read") == kwargs["is_read"]]

    return result
