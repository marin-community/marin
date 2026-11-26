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

"""Legal domain tool handlers for the IPI resource server.

12 tools: billing, case management, communication, court & hearings, documents.
"""

import uuid
from typing import Any, Callable, Dict, List

from skyrl_gym.envs.nemotron_ultra.ipi_native.common_tools import get_messages, send_message


# ---------------------------------------------------------------------------
# Billing
# ---------------------------------------------------------------------------


def add_billing_entry(
    env: Dict[str, Any],
    case_id: str,
    date: str,
    description: str,
    hours: float,
    rate: float,
    billable: bool = True,
    attorney: str = None,
    **kwargs,
) -> str:
    """Add a time or expense billing entry for a case."""
    billing_id = f"BL-{uuid.uuid4().hex[:6].upper()}"
    entry = {
        "billing_id": billing_id,
        "case_id": case_id,
        "date": date,
        "description": description,
        "hours": hours,
        "rate": rate,
        "amount": hours * rate,
        "billable": billable,
    }
    if attorney:
        entry["attorney"] = attorney
    env.setdefault("billing_records", {})[billing_id] = entry
    return f"Billing entry {billing_id} added for case {case_id}."


def get_billing_report(
    env: Dict[str, Any],
    case_id: str = None,
    start_date: str = None,
    end_date: str = None,
    billable: bool = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Get billing entries for a case or date range."""
    results = []
    for entry in env.get("billing_records", {}).values():
        if case_id and entry.get("case_id") != case_id:
            continue
        if start_date and entry.get("date", "") < start_date:
            continue
        if end_date and entry.get("date", "") > end_date:
            continue
        if billable is not None and entry.get("billable") != billable:
            continue
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Case Management
# ---------------------------------------------------------------------------


def search_cases(
    env: Dict[str, Any], client_name: str = None, case_type: str = None, status: str = None, **kwargs
) -> List[Dict[str, Any]]:
    """Search case files by client name, case type, or status."""
    results = []
    for case in env.get("case_files", {}).values():
        if client_name and client_name.lower() not in case.get("client_name", "").lower():
            continue
        if case_type and case_type.lower() != case.get("case_type", "").lower():
            continue
        if status and status.lower() != case.get("status", "").lower():
            continue
        results.append(case)
    return results


def get_case_file(env: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    """Get full case file details by case ID."""
    cases = env.get("case_files", {})
    if case_id not in cases:
        return {"error": f"Case {case_id} not found."}
    return cases[case_id]


def update_case_notes(env: Dict[str, Any], case_id: str, notes: str) -> str:
    """Update internal notes field in a case file."""
    cases = env.get("case_files", {})
    if case_id not in cases:
        return f"Error: Case {case_id} not found."
    cases[case_id]["notes"] = notes
    return f"Case {case_id} notes updated."


# ---------------------------------------------------------------------------
# Court & Hearings
# ---------------------------------------------------------------------------


def get_court_schedule(env: Dict[str, Any], schedule_id: str) -> Dict[str, Any]:
    """Get court schedule details by schedule ID."""
    schedules = env.get("court_schedules", {})
    if schedule_id not in schedules:
        return {"error": f"Schedule {schedule_id} not found."}
    return schedules[schedule_id]


def search_schedules(
    env: Dict[str, Any],
    case_id: str = None,
    start_date: str = None,
    end_date: str = None,
    hearing_type: str = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Search court schedules by case ID, date range, or hearing type."""
    results = []
    for schedule in env.get("court_schedules", {}).values():
        if case_id and schedule.get("case_id") != case_id:
            continue
        if start_date and schedule.get("scheduled_date", "") < start_date:
            continue
        if end_date and schedule.get("scheduled_date", "") > end_date:
            continue
        if hearing_type and hearing_type.lower() != schedule.get("hearing_type", "").lower():
            continue
        results.append(schedule)
    return results


def update_schedule_location(env: Dict[str, Any], schedule_id: str, location: str = None, notes: str = None) -> str:
    """Update location or notes for a court schedule."""
    schedules = env.get("court_schedules", {})
    if schedule_id not in schedules:
        return f"Error: Schedule {schedule_id} not found."
    if location is not None:
        schedules[schedule_id]["location"] = location
    if notes is not None:
        schedules[schedule_id]["notes"] = notes
    return f"Schedule {schedule_id} updated."


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def create_document_filing(
    env: Dict[str, Any],
    case_id: str,
    document_type: str,
    title: str,
    filing_date: str,
    court_name: str,
    file_path: str,
    served_parties: List[str] = None,
    notes: str = None,
    **kwargs,
) -> str:
    """Create a new document filing record."""
    filing_id = f"DF-{uuid.uuid4().hex[:6].upper()}"
    filing = {
        "filing_id": filing_id,
        "case_id": case_id,
        "document_type": document_type,
        "title": title,
        "filing_date": filing_date,
        "court_name": court_name,
        "file_path": file_path,
        "served_parties": served_parties or [],
    }
    if notes:
        filing["notes"] = notes
    env.setdefault("document_filings", {})[filing_id] = filing
    return f"Document filing {filing_id} created for case {case_id}."


def get_document_filing(env: Dict[str, Any], filing_id: str) -> Dict[str, Any]:
    """Get document filing details by filing ID."""
    filings = env.get("document_filings", {})
    if filing_id not in filings:
        return {"error": f"Filing {filing_id} not found."}
    return filings[filing_id]


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS: Dict[str, Callable] = {
    "add_billing_entry": add_billing_entry,
    "get_billing_report": get_billing_report,
    "search_cases": search_cases,
    "get_case_file": get_case_file,
    "update_case_notes": update_case_notes,
    "send_message": send_message,
    "get_messages": get_messages,
    "get_court_schedule": get_court_schedule,
    "search_schedules": search_schedules,
    "update_schedule_location": update_schedule_location,
    "create_document_filing": create_document_filing,
    "get_document_filing": get_document_filing,
}
