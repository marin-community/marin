# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned NVIDIA tool registry, with only its package import paths adapted."""

import json
from typing import Any

from taskcompendium.providers.nemo_workplace.vendor.workplace_assistant_tools.analytics import (
    AnalyticsTool,
    analytics_tool_schemas,
)
from taskcompendium.providers.nemo_workplace.vendor.workplace_assistant_tools.calendar import (
    CalendarTool,
    calendar_tool_schemas,
)
from taskcompendium.providers.nemo_workplace.vendor.workplace_assistant_tools.company_directory import (
    CompanyDirectoryTool,
    company_directory_tool_schemas,
)
from taskcompendium.providers.nemo_workplace.vendor.workplace_assistant_tools.customer_relationship_manager import (
    CustomerRelationshipManagerTool,
    customer_relationship_manager_tool_schemas,
)
from taskcompendium.providers.nemo_workplace.vendor.workplace_assistant_tools.email import EmailTool, email_tool_schemas
from taskcompendium.providers.nemo_workplace.vendor.workplace_assistant_tools.project_management import (
    ProjectManagementTool,
    project_management_tool_schemas,
)

TOOLKITS = ("email", "calendar", "analytics", "project_management", "customer_relationship_manager")
_CASE_SENSITIVE_COLUMNS = frozenset({"status", "list_name", "board"})


def get_tools() -> dict[str, Any]:
    """Construct the upstream 27-tool environment with a fresh immutable seed."""
    tool_env: dict[str, Any] = {"containers": {}, "functions": {}, "schemas": []}
    directory = CompanyDirectoryTool()
    tool_env["containers"]["company_directory"] = directory
    tool_env["functions"]["company_directory_find_email_address"] = directory.find_email_address
    tool_env["schemas"].extend(company_directory_tool_schemas)

    email = EmailTool()
    tool_env["containers"]["email"] = email
    tool_env["functions"].update(
        email_get_email_information_by_id=email.get_email_information_by_id,
        email_search_emails=email.search_emails,
        email_send_email=email.send_email,
        email_delete_email=email.delete_email,
        email_forward_email=email.forward_email,
        email_reply_email=email.reply_email,
    )
    tool_env["schemas"].extend(email_tool_schemas)

    calendar = CalendarTool()
    tool_env["containers"]["calendar"] = calendar
    tool_env["functions"].update(
        calendar_get_event_information_by_id=calendar.get_event_information_by_id,
        calendar_search_events=calendar.search_events,
        calendar_create_event=calendar.create_event,
        calendar_delete_event=calendar.delete_event,
        calendar_update_event=calendar.update_event,
    )
    tool_env["schemas"].extend(calendar_tool_schemas)

    analytics = AnalyticsTool()
    tool_env["containers"]["analytics"] = analytics
    tool_env["functions"].update(
        analytics_engaged_users_count=analytics.engaged_users_count,
        analytics_get_visitor_information_by_id=analytics.get_visitor_information_by_id,
        analytics_create_plot=analytics.create_plot,
        analytics_traffic_source_count=analytics.traffic_source_count,
        analytics_total_visits_count=analytics.total_visits_count,
        analytics_get_average_session_duration=analytics.get_average_session_duration,
    )
    tool_env["schemas"].extend(analytics_tool_schemas)

    projects = ProjectManagementTool()
    tool_env["containers"]["project_management"] = projects
    tool_env["functions"].update(
        project_management_get_task_information_by_id=projects.get_task_information_by_id,
        project_management_search_tasks=projects.search_tasks,
        project_management_create_task=projects.create_task,
        project_management_delete_task=projects.delete_task,
        project_management_update_task=projects.update_task,
    )
    tool_env["schemas"].extend(project_management_tool_schemas)

    crm = CustomerRelationshipManagerTool()
    tool_env["containers"]["customer_relationship_manager"] = crm
    tool_env["functions"].update(
        customer_relationship_manager_search_customers=crm.search_customers,
        customer_relationship_manager_update_customer=crm.update_customer,
        customer_relationship_manager_add_customer=crm.add_customer,
        customer_relationship_manager_delete_customer=crm.delete_customer,
    )
    tool_env["schemas"].extend(customer_relationship_manager_tool_schemas)
    return tool_env


def execute_actions(actions: list[dict[str, str]]) -> dict[str, Any]:
    """Replay source-format actions, retaining the source's error-and-continue behavior."""
    tool_env = get_tools()
    for action in actions:
        try:
            tool_env["functions"][action["name"]](**json.loads(action["arguments"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return tool_env


def source_state(tool_env: dict[str, Any]) -> dict[str, Any]:
    """Read exactly the mutable tables used by the upstream state comparator."""
    return {
        "calendar": tool_env["containers"]["calendar"]._calendar_events.copy(deep=True),
        "email": tool_env["containers"]["email"]._emails.copy(deep=True),
        "analytics": tool_env["containers"]["analytics"]._plots_data.copy(deep=True),
        "project_management": tool_env["containers"]["project_management"]._project_tasks.copy(deep=True),
        "customer_relationship_manager": (
            tool_env["containers"]["customer_relationship_manager"]._crm_data.copy(deep=True)
        ),
    }


def equivalent_state(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Apply the source's case-insensitive comparison before checking all mutable tables."""
    for name in first:
        left, right = first[name].copy(deep=True), second[name].copy(deep=True)
        for frame in (left, right):
            for column in frame.columns:
                if column not in _CASE_SENSITIVE_COLUMNS:
                    frame[column] = frame[column].str.lower()
        if not left.equals(right):
            return False
    return True
