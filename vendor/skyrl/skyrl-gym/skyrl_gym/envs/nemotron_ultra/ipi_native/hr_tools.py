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
import uuid
from typing import Any, Callable, Dict, List


def search_candidates(env: Dict[str, Any], query: str, status: str = None) -> List[Dict[str, Any]]:
    """Search candidates by name, skills, or status."""
    query_lower = query.lower()
    results = []
    for candidate in env.get("candidates", {}).values():
        name_match = query_lower in candidate.get("name", "").lower()
        skill_match = query_lower in candidate.get("resume", "").lower()
        notes_match = query_lower in candidate.get("application_notes", "").lower()
        position_match = query_lower in candidate.get("position_applied", "").lower()
        if name_match or skill_match or notes_match or position_match:
            if status is None or candidate.get("status", "").lower() == status.lower():
                results.append({"candidate_id": candidate["candidate_id"], "name": candidate["name"]})
    return results


def get_candidate_profile(env: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    """Get full candidate profile."""
    candidates = env.get("candidates", {})
    if candidate_id not in candidates:
        return {"error": f"Candidate {candidate_id} not found."}
    return candidates[candidate_id]


def update_candidate_status(env: Dict[str, Any], candidate_id: str, status: str) -> str:
    """Update a candidate's application status."""
    candidates = env.get("candidates", {})
    if candidate_id not in candidates:
        return f"Error: Candidate {candidate_id} not found."
    candidates[candidate_id]["status"] = status
    return f"Status updated for candidate {candidate_id} to '{status}'."


def schedule_interview(env: Dict[str, Any], candidate_id: str, interviewer_email: str, date: str, time: str) -> str:
    """Schedule an interview for a candidate."""
    interview_id = f"I-{uuid.uuid4().hex[:6].upper()}"
    env.setdefault("interviews", {})[interview_id] = {
        "interview_id": interview_id,
        "candidate_id": candidate_id,
        "interviewer_email": interviewer_email,
        "date": date,
        "time": time,
        "status": "scheduled",
    }
    return f"Interview {interview_id} scheduled for candidate {candidate_id} on {date} at {time}."


def get_interview_schedule(env: Dict[str, Any], date: str = None) -> List[Dict[str, Any]]:
    """Get interview schedule, optionally filtered by date."""
    interviews = env.get("interviews", {})
    results = []
    for interview in interviews.values():
        if date is None or interview.get("date") == date:
            results.append(interview)
    return results


def submit_interview_feedback(env: Dict[str, Any], candidate_id: str, feedback: str, rating: int) -> str:
    """Submit interview feedback and rating for a candidate."""
    interviews = env.get("interviews", {})
    for interview in interviews.values():
        if interview.get("candidate_id") == candidate_id:
            interview["feedback"] = feedback
            interview["rating"] = rating
            return f"Feedback submitted for candidate {candidate_id}."
    # If no existing interview, record feedback anyway
    interview_id = f"I-{uuid.uuid4().hex[:6].upper()}"
    env.setdefault("interviews", {})[interview_id] = {
        "interview_id": interview_id,
        "candidate_id": candidate_id,
        "feedback": feedback,
        "rating": rating,
    }
    return f"Feedback submitted for candidate {candidate_id}."


def send_offer(env: Dict[str, Any], candidate_id: str, position: str, salary: str, start_date: str) -> str:
    """Send a job offer to a candidate."""
    env.setdefault("sent_actions", []).append(
        {
            "type": "offer",
            "candidate_id": candidate_id,
            "position": position,
            "salary": salary,
            "start_date": start_date,
        }
    )
    return f"Offer sent to candidate {candidate_id} for position '{position}'."


def send_email(env: Dict[str, Any], recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    msg = {
        "from": env.get("current_user", {}).get("email", "system@company.example"),
        "to": recipient,
        "subject": subject,
        "body": body,
        "timestamp": "sent",
    }
    # Handle both "messages" and "inbox" keys
    if "inbox" in env:
        env["inbox"].append(msg)
    else:
        env.setdefault("messages", []).append(msg)
    return f"Email sent to {recipient}."


def get_inbox(env: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get inbox messages."""
    return env.get("inbox", env.get("messages", []))


def get_open_positions(env: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get all open job positions."""
    return list(env.get("open_positions", {}).values())


def create_job_posting(env: Dict[str, Any], title: str, description: str, requirements: str) -> str:
    """Create a new job posting."""
    pos_id = f"POS-{uuid.uuid4().hex[:6].upper()}"
    env.setdefault("open_positions", {})[pos_id] = {
        "position_id": pos_id,
        "title": title,
        "description": description,
        "requirements": requirements,
        "status": "open",
    }
    return f"Job posting '{title}' created with ID {pos_id}."


TOOL_HANDLERS: Dict[str, Callable] = {
    "search_candidates": search_candidates,
    "get_candidate_profile": get_candidate_profile,
    "update_candidate_status": update_candidate_status,
    "schedule_interview": schedule_interview,
    "get_interview_schedule": get_interview_schedule,
    "submit_interview_feedback": submit_interview_feedback,
    "send_offer": send_offer,
    "send_email": send_email,
    "get_inbox": get_inbox,
    "get_open_positions": get_open_positions,
    "create_job_posting": create_job_posting,
}
