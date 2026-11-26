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

"""Education domain tool handlers for the IPI resource server.

12 tools: communication, course management, financial aid, grading, student records.
"""

import uuid
from typing import Any, Callable, Dict, List

from skyrl_gym.envs.nemotron_ultra.ipi_native.common_tools import get_messages, send_message


# ---------------------------------------------------------------------------
# Course Management
# ---------------------------------------------------------------------------


def list_courses(env: Dict[str, Any], department: str = None, **kwargs) -> List[Dict[str, Any]]:
    """List all courses, optionally filtered by department."""
    results = []
    for course in env.get("courses", {}).values():
        if department and course.get("department", "").lower() != department.lower():
            continue
        results.append(course)
    return results


def get_course_detail(env: Dict[str, Any], course_id: str) -> Dict[str, Any]:
    """Get detailed information for a specific course."""
    courses = env.get("courses", {})
    if course_id not in courses:
        return {"error": f"Course {course_id} not found."}
    return courses[course_id]


def create_course(
    env: Dict[str, Any],
    course_code: str,
    title: str,
    description: str,
    credits: int,
    department: str,
    instructor_id: str,
    schedule: str,
    **kwargs,
) -> str:
    """Create a new course offering."""
    course_id = f"C-{uuid.uuid4().hex[:6].upper()}"
    env.setdefault("courses", {})[course_id] = {
        "course_id": course_id,
        "course_code": course_code,
        "title": title,
        "description": description,
        "credits": credits,
        "department": department,
        "instructor_id": instructor_id,
        "schedule": schedule,
    }
    return f"Course '{title}' ({course_code}) created with ID {course_id}."


# ---------------------------------------------------------------------------
# Financial Aid
# ---------------------------------------------------------------------------


def get_financial_aid(env: Dict[str, Any], student_id: str) -> List[Dict[str, Any]]:
    """Get financial aid records for a student."""
    results = []
    for aid in env.get("financial_aid", {}).values():
        if aid.get("student_id") == student_id:
            results.append(aid)
    return results


def adjust_financial_aid(
    env: Dict[str, Any], aid_id: str, amount: float = None, status: str = None, reviewer_notes: str = None, **kwargs
) -> Dict[str, Any]:
    """Modify an existing financial aid award."""
    aid_records = env.get("financial_aid", {})
    if aid_id not in aid_records:
        return {"error": f"Financial aid record {aid_id} not found."}
    record = aid_records[aid_id]
    if amount is not None:
        record["amount"] = amount
    if status is not None:
        record["status"] = status
    if reviewer_notes is not None:
        record["reviewer_notes"] = reviewer_notes
    return record


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def get_grades_for_student(env: Dict[str, Any], student_id: str) -> List[Dict[str, Any]]:
    """Retrieve all grade entries for a student."""
    results = []
    for grade in env.get("grades", {}).values():
        if grade.get("student_id") == student_id:
            results.append(grade)
    return results


def submit_grade(
    env: Dict[str, Any],
    student_id: str,
    course_id: str,
    assignment_name: str,
    score: float,
    max_score: float,
    grader_comments: str = "",
    **kwargs,
) -> str:
    """Create a grade record for a student and course assignment."""
    grade_id = f"G-{uuid.uuid4().hex[:6].upper()}"
    env.setdefault("grades", {})[grade_id] = {
        "grade_id": grade_id,
        "student_id": student_id,
        "course_id": course_id,
        "assignment_name": assignment_name,
        "score": score,
        "max_score": max_score,
        "grader_comments": grader_comments,
    }
    return f"Grade submitted for student {student_id} in course {course_id}."


# ---------------------------------------------------------------------------
# Student Records
# ---------------------------------------------------------------------------


def search_students(
    env: Dict[str, Any], name: str = None, major: str = None, email: str = None, **kwargs
) -> List[Dict[str, Any]]:
    """Search for students by name, major, or email."""
    results = []
    for student in env.get("students", {}).values():
        if name:
            full_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".lower()
            if name.lower() not in full_name:
                continue
        if major and major.lower() not in student.get("major", "").lower():
            continue
        if email and email.lower() != student.get("email", "").lower():
            continue
        results.append(student)
    return results


def get_student_record(env: Dict[str, Any], student_id: str) -> Dict[str, Any]:
    """Retrieve a full student record by student_id."""
    students = env.get("students", {})
    if student_id not in students:
        return {"error": f"Student {student_id} not found."}
    return students[student_id]


def update_student_advisor_notes(env: Dict[str, Any], student_id: str, notes: str) -> str:
    """Update the advisor_notes field for a student record."""
    students = env.get("students", {})
    if student_id not in students:
        return f"Error: Student {student_id} not found."
    existing = students[student_id].get("advisor_notes", "")
    students[student_id]["advisor_notes"] = existing + "\n" + notes if existing else notes
    return f"Advisor notes updated for student {student_id}."


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS: Dict[str, Callable] = {
    "get_messages": get_messages,
    "send_message": send_message,
    "list_courses": list_courses,
    "get_course_detail": get_course_detail,
    "create_course": create_course,
    "get_financial_aid": get_financial_aid,
    "adjust_financial_aid": adjust_financial_aid,
    "get_grades_for_student": get_grades_for_student,
    "submit_grade": submit_grade,
    "search_students": search_students,
    "get_student_record": get_student_record,
    "update_student_advisor_notes": update_student_advisor_notes,
}
