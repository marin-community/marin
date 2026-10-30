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

from skyrl_gym.envs.nemotron_ultra.ipi_native.common_tools import get_messages, send_message


def search_patients(env: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """Search patients by name or condition."""
    query_lower = query.lower()
    results = []
    for patient in env["patients"].values():
        name_match = query_lower in patient.get("name", "").lower()
        condition_match = any(query_lower in c.lower() for c in patient.get("conditions", []))
        if name_match or condition_match:
            results.append({"patient_id": patient["patient_id"], "name": patient["name"]})
    return results


def get_patient_record(env: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
    """Get full patient record by patient ID."""
    if patient_id not in env["patients"]:
        return {"error": f"Patient {patient_id} not found."}
    return env["patients"][patient_id]


def update_patient_chart(env: Dict[str, Any], patient_id: str, notes: str) -> str:
    """Update clinical notes for a patient chart."""
    if patient_id not in env["patients"]:
        return f"Error: Patient {patient_id} not found."
    existing = env["patients"][patient_id].get("chart_notes", "")
    env["patients"][patient_id]["chart_notes"] = existing + "\n" + notes if existing else notes
    return f"Chart updated for patient {patient_id}."


def get_schedule(env: Dict[str, Any], date: str) -> List[Dict[str, Any]]:
    """Get the appointment schedule for a given date."""
    results = []
    for appt in env["appointments"].values():
        if appt.get("date") == date:
            results.append(appt)
    return results


def schedule_appointment(env: Dict[str, Any], patient_id: str, date: str, time: str, reason: str) -> str:
    """Schedule a new appointment."""
    appt_id = f"A{uuid.uuid4().hex[:6].upper()}"
    env["appointments"][appt_id] = {
        "appointment_id": appt_id,
        "patient_id": patient_id,
        "date": date,
        "time": time,
        "reason": reason,
        "status": "scheduled",
    }
    return f"Appointment {appt_id} scheduled for patient {patient_id} on {date} at {time}."


def cancel_appointment(env: Dict[str, Any], appointment_id: str) -> str:
    """Cancel an appointment by ID."""
    if appointment_id not in env["appointments"]:
        return f"Error: Appointment {appointment_id} not found."
    env["appointments"][appointment_id]["status"] = "cancelled"
    return f"Appointment {appointment_id} has been cancelled."


def send_referral(env: Dict[str, Any], patient_id: str, specialist_email: str, reason: str) -> str:
    """Send a referral for a patient to a specialist."""
    if "sent_actions" not in env:
        env["sent_actions"] = []
    env["sent_actions"].append(
        {"type": "referral", "patient_id": patient_id, "specialist_email": specialist_email, "reason": reason}
    )
    return f"Referral sent for patient {patient_id} to {specialist_email}."


def prescribe_medication(env: Dict[str, Any], patient_id: str, medication: str, dosage: str, instructions: str) -> str:
    """Prescribe medication to a patient."""
    if patient_id not in env.get("prescriptions", {}):
        env.setdefault("prescriptions", {})[patient_id] = []
    env["prescriptions"][patient_id].append({"medication": medication, "dosage": dosage, "instructions": instructions})
    return f"Prescribed {medication} {dosage} for patient {patient_id}."


def get_prescriptions(env: Dict[str, Any], patient_id: str) -> List[Dict[str, Any]]:
    """Get prescriptions for a patient."""
    return env.get("prescriptions", {}).get(patient_id, [])


def get_lab_results(env: Dict[str, Any], patient_id: str) -> List[Dict[str, Any]]:
    """Get lab results for a patient."""
    return env.get("lab_results", {}).get(patient_id, [])


TOOL_HANDLERS: Dict[str, Callable] = {
    "search_patients": search_patients,
    "get_patient_record": get_patient_record,
    "update_patient_chart": update_patient_chart,
    "get_schedule": get_schedule,
    "schedule_appointment": schedule_appointment,
    "cancel_appointment": cancel_appointment,
    "send_referral": send_referral,
    "send_message": send_message,
    "get_messages": get_messages,
    "prescribe_medication": prescribe_medication,
    "get_prescriptions": get_prescriptions,
    "get_lab_results": get_lab_results,
}
