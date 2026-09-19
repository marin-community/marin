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
import os

import pandas as pd


HARDCODED_CURRENT_TIME = pd.to_datetime("2023-11-30T23:59:00")


class EmailTool:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "csv_data", "processed", "emails.csv")
        self._emails = pd.read_csv(data_path, dtype=str)

    def get_email_information_by_id(self, email_id=None, field=None):
        if not email_id:
            return "Email ID not provided."
        if not field:
            return "Field not provided."

        try:
            email = self._emails[self._emails["email_id"] == email_id].to_dict(orient="records")
            if email:
                if field in email[0]:
                    return {field: email[0][field]}
                else:
                    return f"Field '{field}' not found. Available fields: {', '.join(email[0].keys())}"
            else:
                return "Email not found."
        except Exception as e:
            return f"Error retrieving email information: {e}"

    def search_emails(self, query="", date_min=None, date_max=None, page=1, page_size=5):
        try:
            query_words = query.lower().split()

            # Filter function to check if all query words are in any of the specified fields
            def filter_emails(row):
                combined_fields = f"{row['subject']} {row['body']} {row['sender/recipient']}".lower()
                return all(word in combined_fields for word in query_words)

            # Apply filter function across all rows
            filtered_emails = self._emails.apply(filter_emails, axis=1)
            emails = self._emails[filtered_emails].copy()
            emails["sent_datetime_tmp"] = pd.to_datetime(emails["sent_datetime"])
            emails = emails.sort_values("sent_datetime_tmp", ascending=False)
            emails.drop(columns=["sent_datetime_tmp"], inplace=True)
            emails = emails.to_dict(orient="records")

            if date_min:
                emails = [
                    email
                    for email in emails
                    if pd.Timestamp(email["sent_datetime"]).date() >= pd.Timestamp(date_min).date()
                ]
            if date_max:
                # inclusive, remove time from timestamp
                emails = [
                    email
                    for email in emails
                    if pd.Timestamp(email["sent_datetime"]).date() <= pd.Timestamp(date_max).date()
                ]

            # Calculate pagination
            total_emails = len(emails)
            total_pages = (total_emails + page_size - 1) // page_size  # Ceiling division

            # Validate page number
            page = max(1, min(page, total_pages)) if total_pages > 0 else 1

            # Calculate start and end indices for slicing
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_emails)

            if len(emails):
                return {
                    "emails": emails[start_idx:end_idx],
                    "pagination": {
                        "total_emails": total_emails,
                        "page": page,
                        "page_size": page_size,
                        "total_pages": total_pages,
                    },
                }
            else:
                return "No emails found."
        except Exception as e:
            return f"Error searching emails: {e}"

    def send_email(self, recipient=None, subject=None, body=None):
        if not recipient or not subject or not body:
            return "Recipient, subject, or body not provided."
        if "@" not in recipient or "." not in recipient:
            return "Invalid recipient email address."

        try:
            recipient = recipient.lower()
            email_id = str(int(self._emails["email_id"].max()) + 1) if not self._emails.empty else "1"
            sent_datetime = str(HARDCODED_CURRENT_TIME)

            # Create a new row with named columns instead of positional insertion
            new_email = {
                "email_id": email_id,
                "inbox/outbox": "outbox",
                "sender/recipient": recipient,
                "subject": subject,
                "sent_datetime": sent_datetime,
                "body": body,
            }

            self._emails = pd.concat([self._emails, pd.DataFrame([new_email])], ignore_index=True)
            return "Email sent successfully."
        except Exception as e:
            return f"Error sending email: {e}"

    def delete_email(self, email_id=None):
        if not email_id:
            return "Email ID not provided."

        try:
            if email_id in self._emails["email_id"].values:
                self._emails = self._emails[self._emails["email_id"] != email_id]
                return "Email deleted successfully."
            else:
                return "Email not found."
        except Exception as e:
            return f"Error deleting email: {e}"

    def forward_email(self, email_id=None, recipient=None):
        if not email_id or not recipient:
            return "Email ID or recipient not provided."
        if "@" not in recipient or "." not in recipient:
            return "Invalid recipient email address."

        try:
            if email_id not in self._emails["email_id"].values:
                return "Email not found."

            recipient = recipient.lower()
            email = self._emails[self._emails["email_id"] == email_id].to_dict(orient="records")[0]
            result = self.send_email(recipient, f"FW: {email['subject']}", email["body"])

            if "Error" in result or "Invalid" in result:
                return result
            return "Email forwarded successfully."
        except Exception as e:
            return f"Error forwarding email: {e}"

    def reply_email(self, email_id=None, body=None):
        if not email_id or not body:
            return "Email ID or body not provided."

        try:
            if email_id not in self._emails["email_id"].values:
                return "Email not found."

            email = self._emails[self._emails["email_id"] == email_id].to_dict(orient="records")[0]

            # Handle reply differently based on whether it's an inbox or outbox email
            recipient = email["sender/recipient"]
            if email["inbox/outbox"] == "outbox":
                # For outbox emails, reply to the person we sent to originally
                pass  # recipient is already set correctly

            result = self.send_email(recipient, f"RE: {email['subject']}", body)

            if "Error" in result or "Invalid" in result:
                return result
            return "Email replied successfully."
        except Exception as e:
            return f"Error replying to email: {e}"


schema_get_email_information_by_id = {
    "type": "function",
    "name": "email_get_email_information_by_id",
    "description": "Retrieves specific details of an email by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "email_id": {"type": "string", "description": "Unique ID of the email"},
            "field": {
                "type": "string",
                "description": "Specific field to return. Available fields: 'email_id', 'inbox/outbox', 'sender/recipient', 'subject', 'sent_datetime', 'body'",
            },
        },
        "required": ["email_id", "field"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_search_emails = {
    "type": "function",
    "name": "email_search_emails",
    "description": "Searches for emails matching the given query across subject, body, or sender fields. The function matches an email if all words in the query appear in any of these fields.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, matching terms in subject, body, or sender/recipient fields",
            },
            "date_min": {
                "type": "string",
                "description": "Lower date limit for the email's sent date (inclusive). Format: YYYY-MM-DD",
            },
            "date_max": {
                "type": "string",
                "description": "Upper date limit for the email's sent date (inclusive). Format: YYYY-MM-DD",
            },
            "page": {
                "type": "integer",
                "description": "Page number of results to return",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of emails per page",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_send_email = {
    "type": "function",
    "name": "email_send_email",
    "description": "Sends an email to the specified recipient.",
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Email address of the recipient",
            },
            "subject": {"type": "string", "description": "Subject line of the email"},
            "body": {"type": "string", "description": "Body content of the email"},
        },
        "required": ["recipient", "subject", "body"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_delete_email = {
    "type": "function",
    "name": "email_delete_email",
    "description": "Deletes an email by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "email_id": {
                "type": "string",
                "description": "Unique ID of the email to be deleted",
            }
        },
        "required": ["email_id"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_forward_email = {
    "type": "function",
    "name": "email_forward_email",
    "description": "Forwards an email to the specified recipient.",
    "parameters": {
        "type": "object",
        "properties": {
            "email_id": {
                "type": "string",
                "description": "Unique ID of the email to be forwarded",
            },
            "recipient": {
                "type": "string",
                "description": "Email address of the recipient",
            },
        },
        "required": ["email_id", "recipient"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_reply_email = {
    "type": "function",
    "name": "email_reply_email",
    "description": "Replies to an email by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "email_id": {
                "type": "string",
                "description": "Unique ID of the email to be replied",
            },
            "body": {"type": "string", "description": "Body content of the email"},
        },
        "required": ["email_id", "body"],
        "additionalProperties": False,
    },
    "strict": False,
}

# List of all email tool schemas
email_tool_schemas = [
    schema_get_email_information_by_id,
    schema_search_emails,
    schema_send_email,
    schema_delete_email,
    schema_forward_email,
    schema_reply_email,
]
