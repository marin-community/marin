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


class CustomerRelationshipManagerTool:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(
            current_dir,
            "..",
            "csv_data",
            "processed",
            "customer_relationship_manager_data.csv",
        )
        self._crm_data = pd.read_csv(data_path, dtype=str)

    def search_customers(
        self,
        customer_name=None,
        customer_email=None,
        product_interest=None,
        status=None,
        assigned_to_email=None,
        last_contact_date_min=None,
        last_contact_date_max=None,
        follow_up_by_min=None,
        follow_up_by_max=None,
        page=1,
        page_size=5,
    ):
        customers = self._crm_data.copy()
        if not any(
            [
                customer_name,
                customer_email,
                product_interest,
                status,
                assigned_to_email,
                last_contact_date_min,
                last_contact_date_max,
                follow_up_by_min,
                follow_up_by_max,
            ]
        ):
            return "No search parameters provided. Please provide at least one parameter."

        if customer_name:
            customers = customers[customers["customer_name"].str.contains(customer_name, case=False)]
        if customer_email:
            customers = customers[customers["customer_email"].str.contains(customer_email, case=False)]
        if product_interest:
            customers = customers[customers["product_interest"].str.contains(product_interest, case=False)]
        if status:
            customers = customers[customers["status"].str.contains(status, case=False)]
        if assigned_to_email:
            customers = customers[customers["assigned_to_email"].str.contains(assigned_to_email, case=False)]
        if last_contact_date_min:
            customers = customers[customers["last_contact_date"] >= last_contact_date_min]
        if last_contact_date_max:
            customers = customers[customers["last_contact_date"] <= last_contact_date_max]
        if follow_up_by_min:
            customers = customers[customers["follow_up_by"] >= follow_up_by_min]
        if follow_up_by_max:
            customers = customers[customers["follow_up_by"] <= follow_up_by_max]

        # Convert to records for pagination
        customer_records = customers.to_dict(orient="records")

        # Calculate pagination
        total_customers = len(customer_records)
        total_pages = (total_customers + page_size - 1) // page_size  # Ceiling division

        # Validate page number
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1

        # Calculate start and end indices for slicing
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_customers)

        if total_customers > 0:
            return {
                "customers": customer_records[start_idx:end_idx],
                "pagination": {
                    "total_customers": total_customers,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": total_pages,
                },
            }
        else:
            return "No customers found."

    def update_customer(self, customer_id=None, field=None, new_value=None):
        if not customer_id or not field or not new_value:
            return "Customer ID, field, or new value not provided."

        if field == "status" and new_value not in [
            "Qualified",
            "Won",
            "Lost",
            "Lead",
            "Proposal",
        ]:
            return "Status not valid. Please choose from: 'Qualified', 'Won', 'Lost', 'Lead', 'Proposal'"

        if field == "product_interest" and new_value not in [
            "Software",
            "Hardware",
            "Services",
            "Consulting",
            "Training",
        ]:
            return "Product interest not valid. Please choose from: 'Software', 'Hardware', 'Services', 'Consulting', 'Training'"

        if field in ["customer_email", "assigned_to_email"]:
            new_value = new_value.lower()

        if customer_id in self._crm_data["customer_id"].values:
            if field in self._crm_data.columns:
                self._crm_data.loc[self._crm_data["customer_id"] == customer_id, field] = new_value
                return "Customer updated successfully."
            else:
                return "Field not valid. Please choose from: 'customer_name', 'assigned_to_email', 'customer_email', 'customer_phone', 'last_contact_date', 'product_interest', 'status', 'notes', 'follow_up_by'"
        return "Customer not found."

    def add_customer(
        self,
        customer_name=None,
        assigned_to_email=None,
        status=None,
        customer_email=None,
        customer_phone=None,
        last_contact_date=None,
        product_interest=None,
        notes="",
        follow_up_by=None,
    ):
        if not all([customer_name, assigned_to_email, status]):
            return "Please provide all required fields: customer_name, assigned_to_email, status."

        assigned_to_email = assigned_to_email.lower()
        if customer_email:
            customer_email = customer_email.lower()

        new_id = str(int(self._crm_data["customer_id"].max()) + 1).zfill(8)
        new_customer = pd.DataFrame(
            {
                "customer_id": [new_id],
                "customer_name": [customer_name],
                "customer_email": [customer_email],
                "customer_phone": [customer_phone],
                "last_contact_date": [last_contact_date],
                "product_interest": [product_interest],
                "status": [status],
                "assigned_to_email": [assigned_to_email],
                "notes": [notes],
                "follow_up_by": [follow_up_by],
            }
        )
        self._crm_data = pd.concat([self._crm_data, new_customer], ignore_index=True)
        return new_id

    def delete_customer(self, customer_id=None):
        if not customer_id:
            return "Customer ID not provided."
        if customer_id not in self._crm_data["customer_id"].values:
            return "Customer not found."
        self._crm_data = self._crm_data[self._crm_data["customer_id"] != customer_id]
        return "Customer deleted successfully."


schema_search_customers = {
    "type": "function",
    "name": "customer_relationship_manager_search_customers",
    "description": "Searches for customers based on the given parameters with pagination support.",
    "parameters": {
        "type": "object",
        "properties": {
            "customer_name": {"type": "string", "description": "Name of the customer"},
            "customer_email": {
                "type": "string",
                "description": "Email address of the customer",
            },
            "product_interest": {
                "type": "string",
                "description": "Product interest of the customer",
            },
            "status": {
                "type": "string",
                "description": "Current status of the customer",
            },
            "assigned_to_email": {
                "type": "string",
                "description": "Email address of the person assigned to the customer",
            },
            "last_contact_date_min": {
                "type": "string",
                "description": "Minimum last contact date. Format: YYYY-MM-DD",
            },
            "last_contact_date_max": {
                "type": "string",
                "description": "Maximum last contact date. Format: YYYY-MM-DD",
            },
            "follow_up_by_min": {
                "type": "string",
                "description": "Minimum follow up date. Format: YYYY-MM-DD",
            },
            "follow_up_by_max": {
                "type": "string",
                "description": "Maximum follow up date. Format: YYYY-MM-DD",
            },
            "page": {
                "type": "integer",
                "description": "Page number of results to return",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of customers per page",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_update_customer = {
    "type": "function",
    "name": "customer_relationship_manager_update_customer",
    "description": "Updates a customer record by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "ID of the customer"},
            "field": {
                "type": "string",
                "description": "Field to update. Available fields are: 'customer_name', 'assigned_to_email', 'customer_email', 'customer_phone', 'last_contact_date', 'product_interest', 'status', 'notes', 'follow_up_by'",
            },
            "new_value": {"type": "string", "description": "New value for the field"},
        },
        "required": ["customer_id", "field", "new_value"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_add_customer = {
    "type": "function",
    "name": "customer_relationship_manager_add_customer",
    "description": "Adds a new customer record.",
    "parameters": {
        "type": "object",
        "properties": {
            "customer_name": {"type": "string", "description": "Name of the customer"},
            "assigned_to_email": {
                "type": "string",
                "description": "Email address of the person assigned to the customer",
            },
            "status": {
                "type": "string",
                "description": "Current status of the customer. One of: 'Qualified', 'Won', 'Lost', 'Lead', 'Proposal'",
            },
            "customer_email": {
                "type": "string",
                "description": "Email address of the customer",
            },
            "customer_phone": {
                "type": "string",
                "description": "Phone number of the customer",
            },
            "last_contact_date": {
                "type": "string",
                "description": "The last date the customer was contacted. Format: YYYY-MM-DD",
            },
            "product_interest": {
                "type": "string",
                "description": "Product interest of the customer. One of: 'Software', 'Hardware', 'Services', 'Consulting', 'Training'",
            },
            "notes": {"type": "string", "description": "Notes about the customer"},
            "follow_up_by": {
                "type": "string",
                "description": "Date for the next follow up. Format: YYYY-MM-DD",
            },
        },
        "required": [
            "customer_name",
            "assigned_to_email",
            "status",
        ],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_delete_customer = {
    "type": "function",
    "name": "customer_relationship_manager_delete_customer",
    "description": "Deletes a customer record by ID.",
    "parameters": {
        "type": "object",
        "properties": {"customer_id": {"type": "string", "description": "ID of the customer"}},
        "required": ["customer_id"],
        "additionalProperties": False,
    },
    "strict": False,
}

customer_relationship_manager_tool_schemas = [
    schema_search_customers,
    schema_update_customer,
    schema_add_customer,
    schema_delete_customer,
]
