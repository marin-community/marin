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


METRICS = ["total_visits", "session_duration_seconds", "user_engaged"]
METRIC_NAMES = ["total visits", "average session duration", "engaged users"]


class AnalyticsTool:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        """
        Resets the analytics data to the original state.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "csv_data", "processed", "analytics_data.csv")
        self._analytics_data = pd.read_csv(data_path, dtype=str)
        self._analytics_data["user_engaged"] = self._analytics_data["user_engaged"] == "True"  # Convert to boolean
        self._plots_data = pd.DataFrame(columns=["file_path"])

    def get_visitor_information_by_id(self, visitor_id=None):
        if not visitor_id:
            return "Visitor ID not provided."
        visitor_data = self._analytics_data[self._analytics_data["visitor_id"] == visitor_id].to_dict(orient="records")
        if visitor_data:
            return visitor_data
        else:
            return "Visitor not found."

    def create_plot(self, time_min=None, time_max=None, value_to_plot=None, plot_type=None):
        if not time_min:
            return "Start date not provided."
        if not time_max:
            return "End date not provided."
        if value_to_plot not in [
            "total_visits",
            "session_duration_seconds",
            "user_engaged",
            "visits_direct",
            "visits_referral",
            "visits_search_engine",
            "visits_social_media",
        ]:
            return "Value to plot must be one of 'total_visits', 'session_duration_seconds', 'user_engaged', 'visits_direct', 'visits_referral', 'visits_search_engine', 'visits_social_media'"
        if plot_type not in ["bar", "line", "scatter", "histogram"]:
            return "Plot type must be one of 'bar', 'line', 'scatter', or 'histogram'"

        # Plot the data here and save it to a file
        file_path = f"plots/{time_min}_{time_max}_{value_to_plot}_{plot_type}.png"
        self._plots_data.loc[len(self._plots_data)] = [file_path]
        return file_path

    def total_visits_count(self, time_min=None, time_max=None):
        if time_min:
            data = self._analytics_data[self._analytics_data["date_of_visit"] >= time_min]
        else:
            data = self._analytics_data
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        return data.groupby("date_of_visit").size().to_dict()

    def engaged_users_count(self, time_min=None, time_max=None):
        if time_min:
            data = self._analytics_data[self._analytics_data["date_of_visit"] >= time_min]
        else:
            data = self._analytics_data[:]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        data["user_engaged"] = data["user_engaged"].astype(bool).astype(int)

        return data.groupby("date_of_visit").sum()["user_engaged"].to_dict()

    def traffic_source_count(self, time_min=None, time_max=None, traffic_source=None):
        if time_min:
            data = self._analytics_data[self._analytics_data["date_of_visit"] >= time_min]
        else:
            data = self._analytics_data[:]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]

        if traffic_source:
            data["visits_from_source"] = (data["traffic_source"] == traffic_source).astype(int)
            return data.groupby("date_of_visit").sum()["visits_from_source"].to_dict()
        else:
            return data.groupby("date_of_visit").size().to_dict()

    def get_average_session_duration(self, time_min=None, time_max=None):
        if time_min:
            data = self._analytics_data[self._analytics_data["date_of_visit"] >= time_min]
        else:
            data = self._analytics_data
        if time_max:
            data = data[data["date_of_visit"] <= time_max]

        data["session_duration_seconds"] = data["session_duration_seconds"].astype(float)
        return (
            data[["date_of_visit", "session_duration_seconds"]]
            .groupby("date_of_visit")
            .mean()["session_duration_seconds"]
            .to_dict()
        )


schema_get_visitor_information = {
    "type": "function",
    "name": "analytics_get_visitor_information_by_id",
    "description": "Returns the analytics data for a given visitor ID.",
    "parameters": {
        "type": "object",
        "properties": {"visitor_id": {"type": "string", "description": "ID of the visitor"}},
        "required": ["visitor_id"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_create_plot = {
    "type": "function",
    "name": "analytics_create_plot",
    "description": "Plots the analytics data for a given time range and value.",
    "parameters": {
        "type": "object",
        "properties": {
            "time_min": {
                "type": "string",
                "description": "Start date of the time range. Date format is YYYY-MM-DD",
            },
            "time_max": {
                "type": "string",
                "description": "End date of the time range. Date format is YYYY-MM-DD",
            },
            "value_to_plot": {
                "type": "string",
                "description": "Value to plot. Available values are: 'total_visits', 'session_duration_seconds', 'user_engaged', 'visits_direct', 'visits_referral', 'visits_search_engine', 'visits_social_media'",
            },
            "plot_type": {
                "type": "string",
                "description": "Type of plot. Can be 'bar', 'line', 'scatter' or 'histogram'",
            },
        },
        "required": ["time_min", "time_max", "value_to_plot", "plot_type"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_total_visits_count = {
    "type": "function",
    "name": "analytics_total_visits_count",
    "description": "Returns the total number of visits within a specified time range.",
    "parameters": {
        "type": "object",
        "properties": {
            "time_min": {
                "type": "string",
                "description": "Start date of the time range. Date format is YYYY-MM-DD",
            },
            "time_max": {
                "type": "string",
                "description": "End date of the time range. Date format is YYYY-MM-DD",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_engaged_users_count = {
    "type": "function",
    "name": "analytics_engaged_users_count",
    "description": "Returns the number of engaged users within a specified time range.",
    "parameters": {
        "type": "object",
        "properties": {
            "time_min": {
                "type": "string",
                "description": "Start date of the time range. Date format is YYYY-MM-DD",
            },
            "time_max": {
                "type": "string",
                "description": "End date of the time range. Date format is YYYY-MM-DD",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_traffic_source_count = {
    "type": "function",
    "name": "analytics_traffic_source_count",
    "description": "Returns the number of visits from a specific traffic source within a specified time range.",
    "parameters": {
        "type": "object",
        "properties": {
            "time_min": {
                "type": "string",
                "description": "Start date of the time range. Date format is YYYY-MM-DD",
            },
            "time_max": {
                "type": "string",
                "description": "End date of the time range. Date format is YYYY-MM-DD",
            },
            "traffic_source": {
                "type": "string",
                "description": "Traffic source to filter the visits. Available values are: 'direct', 'referral', 'search engine', 'social media'",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_get_average_session_duration = {
    "type": "function",
    "name": "analytics_get_average_session_duration",
    "description": "Returns the average session duration within a specified time range.",
    "parameters": {
        "type": "object",
        "properties": {
            "time_min": {
                "type": "string",
                "description": "Start date of the time range. Date format is YYYY-MM-DD",
            },
            "time_max": {
                "type": "string",
                "description": "End date of the time range. Date format is YYYY-MM-DD",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}

analytics_tool_schemas = [
    schema_get_visitor_information,
    schema_create_plot,
    schema_total_visits_count,
    schema_engaged_users_count,
    schema_traffic_source_count,
    schema_get_average_session_duration,
]
