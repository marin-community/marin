# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable delivery of signed Grafana alert webhooks through Cloud Tasks."""

import dataclasses
import os
from collections.abc import Mapping
from typing import Protocol

import google.cloud.tasks_v2 as tasks_v2

SIGNATURE_HEADER = "X-Grafana-Alerting-Signature"
TIMESTAMP_HEADER = "X-Grafana-Alerting-Signature-Timestamp"
ENQUEUE_TIMEOUT = 10.0


class AlertQueue(Protocol):
    """Queue one raw, signed Grafana webhook for the ops ingest service."""

    def enqueue(self, *, body: bytes, headers: Mapping[str, str]) -> str: ...


@dataclasses.dataclass(frozen=True)
class CloudTasksAlertQueue:
    """Cloud Tasks queue whose deliveries authenticate as the ops dispatcher."""

    client: tasks_v2.CloudTasksClient
    parent: str
    target_url: str
    target_audience: str
    dispatch_service_account: str

    @staticmethod
    def from_environment() -> "CloudTasksAlertQueue | None":
        """Build the production queue, or disable the relay when no queue env is set."""

        keys = {
            "project": "OPS_ALERT_QUEUE_PROJECT",
            "location": "OPS_ALERT_QUEUE_LOCATION",
            "queue": "OPS_ALERT_QUEUE",
            "target_url": "OPS_ALERT_TARGET_URL",
            "target_audience": "OPS_ALERT_TARGET_AUDIENCE",
            "dispatch_service_account": "OPS_ALERT_DISPATCH_SERVICE_ACCOUNT",
        }
        values = {name: os.environ.get(env) for name, env in keys.items()}
        if not any(value is not None for value in values.values()):
            return None
        missing = [keys[name] for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"incomplete ops alert queue environment; missing {', '.join(sorted(missing))}")
        client = tasks_v2.CloudTasksClient()
        return CloudTasksAlertQueue(
            client=client,
            parent=client.queue_path(
                os.environ[keys["project"]],
                os.environ[keys["location"]],
                os.environ[keys["queue"]],
            ),
            target_url=os.environ[keys["target_url"]],
            target_audience=os.environ[keys["target_audience"]],
            dispatch_service_account=os.environ[keys["dispatch_service_account"]],
        )

    def enqueue(self, *, body: bytes, headers: Mapping[str, str]) -> str:
        task = tasks_v2.Task(
            http_request=tasks_v2.HttpRequest(
                http_method=tasks_v2.HttpMethod.POST,
                url=self.target_url,
                headers=dict(headers),
                body=body,
                oidc_token=tasks_v2.OidcToken(
                    service_account_email=self.dispatch_service_account,
                    audience=self.target_audience,
                ),
            )
        )
        return self.client.create_task(parent=self.parent, task=task, timeout=ENQUEUE_TIMEOUT).name
