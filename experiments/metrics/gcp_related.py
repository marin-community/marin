import json
import logging
import os.path
from dataclasses import dataclass
from datetime import datetime, timedelta

import fsspec

logger = logging.getLogger("ray")


@dataclass
class GCP_API_CONFIG:
    output_path: str
    PROJECT_ID: str = "hai-gcp-models"
    TIME_SINCE: str = (datetime.now() - timedelta(days=7)).isoformat("T") + "Z"  # Format in RFC3339


@dataclass
class NUM_RESTART_CONFIG(GCP_API_CONFIG):
    instance_substr: str = "ray-marin-us-central2-head"


def get_number_of_restarts(config: NUM_RESTART_CONFIG) -> int:
    """Get the number of restarts for a specific instance substring"""
    # Initialize the logging client
    from google.cloud import logging as gcp_logging

    client = gcp_logging.Client()

    # Log filter to get both creation and deletion events
    log_filter = f"""
        resource.type="gce_instance"
        (
            protoPayload.methodName="v1.compute.instances.insert"
        )
        timestamp >= "{config.TIME_SINCE}"
        severity=NOTICE
        protoPayload.resourceName:"{config.instance_substr}"
    """
    # Run the query to get logs matching the filter
    entries = client.list_entries(filter_=log_filter, order_by=gcp_logging.DESCENDING)

    # Process and return the relevant log entries
    events = {}
    for entry in entries:
        # Parse the log entry to get relevant details
        event = {
            "instance_id": entry.payload.get("resourceName", ""),
            "action": entry.payload.get("methodName").split(".")[-1],  # "insert" or "delete"
            "timestamp": entry.timestamp.isoformat(),
            "zone": entry.resource.labels.get("zone", ""),
            "user": entry.payload.get("authenticationInfo", {}).get("principalEmail", "unknown"),
        }
        events[event["instance_id"]] = event

    events = list(events.values())
    # Display the recent instance creation and deletion events
    if events:
        logging.info("Instance creation and deletion events in the past week:")
        for event in events:
            logging.info(
                f"Instance ID: {event['instance_id']}, Action: {event['action']}, "
                f"Timestamp: {event['timestamp']}, Zone: {event['zone']}, User: {event['user']}"
            )
    else:
        logging.error("No instance creation or deletion events in the past week.")

    with fsspec.open(os.path.join(config.output_path, "metric.json"), "w") as f:
        print(json.dumps({"Number of Ray cluster restart": len(events), "Ray restart events": events}), file=f)

    return len(events)
