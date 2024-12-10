import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger("ray")


@dataclass
class GcpApiConfig:
    project_id: str = "hai-gcp-models"
    time_since: str = (datetime.now() - timedelta(days=7)).isoformat("T") + "Z"  # Format in RFC3339


@dataclass
class NumRestartConfig(GcpApiConfig):
    instance_substr: str = "ray-marin-us-central2-head"


def get_gcp_restart_events(config: NumRestartConfig) -> list[dict]:
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
        timestamp >= "{config.time_since}"
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

    return events
