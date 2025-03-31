import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

# from google.api_core import exceptions
# from google.cloud import bigquery
# from google.cloud import billing as gcp_billing

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


def get_gcp_spend(config: GcpApiConfig) -> dict:
    """Get the GCP spend for the last week"""
    try:
        # Initialize clients
        bq_client = bigquery.Client(project=config.project_id)
        billing_client = gcp_billing.CloudBillingClient()

        # Get billing account
        try:
            billing_accounts = billing_client.list_billing_accounts()
            billing_account_id = next(billing_accounts).name.split("/")[-1]
            logger.info(f"Using billing account: {billing_account_id}")
        except StopIteration:
            logger.error("No billing accounts found")
            raise ValueError("No billing accounts found") from None
        except exceptions.PermissionDenied:
            logger.error("Insufficient permissions")
            raise ValueError("Insufficient permissions") from None

        # Query billing data
        query = f"""
            SELECT
                service.description AS service,
                sku.description AS sku,
                project.id AS project_id,
                SUM(cost) AS total_cost,
                usage_start_time
            FROM `{config.project_id}.{config.billing_dataset}.{config.table_name}`
            WHERE usage_start_time >= TIMESTAMP("{config.time_since}")
            GROUP BY service, sku, project_id, usage_start_time
            ORDER BY total_cost DESC
        """
        results = bq_client.query(query).result()

        # Parse results
        budgets = {
            "Total Spend": 0.0,
            "Services": {},
            "Clusters": {"us-central2": 0.0, "eu-west4": 0.0, "us-west1": 0.0},
        }

        for row in results:
            service, sku = row.service, row.sku
            total_cost = float(row.total_cost)

            # Service costs
            if service not in budgets["Services"]:
                budgets["Services"][service] = {"total": 0.0, "skus": {}}
            budgets["Services"][service]["total"] += total_cost
            budgets["Services"][service]["skus"][sku] = budgets["Services"][service]["skus"].get(sku, 0.0) + total_cost

            # Cluster costs
            for cluster in budgets["Clusters"]:
                if cluster in sku.lower() or cluster in service.lower():
                    budgets["Clusters"][cluster] += total_cost

            budgets["Total Spend"] += total_cost

        logger.info(f"Total spend: ${budgets['Total Spend']:.2f}")
        for cluster, spend in budgets["Clusters"].items():
            logger.info(f"Cluster {cluster} spend: ${spend:.2f}")

        return budgets

    except exceptions.GoogleAPIError as e:
        logger.error(f"Failed to query billing data: {e!s}")
        raise RuntimeError(f"Failed to query billing data: {e!s}") from None


def get_weekly_spend(billing_dataset: str, table_name: str) -> dict:
    config = GcpApiConfig(billing_dataset=billing_dataset, table_name=table_name)
    return get_gcp_spend(config)
