"""Module for tracking GCP spending metrics."""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

from google.cloud import billing
from google.cloud import bigquery
from google.cloud import storage

logger = logging.getLogger(__name__)


@dataclass
class GcpSpendingConfig:
    """Configuration for GCP spending metrics."""
    project_id: str
    dataset_id: str
    billing_table_id: str
    time_since: str = (datetime.now() - timedelta(days=7)).isoformat("T") + "Z"  # Format in RFC3339
    billing_account: str = None  # Will be fetched if not provided


def get_billing_account(project_id: str) -> str:
    """Get the billing account associated with the project."""
    client = billing.CloudBillingClient()
    project_name = f"projects/{project_id}"
    
    try:
        billing_info = client.get_project_billing_info(name=project_name)
        return billing_info.billing_account_name
    except Exception as e:
        logger.error(f"Error getting billing account: {e}")
        return None


def get_storage_costs(config: GcpSpendingConfig) -> Dict[str, Dict]:
    """Get storage costs by bucket and operation type."""
    storage_client = storage.Client(project=config.project_id)
    costs = {}

    # Convert iterator to list so we can iterate multiple times
    buckets = list(storage_client.list_buckets())

    print("List of buckets:")
    print([bucket.name for bucket in buckets])

    BUCKETS_OF_INTEREST = ['marin-data', 'marin-eu-west4', 'marin-us-central2', 'marin-us-east1', 'marin-us-east5', 'marin-us-west4']

    try:
        # Only process buckets of interest
        for bucket in [b for b in buckets if b.name in BUCKETS_OF_INTEREST]:
            bucket = storage_client.get_bucket(bucket.name)
            costs[bucket.name] = {
                "class": bucket.storage_class,
                "location": bucket.location,
            }
    except Exception as e:
        logger.error(f"Error getting storage costs: {e}")
    
    return costs


def get_billing_details(config: GcpSpendingConfig) -> List[Dict]:
    """Get detailed billing information for the project using BigQuery."""
    client = bigquery.Client(project=config.project_id)
    
    # Get costs for the last 30 days by default
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"Getting costs from {start_time} to {end_time}")
    
    query = f"""
    SELECT
        service.description AS service,
        location.region AS region,
        sku.description AS sku,
        SUM(cost) AS total_cost,
        SUM(usage.amount) AS total_usage,
        usage.unit AS usage_unit,
        currency
    FROM
        `{config.project_id}.{config.dataset_id}.{config.billing_table_id}`
    WHERE
        usage_start_time BETWEEN TIMESTAMP('{start_time.strftime('%Y-%m-%d')}') 
        AND TIMESTAMP('{end_time.strftime('%Y-%m-%d')}')
    GROUP BY
        service, region, sku, usage_unit, currency
    ORDER BY
        total_cost DESC
    """
    
    try:
        print("Running BigQuery...")
        query_job = client.query(query)
        
        costs = []
        for row in query_job:
            try:
                costs.append({
                    "service": row.service or "unknown",
                    "sku": row.sku or "unknown",
                    "cost": {
                        "amount": float(row.total_cost or 0),
                        "currency": row.currency
                    },
                    "usage": {
                        "unit": row.usage_unit or "unknown",
                        "amount": float(row.total_usage or 0)
                    },
                    "location": row.region or "unknown"
                })
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue
                
        return costs
    except Exception as e:
        logger.error(f"Error running BigQuery: {e}")
        return []


def get_gcp_spending_metrics(config: GcpSpendingConfig) -> Dict:
    """Get comprehensive GCP spending metrics."""

    print("About to get storage costs")
    storage_costs = get_storage_costs(config)
    
    print("About to get billing details")
    billing_details = get_billing_details(config)
    
    metrics = {
        "storage": storage_costs,
        "billing": billing_details
    }
    
    print("About to aggregate costs by service")

    # Aggregate costs by service
    service_costs = {}
    for item in metrics["billing"]:
        service = item["service"]
        cost = item["cost"]["amount"]
        service_costs[service] = service_costs.get(service, 0) + cost
    
    metrics["service_costs"] = service_costs
    
    # Calculate total cost
    metrics["total_cost"] = sum(service_costs.values())
    
    return metrics
