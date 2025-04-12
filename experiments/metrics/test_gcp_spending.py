"""Test script for GCP spending metrics."""
import logging
from pprint import pprint

from gcp_spending import GcpSpendingConfig, get_gcp_spending_metrics

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    """Test GCP spending metrics."""
    config = GcpSpendingConfig(
        project_id="hai-gcp-models",
        dataset_id="billing_data",  # Your BigQuery dataset containing billing data
        billing_table_id="gcp_billing_export_v1_01B3D1_E06B9B_7A3046"  # Your billing export table
    )
    
    try:
        metrics = get_gcp_spending_metrics(config)
        print("\nGCP Spending Metrics:")
        print("===================")
        pprint(metrics)
        
        if metrics.get("total_cost"):
            print(f"\nTotal Cost: ${metrics['total_cost']:.2f}")
        
        if metrics.get("storage"):
            print("\nStorage Costs by Bucket:")
            print("=====================")
            pprint(metrics["storage"])
        
        if metrics.get("service_costs"):
            print("\nCosts by Service:")
            print("===============")
            for service, cost in metrics["service_costs"].items():
                print(f"{service}: ${cost:.2f}")
                
    except Exception as e:
        logging.error(f"Error running test: {e}")

if __name__ == "__main__":
    main()
