import pickle

# Load the data
path = "experiments/optimizer_sweep/Analysis/PhaseI/1d_losses.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

# Examine the structure
print("Data keys:", list(data.keys()))
print("\n" + "="*80)

for optimizer_name, optimizer_data in data.items():
    if optimizer_data:
        print(f"\nOptimizer: {optimizer_name}")
        print(f"Config keys: {list(optimizer_data.keys())}")
        
        # Look at first config
        first_config_key = list(optimizer_data.keys())[0]
        first_config = optimizer_data[first_config_key]
        print(f"First config key: {first_config_key}")
        print(f"First config data keys: {list(first_config.keys())}")
        
        if 'result' in first_config:
            result_data = first_config['result']
            print(f"Result data keys: {list(result_data.keys())}")
            
            # Show some example keys
            example_keys = list(result_data.keys())[:5]
            print(f"Example result keys: {example_keys}")
            
            # Look for config information
            if 'best_config' in result_data:
                print(f"Best config: {result_data['best_config']}")
        
        break  # Just look at first optimizer 