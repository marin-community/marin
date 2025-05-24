from datasets import load_dataset

# Load the dataset
dataset_name = "LLM360/MegaMath"
try:
    dataset = load_dataset(dataset_name)
    print(f"Successfully loaded dataset: {dataset_name}")
    print("Dataset features:")
    print(dataset)
except Exception as e:
    print(f"Error loading dataset {dataset_name}: {e}")
