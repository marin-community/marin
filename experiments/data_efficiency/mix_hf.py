from collections import Counter

import datasets
from datasets import concatenate_datasets
from tqdm import tqdm


# Load source datasets
dataset1 = datasets.load_dataset("konwoo/dclm-100x-tsp-v2", split="train")
dataset2 = datasets.load_dataset("konwoo/dclm-100x-tsp-v2-shuffled", split="train")

#### Submixed dataset
# # Take the first half of tsp-v2
# half_length = len(dataset1) // 2
# dataset1_first_half = dataset1.select(range(half_length))

# # Count occurrences in the selected half so we can remove the same number from tsp-v2-shuffled
# remove_counts = Counter()
# for example in tqdm(dataset1_first_half, desc="Counting selected occurrences"):
#     remove_counts[example["text"]] += 1

# # Build the subset of tsp-v2-shuffled with counted removals applied
# kept_indices = []
# removed_counts = Counter()
# for idx, example in enumerate(tqdm(dataset2, desc="Filtering shuffled dataset")):
#     text = example["text"]
#     if removed_counts[text] < remove_counts[text]:
#         removed_counts[text] += 1
#         continue
#     kept_indices.append(idx)

# dataset2_filtered = dataset2.select(kept_indices)

# # Merge the selected half of tsp-v2 with the filtered tsp-v2-shuffled
# dataset_mix = concatenate_datasets([dataset1_first_half, dataset2_filtered])

# # Uncomment to push to the hub if desired
# dataset_mix.push_to_hub("kothasuhas/dclm-100x-tsp-v2-submixed")

#### Just double dataset

dataset_double = concatenate_datasets([dataset2, dataset2])
dataset_double.push_to_hub("kothasuhas/dclm-100x-tsp-v2-double")

#### Double checking the number of unique examples in the mixed dataset
# dataset_mix = datasets.load_dataset("kothasuhas/dclm-100x-tsp-v2-submixed", split="train")

# unique_examples = set()

# for example in tqdm(dataset_mix):
#     unique_examples.add(example["text"])

# print(len(unique_examples))
