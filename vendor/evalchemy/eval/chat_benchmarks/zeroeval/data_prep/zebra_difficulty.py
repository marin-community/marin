import pandas as pd
import math
import numpy as np


# Function to calculate the difficulty
def calculate_difficulty(N, M):
    difficulty = 1 / (math.factorial(N) ** M)
    return difficulty


# Generate the difficulty table
data = []
for N in range(2, 7):
    row = []
    for M in range(2, 7):
        difficulty = calculate_difficulty(N, M)
        row.append(difficulty)
    data.append(row)

# Create a DataFrame for better visualization
df = pd.DataFrame(data, columns=[f"M={i}" for i in range(2, 7)], index=[f"N={i}" for i in range(2, 7)])

# Convert the difficulties to log scale
log_df = df.applymap(lambda x: np.log10(x))


# Define the categories
def categorize_difficulty(log_value):
    if log_value > -4:
        return "Easy"
    elif -6 < log_value <= -4:
        return "Medium"
    else:
        return "Hard"


# Apply the categorization to the log-transformed DataFrame
category_df = log_df.applymap(categorize_difficulty)

print(category_df)

# Create lists for each category
easy = []
medium = []
hard = []

for N in range(2, 7):
    for M in range(2, 7):
        size = f"{N}*{M}"
        category = category_df.loc[f"N={N}", f"M={M}"]
        if category == "Easy":
            easy.append(size)
        elif category == "Medium":
            medium.append(size)
        else:
            hard.append(size)

# Output the lists
print("easy_sizes = ", easy)
print("medium_sizes = ", medium)
print("hard_sizes = ", hard)
