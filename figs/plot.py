# Python code to process the dataset and plot the loss curves

import pandas as pd
import matplotlib.pyplot as plt

# Set global font size to 18
plt.rcParams.update({'font.size': 18})

# Load the original CSV dat

# Combine processed data and save
processed = pd.read_csv('processed_loss_curves.csv')

# Plotting
fig, ax = plt.subplots()
for label, group in processed.groupby('lr'):
    ax.plot(group['Step'], group['Loss'], label=label)
ax.set_xlabel('Step', fontsize = 18)
ax.set_ylabel('Loss', fontsize = 18)
ax.set_xticks([0, 25000, 50000], ['0', '25k', '50k'], fontsize = 18)

ax.set_ylim(3.08, 3.3)
ax.set_title('Loss On C4-EN', fontsize = 18)
ax.legend(fontsize = 18)
plt.tight_layout()
plt.savefig('pilot_loss_curves.pdf')
