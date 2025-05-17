import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('wandb_export_2025-05-13T08_41_24.676-07_00.csv')

# Define column names for each run
soap4_col = 'sweep-520m-85B-soapeaa7a19flr0.004-wd0.1-minlr0-warmup1000-b10.9-fe941f - eval/paloma/c4_en/loss'
mars_col = 'sweep-520m-85B-mars1c70a7lr0.004-wd0.1-minlr0-warmup1000-b10.95--ef2ded - eval/paloma/c4_en/loss'
soap8_col = 'sweep-520m-85B-soapew298532lr0.008-wd0.1-minlr0-warmup1000-b10.9-59eeb6 - eval/paloma/c4_en/loss'

# Plotting with zoomed y-axis
plt.figure()
plt.plot(df['Step'], df[mars_col], label='Mars 4e-3')
plt.plot(df['Step'], df[soap8_col], label='Soap 8e-3')
plt.plot(df['Step'], df[soap4_col], label='Soap 4e-3')

# Styling
plt.xlabel('Step', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('520M 8x Chinchilla loss on C4/EN', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(2.88, 3.2)  # Zoom in to losses under 3.2
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
# plt.savefig('lr_matters.pdf', bbox_inches='tight')
plt.savefig('lr_matters.png')