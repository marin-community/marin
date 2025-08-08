path = "experiments/optimizer_sweep/Analysis/PhaseI/1d_losses.pkl"
import pickle

with open(path, "rb") as f:
    data = pickle.load(f)
print(data['adamw'])