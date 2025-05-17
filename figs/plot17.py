import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 1. Load and filter for AdamW optimizer ("mini" in this dataset)
df = pd.read_csv('optimizer_loss_scaling.csv')
# 5. Define the two-term scaling law
def scaling_law(xdata, alpha, A, beta, B, gamma):
    N_vals, D_vals = xdata
    return alpha * N_vals**(-A) + beta * D_vals**(-B) + gamma

def get_scaling_law(df, optimizer):
        
    df_adamw = df[df['optimizer'] == optimizer]
    # print(df_adamw)
    # 2. Expected non-embedding parameters per model
    expected_params = {
        "130m": 134217728,
        "300m": 301989888,
        "520m": 536870912,
        "1.2b": 1207959552,
    }

    # 3. Compute N and D
    N = df_adamw['model_size'].map(expected_params).values
    chinchilla = df_adamw['chinchilla'].values
    D = chinchilla * 20 * N

    # 4. Loss values
    loss = df_adamw['loss'].values


    # 6. Initial guesses and fit
    initial_guess = [22, 0.15, 280, 0.28, 1.7]
    popt, _ = curve_fit(scaling_law, (N, D), loss, p0=initial_guess)
    alpha, A, beta, B, gamma = popt

    # 7. Compute RMSE
    rmse = np.sqrt(np.mean((scaling_law((N, D), *popt) - loss)**2))
    return alpha, A, beta, B, gamma, rmse


N = 7207959552
D = 20 * N

N = 7207959552
D = 20 * N


alpha, A, beta, B, gamma, rmse = get_scaling_law(df, 'muon')
print("Fitted parameters for Muon:")
print(f"  α = {alpha:.4f}")
print(f"  A = {A:.4f}")
print(f"  β = {beta:.4f}")
print(f"  B = {B:.4f}")
print(f"  γ = {gamma:.4f}")
print(f"RMSE of the fit: {rmse:.4f}")
muon_loss = scaling_law((N, D), alpha, A, beta, B, gamma)
print(f"Loss for 7B model at 1x chinchilla for Muon: {muon_loss:.4f}")


alpha, A, beta, B, gamma, rmse = get_scaling_law(df, 'adamw')
print("Fitted parameters for AdamW:")
print(f"  α = {alpha:.4f}")
print(f"  A = {A:.4f}")
print(f"  β = {beta:.4f}")
print(f"  B = {B:.4f}")
print(f"  γ = {gamma:.4f}")
print(f"RMSE of the fit: {rmse:.4f}")

# predict the loss for 7B model at 1x chinchilla
adamw_loss = scaling_law((N, D), alpha, A, beta, B, gamma)
print(f"Predicted Loss for 7B model at 1x chinchilla for AdamW: {adamw_loss:.4f}")







# alpha, A, beta, B, gamma, rmse = get_scaling_law(df, 'soap')
# print("Fitted parameters for SOAP:")
# # print(f"  α = {alpha:.4f}")
# # print(f"  A = {A:.4f}")
# # print(f"  β = {beta:.4f}")
# # print(f"  B = {B:.4f}")
# # print(f"  γ = {gamma:.4f}")
# print(f"RMSE of the fit: {rmse:.4f}")

# # predict the loss for 7B model at 1x chinchilla
# N = 7207959552
# D = 20 * N
# muon_loss = scaling_law((N, D), alpha, A, beta, B, gamma)
# print(f"Loss for 7B model at 1x chinchilla for Soap: {muon_loss:.4f}")