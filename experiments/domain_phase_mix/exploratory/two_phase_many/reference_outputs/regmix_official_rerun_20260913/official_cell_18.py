# take the average of top-k simulated data mixture as the optimal data mixture 
k = 128
top_k_samples = samples[np.argsort(simulation)[0:k]]
top_k_samples.shape

# you can get the optimal data mixture by taking the average of top-k samples
optimal_data_mixture = np.mean(top_k_samples, axis=0)
print(optimal_data_mixture)