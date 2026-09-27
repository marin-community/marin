# simulate for Pile-CC
selected = 8
np.random.seed(42)

# token distribution of each domain
prior_dist = [0.11328527, 0.07960865, 0.00391349, 0.1853759, 
              0.05108136, 0.01596293, 0.10175077, 0.00370752, 
              0.06652935, 0.00175077, 0.02708548, 0.23686921, 
              0.01184346, 0.00792997, 0.00803296, 0.03882595, 
              0.04644696]

samples = np.random.dirichlet(prior_dist * 1, 100000)
samples.shape