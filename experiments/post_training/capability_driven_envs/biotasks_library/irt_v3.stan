// Exploratory prior-anchored hierarchical 2PL; no guessing parameter.
data {
  int<lower=1> M;
  int<lower=1> I;
  int<lower=1> N;
  array[N] int<lower=1, upper=M> respondent;
  array[N] int<lower=1, upper=I> item;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  vector[M] theta_raw;
  vector[I] difficulty_z;
  vector[I] log_discrimination_z;
  real mean_difficulty;
  real<lower=0> sd_difficulty;
  real mean_log_discrimination;
  real<lower=0> sd_log_discrimination;
}
transformed parameters {
  vector[M] theta = theta_raw - mean(theta_raw);
  vector[I] difficulty = mean_difficulty + sd_difficulty * difficulty_z;
  vector[I] log_discrimination = mean_log_discrimination + sd_log_discrimination * log_discrimination_z;
}
model {
  // These priors anchor the scale and shrink poorly identified item slopes.
  theta_raw ~ std_normal();
  difficulty_z ~ std_normal();
  log_discrimination_z ~ std_normal();
  mean_difficulty ~ normal(0, 1.5);
  sd_difficulty ~ normal(0, 0.8);
  mean_log_discrimination ~ normal(0, 0.35);
  sd_log_discrimination ~ normal(0, 0.35);
  for (n in 1:N)
    y[n] ~ bernoulli_logit(exp(log_discrimination[item[n]]) * (theta[respondent[n]] - difficulty[item[n]]));
}
