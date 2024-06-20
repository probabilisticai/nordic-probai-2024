
data {
  int N;
  vector[N] x;
  vector[N] y;
}

parameters {
  real beta_0;
  real beta;
  real<lower = 0> sigma;
}

model {
  // prior
  beta ~ normal(2.0, 1.0);
  sigma ~ gamma(1.0, 1.0);

  // likelihood
  y ~ normal(beta_0 + beta * x, sigma);
}

generated quantities {
  array[N] real y_pred = normal_rng(beta_0 + beta * x, sigma);
}
