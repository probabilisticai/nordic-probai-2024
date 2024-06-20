
data {
  int N;
  vector[N] x;
  vector[N] y;
}

parameters {
  real beta;
  real<lower = 0> sigma;
}

model {
  // prior
  beta ~ normal(1, 1);
  sigma ~ gamma(1, 1);
  
  // likelihood
  y ~ normal(x * beta, sigma);
}

generated quantities {
  array[N] real y_pred = normal_rng(beta * x, sigma);
}
