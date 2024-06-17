data {
  int N;
  int T;
  int K;
  matrix[N*T, K] x;
  vector[N*T] y;
  int sample_prior;
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 5);
  sigma ~ normal(0, 5);
  // Don’t model observed outcomes
  // until we’re ready
  if (sample_prior == 0) {
    y ~ normal_id_glm(x, alpha, beta,
                      sigma);
  }
}
generated quantities {
  array[N*T] real ypred;
  ypred = normal_rng(alpha + x * beta,
                     sigma);
}
