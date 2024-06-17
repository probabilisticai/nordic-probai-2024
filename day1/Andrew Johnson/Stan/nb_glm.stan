data {
  int N;
  int T;
  int K;
  array[N*T] int ID;
  matrix[N*T, K] x;
  array[N*T] int y;
}

parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> u_shape;
}

model {
  alpha ~ normal(0, 5);
  beta ~ std_normal();
  u_shape ~ cauchy(0, 5);
  y ~ neg_binomial_2_log_glm(x, alpha, beta, u_shape);
}

generated quantities {
  array[N*T] int ypred = neg_binomial_2_log_rng(alpha + x * beta, u_shape);
}
