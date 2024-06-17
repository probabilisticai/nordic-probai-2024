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
  vector<lower=0>[N] u;
}

transformed parameters {
  vector[N*T] lambda = log(u[ID]) + alpha + x * beta;
}

model {
  alpha ~ normal(0, 5);
  beta ~ std_normal();
  u_shape ~ cauchy(0, 5);
  u ~ gamma(u_shape, u_shape);
  y ~ poisson_log(lambda);
}

generated quantities {
  array[N*T] int ypred = poisson_log_rng(lambda);
}
