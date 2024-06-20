
functions {
  vector sir(real t, vector y, real beta, real gamma, int N) {
    real S = y[1];
    real I = y[2];
    real R = y[3];
    
    vector[3] dy_dt;
    
    dy_dt[1] = - beta * I * S / N;
    dy_dt[2] = beta * I * S / N - gamma * I;
    dy_dt[3] = gamma * I;
    
    return dy_dt;
  }
  
}

data {
  int<lower=1> n_days;
  vector[3] y0;
  real t0;
  array[n_days] real ts;
  int N;
  array[n_days] int cases;
}

parameters {
  real<lower=0> gamma;
  real<lower=0> beta;
  real<lower=0> phi_inv;
}

transformed parameters {
  real phi = 1. / phi_inv;
  array[n_days] vector[3] y = ode_rk45(sir, y0, t0, ts, beta, gamma, N);
}

model {
  //priors
  beta ~ normal(2, 1);
  gamma ~ normal(0.4, 0.5);
  phi_inv ~ exponential(5);
  
  cases ~ neg_binomial_2(y[,2], phi);
}

generated quantities {
  real R0 = beta / gamma;
  real recovery_time = 1 / gamma;
  array[n_days] real pred_cases = neg_binomial_2_rng(y[,2], phi);
  
  vector[n_days] log_lik;
  for (i in 1:n_days)
    log_lik[i] = neg_binomial_2_lpmf(cases[i] | y[i, 2], phi);
}
