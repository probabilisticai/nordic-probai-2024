
functions {
  // specify system of ODEs
  vector sir(real t, vector y, real beta, real gamma, int N) {
    real S = y[1];
    real I = y[2];
    real R = y[3];
    
    vector[3] dy_dt;
    
    // CODE: return derivative of y
    dy_dt[1] = ;
    dy_dt[2] = ;
    dy_dt[3] = ;
    
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
  // CODE: specificy model parameters

}

transformed parameters {
  // Call to ODE integrator
  array[n_days] vector[3] y = ode_rk45(sir, y0, t0, ts, beta, gamma, N);
}

model {
  // CODE: specify priors
  
  // CODE: specify likelihood

}
