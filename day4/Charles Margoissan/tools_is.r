
is_summary <- function (fit, parms, psis_fit, log_ratios) {
  mean <- c()
  var <- c()
  q5 <- c()
  q25 <- c()
  q50 <- c()
  q75 <- c()
  q95 <- c()
  khat <- c()
  
  for (i in 1:length(parms)) {
    parm_draw <- fit$draws(parms[i])
    mean <- c(mean, E_loo(parm_draw, psis_fit, log_ratios = log_ratios, type = "mean")$value)
    var <- c(var, E_loo(parm_draw, psis_fit, log_ratios = log_ratios, type = "var")$value)
    q5 <- c(q5, E_loo(parm_draw, psis_fit, log_ratios = log_ratios, type = "quantile", probs = 0.05)$value)
    q25 <- c(q25, E_loo(parm_draw, psis_fit, log_ratios = log_ratios, type = "quantile", probs = 0.25)$value)
    q50 <- c(q50, E_loo(parm_draw, psis_fit, log_ratios = log_ratios, type = "quantile", probs = 0.5)$value)
    q75 <- c(q75, E_loo(parm_draw, psis_fit, log_ratios = log_ratios, type = "quantile", probs = 0.75)$value)
    q95 <- c(q95, E_loo(parm_draw, psis_fit, log_ratios = log_ratios, type = "quantile", probs = 0.95)$value)
    khat <- c(khat, E_loo(parm_draw, psis_fit, log_ratios = log_ratios, type = "mean")$pareto_k)
  }
  
  summary <- data.frame(parms = parms, mean = mean, var = var, q5 = q5, q25 = q25, q50 = q50,
                        q75 = q75, q95 = q95, khat = khat)
  return(summary)
}
