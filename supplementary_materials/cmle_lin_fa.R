# negative loglikelihood for aj, sigmaj in linear factor model
lin_fa_nll <- function(params,    # vec concatenating aj: (D+1)-dim vec and sigmaj
                       Y_j,      # resp vec on item j
                       thetas,   # matrix: NxD
                       lambda = 0){ # l2 reg         
  a_j = params[-length(params)]
  sigma_j = params[length(params)]
  dot_prod <- cbind(1, thetas) %*% a_j
  dev_sqr <- (Y_j - dot_prod)^2
  l_i <- -log(sqrt(2*pi) * sigma_j) - dev_sqr/(2*sigma_j^2) 
  return(-mean(l_i)+ 
           lambda * sum(a_j[-1]^2))
}

# negative loglikelihood for thetai in linear factor model
lin_fa_nll_theta <- function(theta_i,  # D-dim vec
                       Y_i,      # resp vec on item j
                       sigma_vec,# uniq vars
                       a_mat){   # matrix: Jx(D+1)         
  dot_prod <-  a_mat %*% c(1, theta_i)
  dev_sqr <- (Y_i - dot_prod)^2
  l_i <- -log(sqrt(2*pi) * sigma_vec) - dev_sqr/(2*sigma_vec^2) 
  return(-mean(l_i))
}



# ----- For testing estimator w/ simulations ---------------
# generate responses
lin_fa_resp_gen <- function(a_mat, sigma_vec, thetas){
  # NxJ mat of expectations
  dot_prod <- cbind(1,thetas) %*% t(a_mat)
  # generate noise (N(0, sigma_j^2))
  noise_mat <- matrix(rnorm(nrow(thetas)*nrow(a_mat)),
                      nrow = nrow(thetas),
                      ncol = nrow(a_mat))
  # scale N(0,1) errors by sigma_js
  noise_mat <- t(t(noise_mat) * sigma_vec)
  # add noise to expectations
  resp <- dot_prod + noise_mat
  return(resp)
}



# # code for testing cmle alg.
# # v1: uncorrelated traits
# Cor1 <- diag(rep(1, 4))
# # v2: correlated traits
# Cor2 <- rbind(c(1,.5,.56,.86),
#               c(.5,1,.68,.64),
#               c(.56, .68, 1, .53),
#               c(.86,.64,.53,1))
# # generate thetas
# library(mvtnorm)
# set.seed(0000)
# N <- 1700
# J <- 20
# thetas <- rmvnorm(N, sigma = Cor2)
# D <- ncol(thetas)
# # generate loadings
# loading_str <- matrix(sample(c(.1, 1), size = J*D,
#                              replace = T,prob = c(.7,.3)),
#                       nrow = J, ncol  =D)
# a_mat <- loading_str * runif(J*D)
# norm_ind <- which(rowSums(a_mat**2)>1)
# a_mat[norm_ind,] <- a_mat[norm_ind,]/(sqrt(sum(a_mat[norm_ind,]**2))*1.1)
# # get uniqueness variances
# sigma_vec <- sqrt(1-rowSums(a_mat^2))
# # add intercept ~ N(0,1)
# a_mat <- cbind(rnorm(J), a_mat)
# # generate resps
# resp <- lin_fa_resp_gen(a_mat, sigma_vec, thetas)
# # estimate loadings and sigmas
# a_ests <- matrix(NA, nrow = J, ncol  =(D+1))
# sig_ests <- numeric(J)
# for(j in 1:J){
#   init <- c(numeric(5),1)
#   out <- optim(init,fn = lin_fa_nll,
#                       Y_j = resp[,j], thetas = thetas)
#   a_ests[j,] <- out$par[-(D+2)]
#   sig_ests[j] <- out$par[(D+2)]
# }
# 
# plot(a_ests, a_mat)
# abline(0,1)
# plot(sigma_vec, sig_ests)
# abline(0,1)
# # estimate thetas instead
# theta_ests <- matrix(NA, nrow = N, ncol = D)
# for(i in 1:N){
#   init <- numeric(D)
#   out <- optim(init,fn = lin_fa_nll_theta,
#                Y_i = resp[i,], sigma_vec = sigma_vec,
#                a_mat = a_mat)
#   theta_ests[i,] <- out$par
# }
# par(mfrow = c(2,2))
# for(d in 1:D){
#   plot(theta_ests[,d], thetas[,d],
#        main = paste0('F',d,': ',
#                      round(cor(theta_ests[,d],thetas[,d]),2)))
# }