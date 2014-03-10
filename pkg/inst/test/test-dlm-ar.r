#
# Sanity checks for multi-dim array conversions in r-java bridging, etc.
# 
# Author: bwillar0
###############################################################################

Tt = 10 
sigma = sqrt(0.2)
psi = c(3, 0.2)
x = rep(0, Tt)
xC = sqrt(1)
y = rep(0, Tt)
for (i in 2:(Tt-1)) {
  x[i] = psi[1] + psi[2] * x[i] + rnorm(1, 0, sigma * xC) 
  y[i] = x[i] + rnorm(1, 0, sigma)
}

FF = c(0)
m0 = c(0)
C0 = diag(100,1,1)
mPsi0 = c(1, 1)
CPsi0 = diag(c(100,100))
sigma2Scale = 2 
sigma2Shape = 1
numSubSamples = 3
numParticles = 20 
seed = 112352
ar.res = pb.dlm.ar(
    y, 
    FF,
    m0,
    C0,
    mPsi0,
    CPsi0,
    sigma2Scale,
    sigma2Shape,
    numSubSamples,
    numParticles,
    seed)

expect_true(all(round(rowSums(exp(ar.res$logWeights))) == 1),
    "normalized log weights")
expect_true(all(apply(ar.res$psiCovs, c(1,2), 
            function(x) {isSymmetric(matrix(x, 2))})),
    "symmetric covariance matrices")

