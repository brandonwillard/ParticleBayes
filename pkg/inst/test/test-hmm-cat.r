# TODO: Add comment
# 
# Author: bwillar0
###############################################################################

T = 50
N = 2 # observation categories
M = 3 # hidden states
p0 = rep(1/M, M)
P = matrix(rep(1/M, M, each=M), M, M)
py = matrix(rep(1/N, N, each=M), M, N)
y = rep(0,T)

prun = p0
for(i in 1:T) {
  prun = (P %*% prun)
  y[i] = sample.int(N, 1, prob=t(py) %*% prun)
}

hmmClassProbs = p0
hmmTransProbs = P 
emissionProbs = py
numParticles = 100
seed = 12335
res = pb.hmm.cat(
    y,
    hmmClassProbs,
    hmmTransProbs,
    emissionProbs,
    numParticles,
    seed)

expect_true(F, "hmm test")
