library(ParticleBayes)


#
# Let's discretize part of a common distribution
# and see how water-filling resamples a subset of it.
#
wfN = 8
x = seq(0, 10, 1)
lambda = 0.5 
log.weights = dt(x, df=1, log=T)
log.weights.sum = 
    log(sum(dt(x, df=1, log=F)))

wf.result = pb.wf.resample(log.weights, wfN, x, log.weights.sum)

# Plot the original distribution...
plot(x, log.weights - log.weights.sum, type='l')

# Show where the water-filling line is...
log.alpha = pb.wf.alpha(log.weights, wfN, log.weights.sum)
abline(h=-log.alpha, col='red')

# Plot the resampled entries...
x.wf = as.integer(lapply(wf.result, function(x) x$value))
y.wf = as.numeric(lapply(wf.result, function(x) log(x$weight)))
idx = order(x.wf)
lines(x.wf[idx], y.wf[idx], col='blue', type='b')



