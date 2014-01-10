library(ParticleBayes)

wfN = 8

x = seq(0, 10, 1)
lambda = 0.5 
log.weights = dt(x, df=1, log=T)
log.weights.sum = 
    log(sum(dt(x, df=1, log=F)))

wf.result = water.filling.resample(log.weights, log.weights.sum, x, wfN)

x.wf = as.integer(names(wf.result))
y.wf = as.numeric(lapply(wf.result, function(x) x$value))

log.alpha = find.log.alpha(log.weights, log.weights.sum, wfN)

plot(x, log.weights - log.weights.sum, type='l')
idx = order(x.wf)
lines(x.wf[idx], y.wf[idx], col='blue', type='b')
abline(h=-log.alpha, col='red')

