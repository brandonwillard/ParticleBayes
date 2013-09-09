
wfN = 8

x = seq(0, 10, 1)
lambda = 0.5 
log.weights = dt(x, df=1, log=T)
log.weights.sum = 
    log(sum(dt(x, df=1, log=F)))

log.weights = c(-5.023414050539481, -2.826189473203262, -5.334106145455251, 
    -3.1368815681190316, -3.1368815681190316, -5.334106145455251, 
    -3.1368815681190316, -5.334106145455251, -5.334106145455251, 
    -3.1368815681190316, -3.1368815681190316, -5.334106145455251, 
    -3.1368815681190316, -5.334106145455251, -5.334106145455251, 
    -3.1368815681190316, -5.334106145455251, -3.1368815681190316, 
    -3.1368815681190316, -5.334106145455251)
x = 1:length(log.weights)

log.weights.sum = 0.30685281944005477

wf.result = water.filling.resample(log.weights, log.weights.sum, x, wfN)

x.wf = as.integer(names(wf.result))
y.wf = as.numeric(lapply(wf.result, function(x) x$value))

wt.mean(x, exp(log.weights))
wt.mean(x.wf, exp(y.wf))

wt.var(x, exp(log.weights))
wt.var(x.wf, exp(y.wf))


log.alpha = find.log.alpha(log.weights - log.weights.sum, wfN)

plot(x, log.weights - log.weights.sum, type='l')
idx = order(x.wf)
lines(x.wf[idx], y.wf[idx], col='blue', type='b')
abline(h=-log.alpha, col='red')
