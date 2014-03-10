# TODO: Add comment
# 
# Author: bwillar0
###############################################################################

wfN = 8
x = seq(0, 10, 1)
log.weights = dt(x, df=1, log=T)
log.weights.sum = 
    log(sum(exp(log.weights)))

wf.result = pb.wf.resample(log.weights, wfN, support=x, logWeightsSum=log.weights.sum)

x.wf = as.integer(names(wf.result))
y.wf = as.numeric(lapply(wf.result, function(x) x$x))

log.alpha = pb.wf.alpha(log.weights, log.weights.sum, wfN)
log.weights.normed = log.weights - log.weights.sum
