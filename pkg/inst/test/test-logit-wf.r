# TODO: remove BayesLogit stuff, make independent tests 
# 
# Author: bwillar0
###############################################################################

library(devtools)
dev_mode(on=T)

#system("cd ~/projects/StatsLibExtensions; mvn clean install -Dmaven.test.skip=true")  
system("cd ~/projects/particlelearningmodels/ParticleLearningModels; mvn clean install -Dmaven.test.skip=true")  
system("cd ~/projects/ParticleBayes; mvn clean package -Dmaven.test.skip=true")

install_local("./pkg")

library(ParticleBayes)
library(plyr)

#
# Real data
#
#data(spambase);
#sbase = spambase[seq(1,nrow(spambase),10),];
#X = model.matrix(is.spam ~ word.freq.free + word.freq.1999, data=sbase);
#y = sbase$is.spam;

#
# Simulated data
#
Nt = 100
#X = matrix(c(rep(1, Nt+1), seq(0, 10, length.out=Nt+1)), ncol=2)
#beta = c(-0.7, 0.2)
#X = matrix(seq(0, 100, length.out=Nt+1), ncol=1)
X = matrix(seq(-10, 10, length.out=Nt+1), ncol=1)
beta = c(0.2)
Xb = X %*% beta
z = Xb + rlogis(nrow(X), 0, 1)
y = ifelse(z > 0, 1, 0)

z.ci = t(qlogis(c(0.05,.5,0.95), 0, 1) + t(matrix(rep(Xb,3),ncol=3)))
print(z.ci)


#
# Initialize our logit
#
m0 = matrix(rep(0, ncol(X)), ncol=1, nrow=ncol(X))
C0 = diag(ncol(X)) * 1
G = diag(nrow(C0))
W = diag(nrow(C0)) * 0.0
numParticles = 1000
seed = 1235325
wf.results = pb.logit.test(
    y, 
    X, 
    m0, 
    C0, 
    G,
    W, 
    numParticles,
    seed,
    3)

stopifnot(all(round(rowSums(exp(wf.results$logWeights)), digits=4) == 1))


#
# make some assessments 
#
wf.results$betas[2,,]
wf.beta.mean = sapply(1:length(y), function(i) {
      wf.results$betas[i,,] * exp(wf.results$logWeights[i,])
    }, simplify=F)
wf.beta.mean = laply(wf.beta.mean, function(x) {colSums(as.matrix(x))})
wf.obs.err = y - plogis(rowSums(as.matrix(wf.beta.mean) * X))
summary(wf.obs.err)
sum(abs(wf.obs.err))

#
# try some other models
#
library(BayesLogit)
em.results = logit.EM(y, X)
em.beta.mean = em.results$beta
em.obs.err = y - plogis(X %*% em.beta.mean)
sum(abs(em.obs.err))

mlogit.results = mlogit(y, X, samp=1500, burn=500)
mlogit.beta.mean = colMeans(as.matrix(mlogit.results$beta[,,1]))
mlogit.obs.err = y - plogis(X %*% mlogit.beta.mean)
sum(abs(mlogit.obs.err))

library(binomlogit)
blogit.results = dRUMAuxMix(as.vector(y), rep(1, length(y)), X, 
    sim=1500, burn=500, m0, C0) 
blogit.beta.mean = colMeans(as.matrix(blogit.results$beta[1,]))
blogit.obs.err = y - plogis(X %*% blogit.beta.mean)
sum(abs(blogit.obs.err))


#
# Plots (that shouldn't be in a unit test)
#
while(length(dev.list()) < 3) {
  dev.new()
}

library(ggplot2)

resid.data = data.frame(iter = seq_along(wf.obs.err),
    type = "wf",
    abs.y.error = abs(wf.obs.err))
resid.data = rbind(resid.data, 
    data.frame(iter = seq_along(em.obs.err),
      type = "em",
      abs.y.error  = abs(em.obs.err)))
resid.data = rbind(resid.data, 
    data.frame(iter = seq_along(blogit.obs.err),
      type = "blogit",
      abs.y.error = abs(blogit.obs.err)))
resid.data = rbind(resid.data, 
    data.frame(iter = seq_along(mlogit.obs.err),
      type = "mlogit",
      abs.y.error = abs(mlogit.obs.err)))

resid.box.plot = ggplot(resid.data, aes(x=iter, y=abs.y.error, 
        colour=type, group=type))
resid.box.plot = resid.box.plot + geom_boxplot() + scale_y_sqrt()
plot(resid.box.plot)  

dev.set(dev.next())


resid.plot = ggplot(resid.data, aes(x=iter, y=abs.y.error, 
        colour=type, group=type))
resid.plot = resid.plot + geom_point() + geom_path() + scale_y_sqrt()
plot(resid.plot)

dev.set(dev.next())


library(Hmisc)
library(reshape2)
wf.beta.quants = adply(1:length(y), 1, function(i) {
      data = wf.results$betas[i,,, drop=F]
      weights = exp(wf.results$logWeights[i,])

      res = adply(data, c(1), function(pdata) {
           wtd.quantile(pdata, 
              weights=weights, 
              probs=c(0.05, 0.5, 0.95),
              normwt=T)
          })
      colnames(res)[1] = "beta.term"
      return(res)
    })

wf.beta.quants = melt(wf.beta.quants, measure.vars=3:5,
    variable.name="quantile")
colnames(wf.beta.quants)[1] = "time"
wf.beta.quants$time = as.numeric(wf.beta.quants$time)
wf.beta.quants = cbind(type="wf", wf.beta.quants) 

blogit.beta.quants = adply(blogit.results$beta, c(1), function(pdata) {
     return(quantile(pdata, probs=c(0.05, 0.5, 0.95)))
    })
colnames(blogit.beta.quants)[1] = "beta.term"
blogit.beta.quants = melt(blogit.beta.quants, measure.vars=2:4,
    variable.name="quantile")
blogit.beta.quants = cbind(type="blogit", blogit.beta.quants) 
blogit.beta.quants = cbind(time=rep(1:Nt, each=3), blogit.beta.quants)

mlogit.beta.quants = adply(mlogit.results$beta, c(2,3), function(pdata) {
     return(quantile(pdata, probs=c(0.05, 0.5, 0.95)))
    })
colnames(mlogit.beta.quants)[1] = "beta.term"
mlogit.beta.quants = melt(mlogit.beta.quants[-2], measure.vars=2:4,
    variable.name="quantile")
mlogit.beta.quants = cbind(type="mlogit", mlogit.beta.quants) 
mlogit.beta.quants = cbind(time=rep(1:Nt, each=3), mlogit.beta.quants)


beta.plot.data = rbind(wf.beta.quants, blogit.beta.quants, mlogit.beta.quants) 
beta.plot = ggplot(beta.plot.data, 
    aes(x=time, y=value, 
        group=interaction(beta.term,quantile,type), 
        colour=type))
#beta.plot = beta.plot + facet_grid(beta.term~., scales="free_y")
beta.plot = beta.plot + geom_line()
beta.plot = beta.plot + geom_hline(aes(yintercept=beta), 
    data.frame(beta.term=seq_along(beta), quantile=c("%50", "%50")))
#beta.plot = beta.plot + geom_ribbon(data=predframe,aes(ymin=lwr,ymax=upr),alpha=0.3)

print(beta.plot)
