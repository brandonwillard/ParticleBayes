#
# FYI: Compile the jars for 1.6, otherwise, rJava simply won't load the classes.  No errors,
# just no classes.  Also, use javap -s blah.class to get the actual signatures that rJava needs.
# Much easier that way.
#

java.check.ex.print.stack <- function() {
  if (!is.null(e<-.jgetEx())) {
    print("Java exception was raised")
    baos <- new(J("java.io.StringWriter"))
    ps <- .jnew("java.io.PrintWriter", .jcast(baos, new.class="java/io/Writer"))
    e$printStackTrace(ps)
    cat(baos$toString())
    return(T)
  } else {
    return(F)
  }
}

find.log.alpha <- function(logWeights, logWeightsSum=NULL, N) {
  jlogWeights = as.double(logWeights)
  jN = as.integer(N)
  stopifnot(N > 1)

  if (is.null(logWeightsSum)) {
    jlogSum = .jcall("com.statslibextensions.statistics.ExtSamplingUtils", "D","logSum", 
                     jlogWeights, check=F)
    if(java.check.ex.print.stack())
      return(NULL)
  } else {
    jlogSum = as.double(logWeightsSum)
  }
  # do this just in case
  jlogWeights = jlogWeights - jlogSum

  result = .jcall("com.statslibextensions.statistics.ExtSamplingUtils", "D","findLogAlpha", 
                  jlogWeights, jlogSum, jN, check=F)
  if(java.check.ex.print.stack())
    return(NULL)
  
  return(result)
}

water.filling.resample <- function(logWeights, logWeightsSum=NULL, support, N, seed=NULL) {
  jrng = .jnew("java.util.Random")
  if (!is.null(seed))
    jrng$setSeed(seed)
  jobjects = new(J("java.util.ArrayList"))
  for (obj in support) {
    jobjects$add(as.character(obj))
  }
  jlogWeights = as.double(logWeights)
  jN = as.integer(N)

  if (is.null(logWeightsSum)) {
    jlogSum = .jcall("com.statslibextensions.statistics.ExtSamplingUtils", 
                     "D","logSum", jlogWeights, check=F)
  } else {
    jlogSum = as.double(logWeightsSum)
  }

  jresult = .jcall("com.statslibextensions.statistics.ExtSamplingUtils", 
                  "Lcom/statslibextensions/statistics/distribution/WFCountedDataDistribution;",
                  "waterFillingResample", jlogWeights, jlogSum, 
                  .jcast(jobjects, new.class="java/util/List"), jrng, jN, check=F)

  if (java.check.ex.print.stack())
    return(NULL)

  aList = .jconvertCountedDataDistribution(jresult)
  
  return(aList)
}

.jconvertCountedDataDistribution <- function(jobj) {
  jmap = jobj$asMap()
  # could just use lapply...
  jkeySet = .jrcall(jmap,"keySet")
  jiter = .jrcall(jkeySet,"iterator")
  aList = list()
  while(.jrcall(jiter,"hasNext")) {
    jkey = .jrcall(jiter,"next", simplify=F);
    jskey = as.character(.jsimplify(jkey))
    jval = .jrcall(jmap,"get",jkey)
    if (.jinstanceof(jval, "com.statslibextensions.math.MutableDoubleCount")) {
      aList[[jskey]]$count = jval$getCount() 
      aList[[jskey]]$value = jval$getValue() 
    } else {
      aList[[jskey]] = jval$getValue()
    }
  }
  return(aList)
}

wfLogit <- function(y, X, m0, C0, numParticles = 100, seed=NULL) {
  jrng = .jnew("java.util.Random")
  if (!is.null(seed))
    jrng$setSeed(seed)
  jmatFact = J("gov.sandia.cognition.math.matrix.MatrixFactory")$getDefault()
  jvecFact = J("gov.sandia.cognition.math.matrix.VectorFactory")$getDefault()
  jm0 = jvecFact$copyArray(.jarray(m0)) 
  jC0 = jmatFact$copyArray(.jarray(C0, dispatch=T)) 
  jinitPrior = new(J("gov.sandia.cognition.statistics.distribution.MultivariateGaussian"),
                     jm0, jC0)
  jmodelCovar = jmatFact$copyArray(.jarray(diag(nrow(C0)), dispatch=T))
  jmodelCovar = jmatFact$copyArray(.jarray(diag(nrow(C0)), dispatch=T))
  jF = jmatFact$copyArray(.jarray(t(as.matrix(X[1,])), dispatch=T)) 
  jG = jmatFact$copyArray(.jarray(diag(nrow(C0)), dispatch=T))
  #jLogitPLFilter = .jnew("plm.logit.fruehwirth.FruehwirthLogitPLFilter", 
  #                       jinitPrior, jF, jG, jmodelCovar, jrng)
  jLogitPLFilter = new(J("plm.logit.fruehwirth.FruehwirthLogitPLFilter"), 
                         jinitPrior, jF, jG, jmodelCovar, jrng)
  jLogitPLFilter$setNumParticles(as.integer(numParticles))
  jParticles = jLogitPLFilter$createInitialLearnedObject()
  distsList = vector("list", length(y))
  for (i in 1:length(y)) {
    obs = y[i]
    jobsVect = jvecFact$copyArray(.jarray(obs)) 
    jobsData = jmatFact$copyArray(.jarray(t(as.matrix(X[i,])), dispatch=T)) 
    jobs = J("com.statslibextensions.util.ObservedValue")$create(jobsVect, jobsData)
    .jcall(jLogitPLFilter, "V", "update", 
           .jcast(jParticles, "gov/sandia/cognition/statistics/DataDistribution"), 
           .jcast(jobs, "com.statslibextensions.util.ObservedValue"), 
            check=F)
    #jLogitPLFilter$update(jParticles, jobs)  
    if (java.check.ex.print.stack())
      break
    jparticleMap = jParticles$asMap()
    beta = t(sapply(jparticleMap$entrySet(), function(x) {
              x$getKey()$getLinearState()$getMean()$toArray()
            }))
    weights = sapply(jparticleMap$entrySet(), function(x) {
              jval = x$getValue()
              if (.jinstanceof(jval, "com.statslibextensions.math.MutableDoubleCount")) {
                weight = jval$getCount() * jval$getValue() 
              } else {
                weight = jval$getValue()
              }
              return(weight)
            })
    thisDist = list("beta"=beta, "weights"=weights)
    distsList[[i]] = thisDist
  }
  return(distsList)
}






