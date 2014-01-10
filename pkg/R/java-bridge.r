#
# FYI: Compile the jars for 1.6, otherwise, rJava simply won't load the classes.  No errors,
# just no classes.  
# Also, for development, use javap -s blah.class to get the actual signatures that rJava needs.
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

#'
#' Computes the water-filling level, alpha, with which 
#' samples, and their weights, are deterministically accepted.
#' 
#' @param logWeights vector of log weights
#' @param logWeightsSum total log weight, i.e. log sum of weights.  The value NULL implies that
#'   	the sum is to be computed by \code{find.log.alpha}.
#' @param N the number of samples.  Must be <= the length of logWeights.
#' @details For details concerning the algorithm see the paper by Nicholas Polson, Brandon Willard (2014).
#' @return numeric value of alpha
#' @references Nicholas G. Polson, Brandon Willard (2014), "Recursive Bayesian Computation".
#' @author Brandon Willard \email{brandonwillard@@gmail.com}
#' @seealso \code{\link{water.filling.resample}}
#' @examples 
#' 
#'   x = seq(0, 10, 1)
#'   log.weights = dt(x, df=1, log=T)
#'   log.weights.sum = log(sum(dt(x, df=1, log=F)))
#' 
#'   log.alpha = find.log.alpha(log.weights, log.weights.sum, 8)
#'   print(log.alpha)
#' @keywords water-filling
#' 
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

wfLogit <- function(y, X, 
    m0 = rep(0, ncol(X)), C0 = diag(ncol(X)), 
    G = diag(nrow(C0)), W = diag(nrow(C0)), 
    numParticles = 1000, waterFilling=T, seed=NULL) {

  jseed = ifelse(is.null(seed), 
          as.integer(.Random.seed[sample(3:length(.Random.seed), 1)]), 
          as.integer(seed))
  jm0 = .jarray(m0) 
  jC0 = .jarray(as.matrix(C0), dispatch=T) 
  jF = .jarray(t(as.matrix(X[1,])), dispatch=T) 
  jG = .jarray(as.matrix(G), dispatch=T)
  jmodelCovar = .jarray(as.matrix(W), dispatch=T)
  jnumParticles = as.integer(numParticles)
  jobsData = .jarray(as.matrix(X), dispatch=T) 
  jFsPlAdapter = J("org.bitbucket.brandonwillard.particlebayes.radapters.FruehwirthLogitPLAdapter")
  jResult = jFsPlAdapter$batchUpdate(jm0, jC0, jF, jG, jmodelCovar, 
                                     .jarray(as.double(y)), 
                                     jnumParticles,
                                     jobsData,
                                     waterFilling,
                                     jseed) 
  rlogWeights = .jevalArray(jResult$getLogWeights(), simplify=T)
  rbetas = .jevalArray(jResult$getStateMeans(), simplify=T)

  return(list(logWeights = rlogWeights, betas = rbetas))
}

wfMultiLogit <- function(y, M, X, 
    m0 = rep(0, ncol(X)), C0 = diag(ncol(X)), 
    G = diag(ncol(X)), W = diag(ncol(X)), 
    numParticles = 1000, seed=NULL) {

  jseed = ifelse(is.null(seed), 
          as.integer(.Random.seed[sample(3:length(.Random.seed), 1)]), 
          as.integer(seed))
  jm0 = .jarray(m0) 
  jC0 = .jarray(as.matrix(C0), dispatch=T) 
  jF = .jarray(t(as.matrix(X[1,])), dispatch=T) 
  jG = .jarray(as.matrix(G), dispatch=T)
  jmodelCovar = .jarray(as.matrix(W), dispatch=T)
  jnumParticles = as.integer(numParticles)
  jnumCategories = as.integer(M)
  jobsData = .jarray(as.matrix(X), dispatch=T) 
  jFsPlAdapter = J("org.bitbucket.brandonwillard.particlebayes.radapters.FruehwirthMultiPLAdapter")
  jResult = jFsPlAdapter$batchUpdate(jm0, jC0, jF, jG, jmodelCovar, 
                                     .jarray(as.double(y)), 
                                     jnumParticles,
                                     jnumCategories,
                                     jobsData,
                                     jseed) 
  rlogWeights = .jevalArray(jResult$getLogWeights(), simplify=T)
  rbetas = .jevalArray(jResult$getStateMeans(), simplify=T)

  return(list(logWeights = rlogWeights, betas = rbetas))
}

catHmmPL <- function(y, M, X, 
    hmmClassProbs, hmmTransProbs, emissionProbs,
    numParticles = 1000, seed=NULL) {

  jseed = ifelse(is.null(seed), 
          as.integer(.Random.seed[sample(3:length(.Random.seed), 1)]), 
          as.integer(seed))
  jhmmClassProbs = .jarray(hmmClassProbs) 
  mhmmTransProbs = as.matrix(hmmTransProbs)
  stopifnot(nrow(mhmmTransProbs) != length(hmmClassProbs))
  jhmmTransProbs = .jarray(mhmmTransProbs, dispatch=T) 
  memissionProbs = as.matrix(emissionProbs)
  stopifnot(nrow(memissionProbs) != length(memissionProbs))
  jemissionProbs = .jarray(memissionProbs, dispatch=T) 
  jFsPlAdapter = J("org.bitbucket.brandonwillard.particlebayes.radapters.CategoricalHmmPLAdapter")
  jResult = jFsPlAdapter$batchUpdate(jhmmClassProbs, jhmmTransProbs, jemissionProbs,
                                     .jarray(as.integer(y)), 
                                     jnumParticles,
                                     jseed) 
  rlogWeights = .jevalArray(jResult$getLogWeights(), simplify=T)
  rclassIds = .jevalArray(jResult$getClassIds(), simplify=T)

  return(list(logWeights = rlogWeights, classIds = rclassIds))
}



