#
# FYI: Compile the jars for 1.6, otherwise, rJava simply won't load the classes.  No errors,
# just no classes.  
# Also, for development, use javap -s blah.class to get the actual signatures that rJava needs.
# Much easier that way.
# FYI: system.file("jri", package="rJava")
#

.jconvertCountedDataDistribution <- function(jobj, ref.support = NULL) {
  jmap = jobj$asMap()
  # could just use lapply...
  jkeySet = .jrcall(jmap,"keySet")
  jiter = .jrcall(jkeySet,"iterator")
  jsize = .jrcall(jkeySet,"size")
  aList = vector(mode="list", length=as.integer(jsize))
  i = 1
  while(.jrcall(jiter,"hasNext")) {
    jkey = .jrcall(jiter,"next", simplify=F);
    jobjKey = .jcast(jkey ,new.class="java.lang.Object")
    idx = as.integer(.jsimplify(jkey))
    rskey = idx 
    if (!is.null(ref.support)) {
      rskey = ref.support[idx]  
    } 
    jval = .jrcall(jmap,"get", jobjKey)
    aList[[i]]$value = rskey
    aList[[i]]$weight = .jrcall(jobj, "getFraction", jobjKey)
    if (.jinstanceof(jval, "com.statslibextensions.math.MutableDoubleCount")) {
      aList[[i]]$count = jval$getCount() 
    } 
    i = i + 1
  }
  return(aList)
}


#'
#' Computes the logarithm of the water-filling level, alpha, with which 
#' samples, and their weights, are deterministically accepted in the
#' water-filling resample method.
#' 
#' @param logWeights vector of log weights
#' @param N the number of samples.  Must be <= the length of logWeights.
#' @param logWeightsSum total log weight, i.e. log sum of weights.  The value 
#'  \code{NULL} implies that the sum is to be computed.
#' @details For details concerning the algorithm see the paper by Nicholas Polson, Brandon Willard (2014).
#' @return numeric value of log(alpha)
#' @references Nicholas G. Polson, Brandon Willard (2014), "Recursive Bayesian Computation".
#' @author Brandon Willard \email{brandonwillard@@gmail.com}
#' @seealso \code{\link{pb.wf.resample}}
#' @examples 
#'   x = seq(0, 10, 1)
#'   log.weights = dt(x, df=1, log=TRUE)
#'   log.weights.sum = log(sum(dt(x, df=1, log=FALSE)))
#'   log.alpha = pb.wf.alpha(log.weights, 8, log.weights.sum)
#'   print(log.alpha)
#' @keywords water-filling
#' @export
pb.wf.alpha <- function(logWeights, N, logWeightsSum=NULL) {
  jlogWeights = as.double(logWeights)
  jN = as.integer(N)
  stopifnot(N > 1)

  if (is.null(logWeightsSum)) {
    jlogSum = .jcall("com.statslibextensions.statistics.ExtSamplingUtils", "D","logSum", 
                     jlogWeights, check=F)
    if(.java.check.ex.print.stack())
      return(NULL)
  } else {
    jlogSum = as.double(logWeightsSum)
  }
  # do this just in case
  #jlogWeights = jlogWeights - jlogSum

  result = .jcall("com.statslibextensions.statistics.ExtSamplingUtils", "D","findLogAlpha", 
                  jlogWeights, jlogSum, jN, check=F)
  if(.java.check.ex.print.stack())
    return(NULL)
  
  return(result)
}

#'
#' Given weights and an associated support, this function will 
#' perform water-filling resampling.  
#' 
#' @param logWeights vector of log weights
#' @param support objects in the suppport that are associated with the \code{logWeights}.
#' @param N the number of samples.  Must be <= the length of logWeights.
#' @param logWeightsSum total log weight, i.e. log sum of weights.  The value 
#'  \code{NULL} implies that the sum is to be computed.
#' @param seed seed for the random number generator.
#' @details For details concerning the algorithm see the paper by Nicholas Polson, Brandon Willard (2014).
#' @return A list containing each resampled object in the support and its associated weight. 
#' @references Nicholas G. Polson, Brandon Willard (2014), "Recursive Bayesian Computation".
#' @author Brandon Willard \email{brandonwillard@@gmail.com}
#' @examples 
#' x = seq(0, 10, 1)
#' log.weights = dt(x, df=1, log=TRUE)
#' 
#' wf.result = pb.wf.resample(log.weights, 8, support=x)
#' @keywords water-filling
#' @export 
pb.wf.resample <- function(logWeights, N, support=NULL, 
    logWeightsSum=NULL, seed=NULL) {
  jrng = .jnew("java.util.Random")
  if (!is.null(seed))
    jrng$setSeed(seed)
  jobjects = new(J("java.util.ArrayList"))

  rsupport = seq_along(logWeights)
  for (obj in rsupport) {
    .jcall(jobjects, "Z", "add", 
        .jcast(
            .jnew("java.lang.Integer", as.integer(obj))
            ,new.class="java.lang.Object")
        )
    #.jcall(jobjects, "B", "add", .jnew("java.lang.Integer", as.integer(obj)))
    #jobjects$add(.jnew("java.lang.Integer", as.integer(obj)))
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
                  .jcast(jobjects, new.class="java/util/Collection"), jrng, jN, check=F)

  if (.java.check.ex.print.stack())
    return(NULL)

  aList = .jconvertCountedDataDistribution(jresult, support)
  
  return(aList)
}

#'
#' Particle filter implementing binomial logistic regression via an EV mixture
#' of normals approximation and water-filling.
#' 
#' @param y dependent variable vector
#' @param X explanatory variable matrix
#' @param m0 initial prior mean vector for the state variable
#' @param C0 initial prior covariance matrix for the state variable
#' @param G constant state evolution matrix
#' @param W constant state evolution covariance matrix
#' @param numParticles number of particles
#' @param seed seed for the random number generator.
#' @details For details concerning the algorithm see the paper by Nicholas Polson, Brandon Willard (2014).
#' @return A list containing each resampled object in the support and its associated weight. 
#' @references 
#' Sylvia Fruehwirth-Schnatter and Rudolf Fruehwirth (2010),
#' "Data augmentation and MCMC for binary and multinomial logit models."
#' In \emph{Statistical Modelling and Regression Structures - Festschrift in Honour
#' of Ludwig Fahrmeir}, T. Kneib and G. Tutz, Eds. Physica-Verlag, Heidelberg, pp. 111-132.
#' 
#' Nicholas G. Polson, Brandon Willard (2014), "Recursive Bayesian Computation".
#' @author Brandon Willard \email{brandonwillard@@gmail.com}
#' @keywords water-filling 
#' @keywords logistic regression
#' @export 
pb.logit.wf <- function(y, X, 
    m0 = rep(0, ncol(X)), C0 = diag(ncol(X)), 
    G = diag(nrow(C0)), W = diag(nrow(C0)), 
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
  jX = .jarray(as.matrix(X), dispatch=T) 
  jFsPlAdapter = J("org.bitbucket.brandonwillard.particlebayes.radapters.LogitFSAdapter")
  jResult = jFsPlAdapter$batchUpdate(
      .jarray(as.double(y)), jX,
      jm0, jC0, 
      jF, jG, jmodelCovar, 
      jnumParticles,
      jseed, 1) 

  T = length(y)
  N = numParticles
  Nm = length(m0)
  rlogWeights = aperm(structure(jResult$getLogWeights(), dim=c(N,T)), c(2,1))
  rbetas = aperm(structure(jResult$getStateMeans(), dim=c(Nm,N,T)), c(3,2,1))

  return(list(logWeights = rlogWeights, betas = rbetas))
}

#' This is only for testing other variants of the logit filter.
#' @export 
pb.logit.test <- function(y, X, 
    m0 = rep(0, ncol(X)), C0 = diag(ncol(X)), 
    G = diag(nrow(C0)), W = diag(nrow(C0)), 
    numParticles = 1000, seed=NULL, version=as.integer(0)) {

  jseed = ifelse(is.null(seed), 
          as.integer(.Random.seed[sample(3:length(.Random.seed), 1)]), 
          as.integer(seed))
  jm0 = .jarray(m0) 
  jC0 = .jarray(as.matrix(C0), dispatch=T) 
  jF = .jarray(t(as.matrix(X[1,])), dispatch=T) 
  jG = .jarray(as.matrix(G), dispatch=T)
  jmodelCovar = .jarray(as.matrix(W), dispatch=T)
  jnumParticles = as.integer(numParticles)
  jX = .jarray(as.matrix(X), dispatch=T) 
  jFsPlAdapter = J("org.bitbucket.brandonwillard.particlebayes.radapters.LogitFSAdapter")
  jResult = jFsPlAdapter$batchUpdate(
      .jarray(as.double(y)), jX,
      jm0, jC0, 
      jF, jG, jmodelCovar, 
      jnumParticles,
      jseed, as.integer(version)) 

  T = length(y)
  N = numParticles
  Nm = length(m0)
  rlogWeights = aperm(structure(jResult$getLogWeights(), dim=c(N,T)), c(2,1))
  rbetas = aperm(structure(jResult$getStateMeans(), dim=c(Nm,N,T)), c(3,2,1))

  return(list(logWeights = rlogWeights, betas = rbetas))
}

#wfMultiLogit <- function(y, M, X, 
#    m0 = rep(0, ncol(X)), C0 = diag(ncol(X)), 
#    G = diag(ncol(X)), W = diag(ncol(X)), 
#    numParticles = 1000, seed=NULL) {
#
#  jseed = ifelse(is.null(seed), 
#          as.integer(.Random.seed[sample(3:length(.Random.seed), 1)]), 
#          as.integer(seed))
#  jm0 = .jarray(m0) 
#  jC0 = .jarray(as.matrix(C0), dispatch=T) 
#  jF = .jarray(t(as.matrix(X[1,])), dispatch=T) 
#  jG = .jarray(as.matrix(G), dispatch=T)
#  jmodelCovar = .jarray(as.matrix(W), dispatch=T)
#  jnumParticles = as.integer(numParticles)
#  jnumCategories = as.integer(M)
#  jobsData = .jarray(as.matrix(X), dispatch=T) 
#  jFsPlAdapter = J("org.bitbucket.brandonwillard.particlebayes.radapters.FruehwirthMultiPLAdapter")
#  jResult = jFsPlAdapter$batchUpdate(jm0, jC0, jF, jG, jmodelCovar, 
#                                     .jarray(as.double(y)), 
#                                     jnumParticles,
#                                     jnumCategories,
#                                     jobsData,
#                                     jseed) 
#  rlogWeights = .jevalArray(jResult$getLogWeights(), simplify=T)
#  rbetas = .jevalArray(jResult$getStateMeans(), simplify=T)
#
#  return(list(logWeights = rlogWeights, betas = rbetas))
#}

#'
#' Particle filter for a Hidden Markov Model with categorical emissions. 
#' 
#' @param y dependent variable vector
#' @param hmmClassProbs vector of initial HMM state probabilities
#' @param hmmTransProbs matrix of HMM state stransitions
#' @param emissionProbs vectors of probabilities for observable categories
#' in each HMM state.  Rows span components, columns span observation categories 
#' @param numParticles number of particles
#' @param seed seed for the random number generator.
#' @details For details concerning the algorithm see the paper by Nicholas Polson, Brandon Willard (2014).
#' @return A list containing a matrix of estimated HMM states and
#' corresponding a vector of weights, for each particle and time
#' @references 
#' Nicholas G. Polson, Brandon Willard (2014), "Recursive Bayesian Computation".
#' @author Brandon Willard \email{brandonwillard@@gmail.com}
#' @keywords water-filling 
#' @keywords hidden markov model
#' @export 
pb.hmm.cat <- function(y,  
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

#'
#' Particle filter for a gaussian AR(1) dynamic linear model.  States
#' have normal priors, AR(1) and state transition constant parameters
#' are a joint normal --psi-- distribution, and sigma2 has an inverse-gamma prior.
#' 
#' @param y dependent variable vector
#' @param FF observation matrix
#' @param m0 state prior mean vector
#' @param C0 state prior covariance matrix
#' @param mPsi0 vector of state transition constant and AR(1) prior means
#' @param CPsi0 state transition constant and AR(1) prior covariance matrix
#' @param sigma2Scale scale parameter for sigma2 inverse-gamma prior
#' @param sigma2Shape shape parameter for sigma2 inverse-gamma prior
#' @param numSubSamples sub samples for state variable
#' @param numParticles number of particles
#' @param seed seed for the random number generator.
#' @details For details concerning the algorithm see the paper by Nicholas Polson, Brandon Willard (2014).
#' @return returns a list containing matrices spanning time, particle and values, where values
#' are logWeights, stateMeans, stateCovs, psiMeans, psiCovs, sigma2Shapes, and sigma2Scales
#' @references 
#' Nicholas G. Polson, Brandon Willard (2014), "Recursive Bayesian Computation".
#' @author Brandon Willard \email{brandonwillard@@gmail.com}
#' @keywords autoregressive
#' @keywords dynamic linear model 
#' @keywords dlm 
#' @keywords water-filling 
#' @keywords hidden markov model
#' @export  
pb.dlm.ar <- function(y, FF,
    m0, C0 = diag(length(m0)), 
    mPsi0 = rep(0, 2*ncol(C0)), CPsi0 = diag(2*ncol(C0)), 
    sigma2Scale = 2, sigma2Shape = 1,  
    numSubSamples = 3, numParticles = 1000, seed=NULL) {

  stopifnot(length(m0) == nrow(C0))
  stopifnot(length(mPsi0) == nrow(CPsi0))
  stopifnot(length(mPsi0)/2 == length(m0))
  stopifnot(numSubSamples >= 1)
  stopifnot(sigma2Scale > 1)
  stopifnot(sigma2Shape > 0)

  jseed = ifelse(is.null(seed), 
          as.integer(.Random.seed[sample(3:length(.Random.seed), 1)]), 
          as.integer(seed))
  jm0 = .jarray(m0) 
  jC0 = .jarray(as.matrix(C0), dispatch=T) 
  jmPsi0 = .jarray(mPsi0) 
  jCPsi0 = .jarray(as.matrix(CPsi0), dispatch=T) 
  jsigma2Scale = as.double(sigma2Scale)
  jsigma2Shape = as.double(sigma2Shape)
  jF = .jarray(as.matrix(FF), dispatch=T) 
  jnumParticles = as.integer(numParticles)
  jnumSubSamples = as.integer(numSubSamples)
  jFsPlAdapter = J("org.bitbucket.brandonwillard.particlebayes.radapters.GaussianArHpPLAdapter")
  jResult = jFsPlAdapter$batchUpdate(
      jm0, jC0, 
      jmPsi0, jCPsi0, 
      jsigma2Shape, jsigma2Scale,
      jF, 
      jnumSubSamples,
      .jarray(as.matrix(y), dispatch=T), 
      jnumParticles, jseed) 

  # to circumvent the .jevalArray problem for rectangular arrays
  # we're now going to return flat arrays and use structure
  T = length(y)
  N = numParticles
  Nm = length(m0)
  Npsi = length(mPsi0)
  rlogWeights = aperm(structure(jResult$getLogWeights(), dim=c(N,T)), c(2,1))
  rstateMeans = aperm(structure(jResult$getStateMeans(), dim=c(Nm, N, T)), c(3,2,1))
  rstateCovs = aperm(structure(jResult$getStateCovs(), dim=c(Nm^2, N, T)), c(3,2,1))
  rpsiMeans = aperm(structure(jResult$getPsiMeans(), dim=c(Npsi, N, T)), c(3,2,1))
  rpsiCovs = aperm(structure(jResult$getPsiCovs(), dim=c(Npsi^2, N, T)), c(3,2,1))
  rsigma2Shapes = aperm(structure(jResult$getSigma2Shapes(), dim=c(N,T)), c(2,1))
  rsigma2Scales = aperm(structure(jResult$getSigma2Scales(), dim=c(N,T)), c(2,1))

  #rlogWeights = .jevalArray(jResult$getLogWeights(), simplify=T)
  #rstateMeans = .jevalArray(jResult$getStateMeans(), simplify=T)
  #rstateCovs = .jevalArray(jResult$getStateCovs(), simplify=T)
  #rpsiMeans = .jevalArray(jResult$getPsiMeans(), simplify=T)
  #rpsiCovs = .jevalArray(jResult$getPsiCovs(), simplify=T)
  #rsigma2Shapes = .jevalArray(jResult$getSigma2Shapes(), simplify=T)
  #rsigma2Scales = .jevalArray(jResult$getSigma2Scales(), simplify=T)

  return(list(logWeights = rlogWeights, 
          stateMeans = rstateMeans,
          stateCovs = rstateCovs,
          psiMeans = rpsiMeans,
          psiCovs = rpsiCovs,
          sigma2Shapes = rsigma2Shapes,
          sigma2Scales = rsigma2Scales
  ))
}

##
## Because .jevalArray with simplify=T isn't working for a 3d array...
##
## TODO, FIXME:  This is a horribly slow approach
##
#.hackedJevalArray = function(obj) {
#  res = sapply(.jevalArray(obj), function(x) sapply(x, .jevalArray, simplify=T), simplify="array")
#  res = aperm(res, c(3,2,1))
#  return(res)
#}
