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
#' @param logWeightsSum total log weight, i.e. log sum of weights.  The value 
#'  \code{NULL} implies that the sum is to be computed.
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
#' @export
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

#'
#' Given weights and an associated support, this function will 
#' perform water-filling resampling.  
#' 
#' @param logWeights vector of log weights
#' @param logWeightsSum total log weight, i.e. log sum of weights.  The value 
#'  \code{NULL} implies that the sum is to be computed.
#' @param support objects in the suppport that are associated with the \code{logWeights}.
#' @param N the number of samples.  Must be <= the length of logWeights.
#' @param seed seed for the random number generator.
#' @details For details concerning the algorithm see the paper by Nicholas Polson, Brandon Willard (2014).
#' @return A list containing each resampled object in the support and its associated weight. 
#' @references Nicholas G. Polson, Brandon Willard (2014), "Recursive Bayesian Computation".
#' @author Brandon Willard \email{brandonwillard@@gmail.com}
#' @examples 
#' x = seq(0, 10, 1)
#' log.weights = dt(x, df=1, log=T)
#' log.weights.sum = log(sum(dt(x, df=1, log=F)))
#' 
#' wf.result = water.filling.resample(log.weights, log.weights.sum, x, 8)
#' @keywords water-filling
#' @export 
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

#'
#' Particle filter implementing binomial logistic regression via a mixture
#' of normals and using water-filling.
#' 
#' @param y dependent variable vector
#' @param X explanatory variable matrix
#' @param m0 initial prior mean vector for the state variable
#' @param C0 initial prior covariance matrix for the state variable
#' @param G constant state evolution matrix
#' @param W constant state evolution covariance matrix
#' @param numParticles number of particles
#' @param waterFilling boolean value determining whether or not to perform water-filling
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
#' @examples TODO
#' @keywords water-filling 
#' @keywords logistic regression
#' @export 
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

#'
#' Particle filter for a Hidden Markov Model with categorical emissions. 
#' 
#' @param y dependent variable vector
#' @param X explanatory variable matrix
#' @param M number of 
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
#' @examples TODO
#' @keywords water-filling 
#' @keywords hidden markov model
#' @export 
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


#'
#' Particle filter for a gaussian AR(1) dynamic linear model.  States
#' have normal priors, AR(1) and state transition constant parameters
#' are a joint normal --psi-- distribution, and sigma2 has an inverse-gamma prior.
#' 
#' @param y dependent variable vector
#' @param m0 state prior mean vector
#' @param C0 state prior covariance matrix
#' @param mPsi0 vector of state transition constant and AR(1) prior means
#' @param CPsi0 state transition constant and AR(1) prior covariance matrix
#' @param sigma2Scale scale parameter for sigma2 inverse-gamma prior
#' @param sigma2Shape shape parameter for sigma2 inverse-gamma prior
#' @param FF observation matrix
#' @param numSubSamples sub samples for state variable
#' @param numParticles number of particles
#' @param seed seed for the random number generator.
#' @details For details concerning the algorithm see the paper by Nicholas Polson, Brandon Willard (2014).
#' @return returns a list containing matrices spanning time, particle and values, where values
#' are logWeights, stateMeans, stateCovs, psiMeans, psiCovs, sigma2Shapes, and sigma2Scales
#' @references 
#' Nicholas G. Polson, Brandon Willard (2014), "Recursive Bayesian Computation".
#' @author Brandon Willard \email{brandonwillard@@gmail.com}
#' @examples 
#' TODO
#' @keywords autoregressive
#' @keywords dynamic linear model 
#' @keywords dlm 
#' @keywords water-filling 
#' @keywords hidden markov model
#' @export  
wfAR <- function(y,  
    m0, C0 = diag(length(m0)), 
    mPsi0 = rep(0, 2*ncol(C0)), CPsi0 = diag(2*ncol(C0)), 
    sigma2Scale = 2, sigma2Shape = 1, FF, 
    numSubSamples = 3, numParticles = 1000, seed=NULL) {

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
  jmodelCovar = .jarray(as.matrix(W), dispatch=T)
  jnumParticles = as.integer(numParticles)
  jnumSubSamples = as.integer(numSubSamples)
  jFsPlAdapter = J("org.bitbucket.brandonwillard.particlebayes.radapters.GaussianArHLAdapter")
  jResult = jFsPlAdapter$batchUpdate(
      jm0, jC0, 
      jmPsi0, jCPsi0, 
      jsigma2Scale, jsigma2Shape,
      jF, 
      jnumSubSamples,
      .jarray(as.double(y)), 
      jnumParticles, jseed) 
  rlogWeights = .jevalArray(jResult$getLogWeights(), simplify=T)
  rstateMeans = .jevalArray(jResult$getStateMeans(), simplify=T)
  rstateCovs = .jevalArray(jResult$getStateCovs(), simplify=T)
  rpsiMeans = .jevalArray(jResult$getPsiMeans(), simplify=T)
  rpsiCovs = .jevalArray(jResult$getPsiCovs(), simplify=T)
  rsigma2Shapes = .jevalArray(jResult$getSigma2Shapes(), simplify=T)
  rsigma2Scales = .jevalArray(jResult$getSigma2Scales(), simplify=T)

  return(list(logWeights = rlogWeights, 
          stateMeans = rstateMeans,
          stateCovs = rstateCovs,
          psiMeans = rpsiMeans,
          psiCovs = rpsiCovs,
          sigma2Shapes = rsigma2Shapes,
          sigma2Scales = rsigma2Scales
  ))
}

