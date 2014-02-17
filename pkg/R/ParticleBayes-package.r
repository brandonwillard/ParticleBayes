#'
#' @name ParticleBayes-package
#' @aliases particlebayes-package, ParticleBayes
#' @title Particle filters for parameter learning
#' @description   
#'   ParticleBayes is an R package that contains a collection of particle filters 
#'   for a subset of hierarchical bayesian models, with a focus on sequentially learning model 
#'   parameters alongside state variables
#'   
#'   Most models are formulated as dynamic linear models, or mixtures thereof, 
#'   although some support for hidden markov models exists.
#'   The models also implement 
#'   \itemize{
#'     \item Water-filling resampling
#'     \item Parameter learning, via Particle Learning where applicable
#'   }
#'   Currently, the implemented filters cover two response types:
#'       
#'         Multinomial/Categorical with a logistic link function
#'   \itemize{
#'     \item 10-component normal mixture approximation
#'     \item TBD
#'     \item Multivariate Gaussian
#'   }
#'   and, naturally, their univariate equivalents.
#' @details
#'   \tabular{ll}{
#'     Package: \tab ParticleBayes\cr
#'     Type: \tab Package\cr
#'     Version: \tab 0.1\cr
#'     Date: \tab 2014-01-09\cr
#'     License: \tab LGPL-3\cr
#'   }  
#' @author
#'   Nicholas Polson
#'   Brandon Willard
#'   Maintainer: Brandon Willard\email{brandonwillard@@gmail.com}
#' @references
#'   Nicholas G. Polson, Brandon Willard (2014), 
#'   "Recursive Bayesian Computation".
#' @keywords package
#'
#' @import rJava
#'
#' @docType package
NULL


