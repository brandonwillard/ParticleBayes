\name{pb.wf.resample}
\alias{pb.wf.resample}
\title{Given weights and an associated support, this function will
perform water-filling resampling.}
\usage{
pb.wf.resample(logWeights, N, support = NULL, logWeightsSum = NULL,
  seed = NULL)
}
\arguments{
  \item{logWeights}{vector of log weights}

  \item{support}{objects in the suppport that are
  associated with the \code{logWeights}.}

  \item{N}{the number of samples.  Must be <= the length of
  logWeights.}

  \item{logWeightsSum}{total log weight, i.e. log sum of
  weights.  The value \code{NULL} implies that the sum is
  to be computed.}

  \item{seed}{seed for the random number generator.}
}
\value{
A list containing each resampled object in the support and
its associated weight.
}
\description{
Given weights and an associated support, this function will
perform water-filling resampling.
}
\details{
For details concerning the algorithm see the paper by
Nicholas Polson, Brandon Willard (2014).
}
\examples{
x = seq(0, 10, 1)
log.weights = dt(x, df=1, log=TRUE)

wf.result = pb.wf.resample(log.weights, 8, support=x)
}
\author{
Brandon Willard \email{brandonwillard@gmail.com}
}
\references{
Nicholas G. Polson, Brandon Willard (2014), "Recursive
Bayesian Computation".
}
\keyword{water-filling}

