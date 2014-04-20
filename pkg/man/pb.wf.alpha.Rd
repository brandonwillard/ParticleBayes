\name{pb.wf.alpha}
\alias{pb.wf.alpha}
\title{Computes the logarithm of the water-filling level, alpha, with which
samples, and their weights, are deterministically accepted in the
water-filling resample method.}
\usage{
pb.wf.alpha(logWeights, N, logWeightsSum = NULL)
}
\arguments{
  \item{logWeights}{vector of log weights}

  \item{N}{the number of samples.  Must be <= the length of
  logWeights.}

  \item{logWeightsSum}{total log weight, i.e. log sum of
  weights.  The value \code{NULL} implies that the sum is
  to be computed.}
}
\value{
numeric value of log(alpha)
}
\description{
Computes the logarithm of the water-filling level, alpha,
with which samples, and their weights, are
deterministically accepted in the water-filling resample
method.
}
\details{
For details concerning the algorithm see the paper by
Nicholas Polson, Brandon Willard (2014).
}
\examples{
x = seq(0, 10, 1)
  log.weights = dt(x, df=1, log=TRUE)
  log.weights.sum = log(sum(dt(x, df=1, log=FALSE)))
  log.alpha = pb.wf.alpha(log.weights, 8, log.weights.sum)
  print(log.alpha)
}
\author{
Brandon Willard \email{brandonwillard@gmail.com}
}
\references{
Nicholas G. Polson, Brandon Willard (2014), "Recursive
Bayesian Computation".
}
\seealso{
\code{\link{pb.wf.resample}}
}
\keyword{water-filling}

